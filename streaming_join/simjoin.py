"""Streaming full-outer similarity join built on datasketch MinHashLSH.

Design contract (see module-level README / tests for the exercised cases):

* The *right* stream is fully indexed once in a :class:`MinHashLSH`; original
  right records and their raw token sets are retained for exact verification.
* The *left* stream is consumed one row at a time.  Every output row is
  yielded before the next left row is pulled, so callers can consume the
  join result line by line without buffering the left stream or the whole
  result set.
* LSH only supplies approximate candidates.  Every candidate is re-checked
  with the exact Jaccard similarity computed from the *original* token
  sets; pairs below the threshold are dropped.  The approximate recall of
  native LSH candidates is therefore preserved; global exactness is not
  claimed.
* Duplicate business IDs and byte-identical rows are never merged: each
  input row gets its own unique internal key.
* Rows whose tokens are missing (``None`` / field absent / accessor
  failure) never match anything and are emitted as unmatched rows.
* Two empty token sets have Jaccard similarity 1.0 and match each other.
  Empty sets are kept out of the LSH (an empty MinHash would collide with
  every other empty signature) and handled by an exact special path.
* If the left iterator raises, the exception propagates out of the
  generator immediately and the unmatched right-side tail is *not*
  emitted as a normal completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from datasketch import MinHash, MinHashLSH

__all__ = [
    "JoinResult",
    "token_set",
    "jaccard",
    "build_right_index",
    "streaming_full_outer_similarity_join",
]

TokenAccessor = Union[str, Callable[[dict], Any]]
_MISSING = object()


@dataclass(frozen=True)
class JoinResult:
    """One output row of the full-outer similarity join.

    Attributes:
        kind: ``"matched"`` for a confirmed left/right pair, ``"left_only"``
            for a left row with no confirmed match, ``"right_only"`` for a
            right row that no left row matched.
        left: original left record (only for ``matched`` / ``left_only``).
        right: original right record (only for ``matched`` / ``right_only``).
        score: exact Jaccard similarity of the original token sets; only
            present for ``matched`` rows.
        left_index / right_index: zero-based position of the row in its
            input stream.  Unique per physical row, so duplicate business
            IDs and identical rows stay distinguishable.
    """

    kind: str
    left: Optional[dict] = None
    right: Optional[dict] = None
    score: Optional[float] = None
    left_index: Optional[int] = None
    right_index: Optional[int] = None

    def to_dict(self) -> dict:
        """JSON-serialisable view of the result row."""
        out: dict[str, Any] = {"kind": self.kind}
        if self.kind in ("matched", "left_only"):
            out["left_index"] = self.left_index
            out["left"] = self.left
        if self.kind in ("matched", "right_only"):
            out["right_index"] = self.right_index
            out["right"] = self.right
        if self.kind == "matched":
            out["score"] = self.score
        return out


def token_set(record: dict, accessor: TokenAccessor = "tokens") -> Any:
    """Extract the deduplicated token set from a record.

    Returns the sentinel :data:`_MISSING` when the record carries no
    usable tokens (absent field, explicit ``None``, or accessor raising).
    A present-but-empty token field yields an empty ``set``, which is a
    valid (matchable) value distinct from missing.
    """
    if callable(accessor):
        try:
            raw = accessor(record)
        except (KeyError, TypeError, AttributeError):
            return _MISSING
    else:
        raw = record.get(accessor, _MISSING)
    if raw is None or raw is _MISSING:
        return _MISSING
    if isinstance(raw, set):
        return set(raw)
    try:
        return set(raw)
    except TypeError:
        # A single non-iterable token is treated as a one-token field.
        return {raw}


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity of two sets.

    Per the join contract, two empty sets have similarity 1.0; an empty
    set against a non-empty set has similarity 0.0.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _encode(token: Any) -> bytes:
    if isinstance(token, bytes):
        return token
    return str(token).encode("utf-8")


def _minhash(tokens: set, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([_encode(t) for t in tokens])
    return mh


def build_right_index(
    right_records: Iterable[dict],
    threshold: float,
    token_field: TokenAccessor = "tokens",
    num_perm: int = 128,
) -> tuple[MinHashLSH, list[dict], list[Optional[set]], set[int], set[int]]:
    """Consume the entire right stream and build the LSH index.

    Returns the index together with everything the streaming phase needs:
    original records, their token sets (``None`` for missing-token rows),
    the internal keys of empty-token rows and of missing-token rows.

    Right rows are materialised here by necessity -- the LSH index is
    built over the whole right side before the left side is touched.
    """
    try:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError as exc:
        # At thresholds at/near 1.0 the optimal params can give b < 2,
        # which MinHashLSH rejects. Fall back to explicit bands: sets with
        # Jaccard 1.0 have identical signatures that collide in every
        # band, so recall for threshold-passing pairs is unaffected; the
        # extra (looser) candidates are removed by exact verification.
        if "bands are too small" not in str(exc):
            raise
        lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
            params=(2, num_perm // 2),
        )
    records: list[dict] = []
    token_sets: list[Optional[set]] = []
    empty_keys: set[int] = set()
    missing_keys: set[int] = set()

    for key, record in enumerate(right_records):
        records.append(record)
        tokens = token_set(record, token_field)
        if tokens is _MISSING:
            # Never inserted into the LSH: missing tokens cannot match.
            token_sets.append(None)
            missing_keys.add(key)
            continue
        token_sets.append(tokens)
        if not tokens:
            # Empty MinHash signatures collide indiscriminately; handle
            # empty sets on an exact path instead.
            empty_keys.add(key)
            continue
        lsh.insert(key, _minhash(tokens, num_perm), check_duplication=False)

    return lsh, records, token_sets, empty_keys, missing_keys


def streaming_full_outer_similarity_join(
    left_records: Iterable[dict],
    right_records: Iterable[dict],
    threshold: float,
    token_field: TokenAccessor = "tokens",
    num_perm: int = 128,
) -> Iterator[JoinResult]:
    """Yield the full-outer similarity join of two record streams.

    Args:
        left_records: iterable of left records; pulled one row at a time.
            May be a lazy generator -- it is never materialised.
        right_records: iterable of right records; consumed fully once to
            build the :class:`MinHashLSH` index.
        threshold: minimum exact Jaccard similarity for a confirmed pair.
        token_field: record key holding the tokens, or a callable taking
            a record and returning the tokens.
        num_perm: number of MinHash permutation functions (same value for
            both sides).

    Yields:
        :class:`JoinResult` rows in streaming order: confirmed matches and
        unmatched left rows are emitted as each left row is processed;
        unmatched right rows (if any) are emitted only after the left
        stream has been consumed *without error*.
    """
    (
        lsh,
        right_records_list,
        right_token_sets,
        empty_right_keys,
        missing_right_keys,
    ) = build_right_index(right_records, threshold, token_field, num_perm)

    matched_right_keys: set[int] = set()

    for left_index, left_record in enumerate(left_records):
        left_tokens = token_set(left_record, token_field)

        if left_tokens is _MISSING:
            # Missing tokens never match anything.
            yield JoinResult(
                kind="left_only", left=left_record, left_index=left_index
            )
            continue

        if not left_tokens:
            # Exact empty-set path: empty matches empty with score 1.0.
            matched = False
            if 1.0 >= threshold:
                for rkey in empty_right_keys:
                    matched_right_keys.add(rkey)
                    matched = True
                    yield JoinResult(
                        kind="matched",
                        left=left_record,
                        right=right_records_list[rkey],
                        score=1.0,
                        left_index=left_index,
                        right_index=rkey,
                    )
            if not matched:
                yield JoinResult(
                    kind="left_only", left=left_record, left_index=left_index
                )
            continue

        # Non-empty left row: ask LSH for approximate candidates, then
        # verify each one against the original token sets exactly.
        candidates = lsh.query(_minhash(left_tokens, num_perm))
        any_match = False
        for rkey in candidates:
            right_tokens = right_token_sets[rkey]
            if right_tokens is None:
                continue
            score = jaccard(left_tokens, right_tokens)
            if score >= threshold:
                matched_right_keys.add(rkey)
                any_match = True
                yield JoinResult(
                    kind="matched",
                    left=left_record,
                    right=right_records_list[rkey],
                    score=score,
                    left_index=left_index,
                    right_index=rkey,
                )
        if not any_match:
            yield JoinResult(
                kind="left_only", left=left_record, left_index=left_index
            )

    # Reached only when the left iterator completed without raising.
    # On a left-stream exception this tail is never produced.
    for rkey, right_record in enumerate(right_records_list):
        if rkey not in matched_right_keys:
            yield JoinResult(
                kind="right_only", right=right_record, right_index=rkey
            )
