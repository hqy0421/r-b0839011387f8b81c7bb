"""Tests for the streaming full-outer similarity join."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simjoin import (  # noqa: E402
    JoinResult,
    jaccard,
    streaming_full_outer_similarity_join,
    token_set,
)

MISSING = object()


def run_join(left, right, threshold=0.5, **kwargs):
    return list(
        streaming_full_outer_similarity_join(
            iter(left), right, threshold=threshold, **kwargs
        )
    )


def rec(rec_id, tokens=MISSING, **extra):
    """Build a record; tokens omitted entirely when not supplied."""
    out = {"id": rec_id, **extra}
    if tokens is not MISSING:
        out["tokens"] = tokens
    return out


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_jaccard_basic_and_empty_rules():
    assert jaccard(set(), set()) == 1.0
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b", "c"}, {"a", "b", "d"}) == pytest.approx(2 / 4)
    assert jaccard({"a"}, {"b"}) == 0.0


def test_token_set_missing_vs_empty():
    sentinel = token_set(rec("x"))  # field absent
    assert sentinel is not None and sentinel is not set()
    assert token_set(rec("x", tokens=None)) is sentinel
    assert token_set(rec("x", tokens=[])) == set()
    assert token_set(rec("x", tokens=["a", "a", "b"])) == {"a", "b"}
    assert token_set({"id": "x"}, lambda r: r["tokens"]) is sentinel


def test_token_set_accessor_failure_is_missing():
    sentinel = token_set(rec("x"))
    assert token_set({"id": "x"}, lambda r: r["tokens"]) is sentinel
    # Attribute access on a plain dict fails -> treated as missing.
    assert token_set({"id": "x"}, lambda r: r.missing) is sentinel


# ---------------------------------------------------------------------------
# Matching behaviour
# ---------------------------------------------------------------------------


def test_identical_sets_match_with_score_one():
    left = [rec("L1", ["a", "b", "c", "d"])]
    right = [rec("R1", ["a", "b", "c", "d"])]
    results = run_join(left, right, threshold=0.5)
    assert len(results) == 1
    r = results[0]
    assert r.kind == "matched"
    assert r.score == 1.0
    assert r.left["id"] == "L1" and r.right["id"] == "R1"
    assert r.left_index == 0 and r.right_index == 0


def test_partial_overlap_score_is_exact_jaccard():
    left = [rec("L1", ["a", "b", "c", "d"])]
    right = [rec("R1", ["a", "b", "c", "e"])]
    results = run_join(left, right, threshold=0.4)
    assert len(results) == 1
    assert results[0].kind == "matched"
    assert results[0].score == pytest.approx(3 / 5)


def test_below_threshold_candidate_is_rejected_and_become_unmatched():
    left = [rec("L1", ["a", "b", "c", "d"])]
    right = [rec("R1", ["x", "y", "z", "w"])]
    results = run_join(left, right, threshold=0.5)
    kinds = sorted(r.kind for r in results)
    assert kinds == ["left_only", "right_only"]


def test_threshold_boundary_is_inclusive():
    # Jaccard = 0.5 exactly: 2 shared / 4 union.
    left = [rec("L1", ["a", "b", "c"])]
    right = [rec("R1", ["a", "b", "d"])]
    hit = run_join(left, right, threshold=0.5)
    assert any(r.kind == "matched" for r in hit)
    # Just above: LSH may or may not surface a candidate, but if it does
    # the exact check must reject; no matched row may ever be emitted.
    strict = run_join(left, right, threshold=0.51)
    assert all(r.kind != "matched" for r in strict)


def test_one_left_matches_several_right_rows_all_emitted():
    left = [rec("L1", ["a", "b", "c", "d", "e"])]
    right = [
        rec("R1", ["a", "b", "c", "d", "e"]),
        rec("R2", ["a", "b", "c", "d", "f"]),
        rec("R3", ["z", "y", "x", "w", "v"]),
    ]
    results = run_join(left, right, threshold=0.5)
    matched = [r for r in results if r.kind == "matched"]
    assert {r.right["id"] for r in matched} == {"R1", "R2"}
    right_only = [r for r in results if r.kind == "right_only"]
    assert {r.right["id"] for r in right_only} == {"R3"}
    # No left_only: the left row had confirmed matches.
    assert all(r.kind != "left_only" for r in results)


# ---------------------------------------------------------------------------
# Duplicate IDs and identical rows are never merged
# ---------------------------------------------------------------------------


def test_duplicate_business_ids_and_identical_rows_kept_separate():
    left = [
        rec("DUP", ["a", "b", "c", "d"]),
        rec("DUP", ["a", "b", "c", "d"]),  # identical row, same business id
    ]
    right = [
        rec("DUP", ["a", "b", "c", "d"]),
        rec("DUP", ["a", "b", "c", "d"]),
    ]
    results = run_join(left, right, threshold=0.5)
    matched = [r for r in results if r.kind == "matched"]
    # Cartesian product of the two duplicates: 4 pairs.
    assert len(matched) == 4
    pairs = {(r.left_index, r.right_index) for r in matched}
    assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(r.score == 1.0 for r in matched)


# ---------------------------------------------------------------------------
# Missing tokens and empty sets
# ---------------------------------------------------------------------------


def test_missing_tokens_never_match_on_either_side():
    left = [
        rec("L-ok", ["a", "b", "c"]),
        rec("L-missing"),  # field absent
        rec("L-null", tokens=None),
    ]
    right = [
        rec("R-ok", ["a", "b", "c"]),
        rec("R-missing"),
        rec("R-null", tokens=None),
    ]
    results = run_join(left, right, threshold=0.5)
    matched = [r for r in results if r.kind == "matched"]
    assert len(matched) == 1
    assert matched[0].left["id"] == "L-ok" and matched[0].right["id"] == "R-ok"
    left_only = {r.left["id"] for r in results if r.kind == "left_only"}
    right_only = {r.right["id"] for r in results if r.kind == "right_only"}
    assert left_only == {"L-missing", "L-null"}
    assert right_only == {"R-missing", "R-null"}


def test_two_empty_sets_match_with_score_one():
    left = [rec("L-empty", [])]
    right = [rec("R-empty", [])]
    results = run_join(left, right, threshold=0.5)
    assert len(results) == 1
    r = results[0]
    assert r.kind == "matched" and r.score == 1.0
    assert r.left["id"] == "L-empty" and r.right["id"] == "R-empty"


def test_empty_does_not_match_nonempty_and_vice_versa():
    left = [rec("L-empty", []), rec("L-full", ["a", "b"])]
    right = [rec("R-full", ["a", "b"]), rec("R-empty", [])]
    results = run_join(left, right, threshold=0.5)
    matched = [r for r in results if r.kind == "matched"]
    assert {(r.left["id"], r.right["id"]) for r in matched} == {
        ("L-full", "R-full"),
        ("L-empty", "R-empty"),
    }
    # Every row matched exactly once; nothing left over.
    assert len(results) == 2


def test_empty_left_without_empty_right_is_left_only():
    left = [rec("L-empty", [])]
    right = [rec("R-full", ["a", "b"]), rec("R-missing")]
    results = run_join(left, right, threshold=0.5)
    assert [r.kind for r in results] == ["left_only", "right_only", "right_only"]


def test_empty_set_matching_respects_threshold():
    # Empty/empty similarity is 1.0, so a threshold > 1.0 is impossible;
    # threshold == 1.0 must still match.
    left = [rec("L", [])]
    right = [rec("R", [])]
    results = run_join(left, right, threshold=1.0)
    assert results[0].kind == "matched"


def test_threshold_one_matches_identical_nonempty_sets_only():
    # MinHashLSH cannot build b>=2 bands at threshold 1.0; the engine
    # falls back to explicit bands and still finds exact duplicates.
    left = [rec("L1", ["a", "b", "c", "d"]), rec("L2", ["a", "b", "c", "x"])]
    right = [rec("R1", ["a", "b", "c", "d"]), rec("R2", ["a", "b", "c", "y"])]
    results = run_join(left, right, threshold=1.0)
    matched = [(r.left["id"], r.right["id"]) for r in results if r.kind == "matched"]
    assert matched == [("L1", "R1")]
    assert {r.left["id"] for r in results if r.kind == "left_only"} == {"L2"}
    assert {r.right["id"] for r in results if r.kind == "right_only"} == {"R2"}


# ---------------------------------------------------------------------------
# Full-outer coverage and streaming semantics
# ---------------------------------------------------------------------------


def test_full_outer_every_unmatched_side_emitted_once():
    left = [rec("L1", ["a", "b", "c"]), rec("L2", ["p", "q", "r"])]
    right = [rec("R1", ["a", "b", "c"]), rec("R2", ["x", "y", "z"])]
    results = run_join(left, right, threshold=0.5)
    matched = [r for r in results if r.kind == "matched"]
    assert len(matched) == 1 and matched[0].right["id"] == "R1"
    assert {r.left["id"] for r in results if r.kind == "left_only"} == {"L2"}
    assert {r.right["id"] for r in results if r.kind == "right_only"} == {"R2"}


def test_left_stream_is_consumed_lazily_row_by_row():
    pulled = []

    def left_gen():
        for i in range(5):
            pulled.append(i)
            yield rec(f"L{i}", [f"t{i}", f"u{i}", f"v{i}", f"w{i}"])

    right = [rec("R0", ["t0", "u0", "v0", "w0"])]
    gen = streaming_full_outer_similarity_join(
        left_gen(), right, threshold=0.5
    )
    first = next(gen)
    # One row pulled, one result already available: no full buffering.
    assert pulled == [0]
    assert first.kind == "matched" and first.left["id"] == "L0"
    second = next(gen)
    assert pulled == [0, 1]
    assert second.kind == "left_only"
    # Drain the rest.
    rest = list(gen)
    assert pulled == [0, 1, 2, 3, 4]
    # R0 was matched by L0 so no right_only tail appears.
    assert all(r.kind != "right_only" for r in rest)


def test_right_only_tail_comes_after_all_left_rows():
    left = [rec("L1", ["a", "b", "c"])]
    right = [rec("R1", ["x"]), rec("R2", ["y"]), rec("R3", ["z"])]
    gen = streaming_full_outer_similarity_join(iter(left), right, threshold=0.5)
    first = next(gen)
    assert first.kind == "left_only"
    tail = list(gen)
    assert [r.kind for r in tail] == ["right_only", "right_only", "right_only"]
    assert [r.right_index for r in tail] == [0, 1, 2]


def test_left_exception_aborts_without_emitting_right_tail():
    class Boom(RuntimeError):
        pass

    def broken_left():
        yield rec("L1", ["a", "b", "c"])
        raise Boom("left stream failed")

    right = [rec("R1", ["x", "y", "z"]), rec("R2", ["p", "q", "r"])]
    gen = streaming_full_outer_similarity_join(broken_left(), right, threshold=0.5)
    first = next(gen)
    assert first.kind == "left_only"
    with pytest.raises(Boom):
        list(gen)  # the right-only tail must never be produced


def test_original_records_are_preserved_unmodified():
    original_left = rec("L1", ["a", "b", "c", "d"], payload={"n": 1})
    original_right = rec("R1", ["a", "b", "c", "d"], payload={"n": 2})
    results = run_join([original_left], [original_right], threshold=0.5)
    m = results[0]
    assert m.left is original_left and m.right is original_right
    assert m.left["payload"] == {"n": 1}


# ---------------------------------------------------------------------------
# Approximate recall on a larger population
# ---------------------------------------------------------------------------


def test_approximate_recall_high_for_near_identical_sets():
    # 40 pairs sharing ~80% of their tokens; LSH at threshold 0.5 should
    # retrieve the large majority, and every emitted score must be exact.
    base = [f"tok{i}" for i in range(50)]
    left, right, expected_pairs = [], [], set()
    for i in range(40):
        l_tokens = base[:40] + [f"L{i}-{j}" for j in range(10)]
        r_tokens = base[:32] + [f"L{i}-{j}" for j in range(8)] + [f"R{i}"]
        left.append(rec(f"L{i}", l_tokens))
        right.append(rec(f"R{i}", r_tokens))
        expected_pairs.add((f"L{i}", f"R{i}"))
    # Distractor rows.
    for i in range(20):
        right.append(rec(f"D{i}", [f"d{i}-{j}" for j in range(50)]))

    results = run_join(left, right, threshold=0.5)
    found = {
        (r.left["id"], r.right["id"])
        for r in results
        if r.kind == "matched"
    }
    hits = len(found & expected_pairs)
    assert hits >= 36  # native approximate recall, not a guarantee of 40/40
    # Exact verification: every emitted pair really clears the threshold.
    for r in results:
        if r.kind == "matched":
            assert r.score >= 0.5
            assert r.score == jaccard(
                set(r.left["tokens"]), set(r.right["tokens"])
            )


def test_result_to_dict_shape():
    left = [rec("L1", ["a", "b"]), rec("L2", ["m", "n"])]
    right = [rec("R1", ["a", "b"]), rec("R2", ["x", "y"])]
    results = run_join(left, right, threshold=0.5)
    dicts = [r.to_dict() for r in results]
    matched = next(d for d in dicts if d["kind"] == "matched")
    assert set(matched) == {"kind", "left_index", "left", "right_index", "right", "score"}
    lo = next(d for d in dicts if d["kind"] == "left_only")
    assert set(lo) == {"kind", "left_index", "left"}
    ro = next(d for d in dicts if d["kind"] == "right_only")
    assert set(ro) == {"kind", "right_index", "right"}
    assert all(isinstance(r, JoinResult) for r in results)
