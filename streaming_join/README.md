# 流式全外相似连接（Streaming Full-Outer Similarity Join）

基于 `datasketch.MinHashLSH` 的两批 token 采集记录的**流式全外连接**。
右侧整批建立 MinHashLSH 索引，左侧逐行消费、逐行产出结果。

## 设计契约

| 需求 | 实现方式 |
| --- | --- |
| 右侧原记录进 MinHashLSH 索引 | `build_right_index()` 用行号作为唯一内部 key 建索引，同时保留原记录和原始 token 集合 |
| 左侧单次逐行取候选 | 生成器逐行 `lsh.query()`，不物化左流 |
| 原始集合精确 Jaccard 核实 | 每个候选都用原始 token 集合计算精确 Jaccard，低于阈值的候选丢弃；输出分数即精确值 |
| 全外连接 | 确认匹配输出 `matched`；无确认匹配的左行输出 `left_only`；未被任何左行匹配的右行输出 `right_only`（尾部） |
| 重复业务 ID / 完全相同行不合并 | 内部 key 是物理行号，与业务 ID 无关；重复行产生笛卡尔积匹配对 |
| 缺失 tokens 不匹配 | 字段缺失、显式 `null`、访问器异常统一视为 missing，不进 LSH、永不匹配，按单侧记录输出 |
| 两个空集合相似度为 1 | 空集合不进 LSH（空 MinHash 签名会无差别碰撞），走精确专用路径：空-空 = 1.0，空-非空 = 0.0 |
| 结果可逐行消费 | 纯生成器；每条结果在拉取下一左行之前 yield；不攒左流、不攒结果 |
| 左流异常不输出右侧尾部 | 右尾只在左迭代器**正常耗尽**后产出；左迭代器抛异常时异常直接传播，生成器终止，无 `right_only` |
| 沿用近似召回 | LSH 只负责候选生成，召回特性与原生 MinHashLSH 一致；精确核验只剔除误报，不声称全局精确连接 |

## 输出行格式（JSONL）

```json
{"kind": "matched", "left_index": 0, "left": {...原记录...}, "right_index": 1, "right": {...原记录...}, "score": 0.667}
{"kind": "left_only", "left_index": 3, "left": {...原记录...}}
{"kind": "right_only", "right_index": 4, "right": {...原记录...}}
```

## 用法

### CLI

```bash
python3 -I cli.py left.jsonl right.jsonl --threshold 0.5 --token-field tokens --num-perm 128
# 左侧支持标准输入：
cat left.jsonl | python3 -I cli.py - right.jsonl
```

每行一个 JSON 对象，输出同样每行一个 JSON 对象并逐行 flush，可直接进 shell 管道。
下游提前关闭管道（如 `head`）时静默退出。

### 库

```python
from simjoin import streaming_full_outer_similarity_join

for result in streaming_full_outer_similarity_join(
    left_records,          # 任意可迭代对象，惰性生成器即可，不会被 list() 物化
    right_records,         # 右侧可迭代对象，建索引时消费一次
    threshold=0.5,
    token_field="tokens",  # 或 callable: lambda r: r["meta"]["tokens"]
    num_perm=128,
):
    if result.kind == "matched":
        ...  # result.left / result.right / result.score
```

## 边界说明

- `threshold=1.0`（及极高阈值）时 MinHashLSH 的最优参数会出现 `b < 2` 而拒绝构造；
  引擎回退到显式 `params=(2, num_perm//2)`。Jaccard 为 1.0 的集合签名完全相同、
  在任意 band 必碰撞，真匹配召回不受影响，放宽的候选由精确核验滤除。
- 内存开销在右侧：LSH 哈希表、右侧原记录及其 token 集合。左侧仅保留当前行，
  外加一个记录已匹配右行 key 的集合（用于全外连接尾部判定）。

## 测试

```bash
python3 -m pytest test_simjoin.py -v
```

22 个用例覆盖：精确分数、阈值边界（含 1.0）、一对多匹配、重复 ID/相同行不合并、
缺失与空集合语义、全外覆盖、逐行惰性、左流异常抑制右尾、原记录保留、
近似召回（40 对近重复集合）等。
