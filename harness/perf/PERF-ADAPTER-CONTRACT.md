# 性能赛道（LongMemEval-M）：PerfAdapter 接口契约（给 Qoder）

目标：为 mem0 / graphiti / cognee 各写 with/without 两个 **PerfAdapter**（共 6 个），
注册进 run_perf，跑通参考基线校验。harness 侧（runner/计时/recall/GT）已就绪，勿改。

**与 LoCoMo adapter 的本质区别**：性能赛道做 **search 层直注**——不走各系统的 LLM
抽取摄入，而是把预计算好的向量/文本/规范共现图直接载入后端，再走各系统的检索路径跑
4 类 query。绕过的是"抽取/建图"，**各系统的检索逻辑（mem0 加性融合、graphiti 三路
hybrid、cognee retriever）仍在被测**（benchmark-plan.md §3.4）。

## 1. 接口（harness/perf/base.py，已固定）

```python
class PerfAdapter(ABC):
    name: str                       # 如 "mem0-neug" / "graphiti-neo4j"
    supported_classes: frozenset    # 支持的查询类子集；不支持的由 runner 标 N/A

    def setup(self, work_dir: str): ...
    def load(self, corpus: PerfCorpus): ...
        # 直注预计算语料：corpus.embeddings (n,1024) + corpus.texts + corpus.graph_edges
        # 不走 LLM 抽取。全系统同一份数据，唯一变量是存储引擎
    # 四类查询，均返回按相关度排序的 session_id 列表：
    def query_vector(self, qvec: np.ndarray, top_k: int) -> list[str]: ...
    def query_fts(self, keywords: list[str], top_k: int) -> list[str]: ...
    def query_hybrid(self, qvec: np.ndarray, keywords: list[str], top_k: int) -> list[str]: ...
    def query_graph(self, seed_session_id: str, max_nodes: int) -> list[str]: ...
    def teardown(self): ...         # 必须正常 close（NeuG checkpoint 纪律）

# PerfCorpus(session_ids, texts, embeddings, graph_edges)
#   session_ids/texts/embeddings 按行对齐；graph_edges 节点编号 = session 行号
```

不支持的查询类保持 `raise NotImplementedError`，并把该类从 `supported_classes` 去掉。

## 2. 数据（已就绪，勿重新生成；均在 data/processed/perf/）

| 文件 | 内容 |
|---|---|
| `corpus_sessions.jsonl` | 51661 session：`{session_id, date, text}` |
| `embed_text-embedding-v3.npy` | (51661, 1024) float32，已 L2 归一 |
| `graph_edges.npy` | (2799244, 2) int32 无向边表，节点=行号（规范共现图） |
| `query_vectors.npy` | (50, 1024)，vector/hybrid 共用此 50 题顺序 |
| `queries_{vector,fts,hybrid,graph}.jsonl` | 各 50 题 + 暴力 ground truth |
| `manifest.json` | 数据集元信息与可比性声明 |

runner 已用 `load_corpus()/load_queries()` 读好并经 `PerfCorpus` 传入，adapter 无需自读。

## 3. 六条硬约束

1. **同一份数据**：load 必须原样载入 corpus 的 embeddings/texts/graph_edges，不得
   各自重新 embedding 或改建图——否则 with/without 不可比。
2. **绕过 LLM 抽取**：load 不得触发任何 LLM 调用（不调 add_episode/cognify 的抽取路径）。
   走各系统的**存储层写入**（vector store insert / 图库 Cypher 直写 / cognee 底层图写入）。
3. **query 走系统检索路径**：query_* 调各系统自己的检索逻辑（不是裸 SQL/裸 backend API），
   这样系统检索架构差异（如 cognee 整图拉内存暴力排序）才被测到。
4. **返回 session_id**：四类查询都返回 corpus 的 session_id 字符串（不是内部 node id），
   runner 用 session_id 对比 GT。load 时自行维护 行号↔session_id↔后端主键 的映射。
5. **N/A 矩阵**：mem0 v2 无图 → `graph_multihop` 标 N/A；graphiti/cognee 四类全支持。
   照 2.4 事实表如实标，N/A 本身是能力完整性测量。
6. **teardown 正常关闭**：NeuG 连接必须 close（checkpoint 纪律）。

## 4. 各系统映射参考（load 机制是实现难点，给出方向）

| 系统 | load（直注，绕 LLM） | query |
|---|---|---|
| mem0 | vector_store.insert(预计算向量 + text payload + session_id)；FTS 若后端支持则建 BM25 索引 | query_vector/hybrid 走 mem0 search 的向量+关键词融合；graph N/A |
| graphiti | 图库直写节点(带向量)+共现边，建 vector/fulltext 索引（用 OpenAIGenericClient 无关，此处无 LLM） | 走 graphiti hybrid search（vector+fulltext+BFS） |
| cognee | 底层图 + 向量直写（绕 cognify 的 LLM 抽取），dataset 隔离 | 走 cognee retriever（HYBRID / CHUNKS_LEXICAL / 图遍历），取检索上下文非 LLM 补全 |

关键难点：各系统"不触发 LLM 的存储层写入"入口不同，需逐个探明（mem0 的
vector_store 抽象、graphiti 的 driver 直写、cognee 的 graphDB provider 写入）。
若某系统的检索路径强制耦合 LLM（无法纯检索），在 adapter 内短路该步并记录到
`results/perf/<name>/NOTES.md`，不要硬编造结果。

## 5. 交付物与验收

- 文件：`harness/perf/adapters/{mem0,graphiti,cognee}_arm.py`，每个文件注册
  with/without 两个 name（如 `mem0-neug`、`mem0-qdrant`）
- 注册方式：`run_perf.py` 的 `_load_adapters()` 里 import 即生效
- 每臂冒烟：
  ```bash
  cd ~/Documents/projects/neug-memory-benchmark
  set -a && source .env && set +a
  .venv/bin/python -m harness.perf.run_perf --adapter <name>
  ```
- 通过标准：load 无异常、四类查询（支持项）返回非空 session_id、
  summary.json 产出、recall 合理（vector/fts/hybrid 对照暴力 GT，with/within 可比）
- **参考基线**：`bruteforce-numpy` 已验证四类 recall=1.0（见 results/perf/bruteforce-numpy/），
  用作你实现时的正确性对照；真实后端 vector recall@10 ≥ 0.95 为合格线（plan §3.2）。

## 6. 已知坑

- graph GT 按 hop 独立截断（hop1[:50]/hop2[:100]）；query_graph 返回完整邻域即 ⊇ GT，
  recall 口径自洽（勿用单一全局 top-N 截断，会丢 hop2，见 bruteforce.py 注释）
- fts GT 是 AND 语义（全含词集合）；query_fts 若后端是 OR/BM25，recall 反映其 AND 贴合度
- NeuG 方言见 results/NEUG-DIALECT-NOTES.md（parameters 传参、bm25 ORDER BY score ASC LIMIT、
  IN 绑参段错误用内联）
- mem0 过滤语法：扁平字段 + 操作符（见 locomo 契约 §6）
