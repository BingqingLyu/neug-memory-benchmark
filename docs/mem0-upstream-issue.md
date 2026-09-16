<!--
mem0 上游 feature-request issue 稿（中文订正版，待确认后转英文终稿）
状态：已并入 pgvector 结果 + B 口径统一。相对初版（对话 LINE 344）的关键订正：
  ① benchmark 两臂 → **三臂公平基线**（neug / qdrant-server / pgvector，都建了真索引）
  ② "比 qdrant local 快 280×/205×" 虚高倍数 → 对两个生产级基线的诚实倍数（vector 快 server 8.1× / 快 pgvector 414×）
  ③ fts_keyword 用修正后的 exact top-k(neug) vs 剪枝 top-k(qdrant) vs AND 语义(pgvector) 三套机制取舍框架
  ④ pgvector vector 1093ms 用 EXPLAIN ANALYZE 铁证诚实标注根因（HNSW 不 filterable + mem0 总带过滤 + planner 误估 → Seq Scan），并声明单 user_id 最坏情况边界、不针对性调优
  ⑤ doc 侧统一 lemmatized-text（B 口径），还原 mem0 core add() 对所有 backend 都存 lemmatized text 的真实行为
  ⑥ 后端能力统计核实更正：上游现有 **25** 个后端（排除本 PR 的 neug）、**15** 个 override keyword_search（原稿 26/16 误含了我们自己的 neug.py）
  ⑦ 图能力框架更正：不再把“原生图索引 0/N”当后端缺陷——mem0 图记忆是 core 层内置、provider-agnostic（设计使然）；neug 三种原生都支持，内置图记忆的 entity_store 也可由 neug 承载，且 neug 引擎级原生图为未来 typed 关系升级留出空间
  ⑧ 414×、Alternatives #1 措辞中性化（数字/实测保留，去掉贬低 pgvector 的语气）
遵循 mem0 .github/ISSUE_TEMPLATE/feature_request.yml：Component / Description(Use Case, Proposed Solution, Alternatives Considered) / AI Assistance
-->

**标题**：`feat(vector-store): 新增 NeuG 作为可选 vector store 后端（原生向量 + 全文 + 图的嵌入式数据库）`

**Component**：`Vector Store`

**Description**：

### Use Case

**现有后端在“引擎原生索引能力”上并不齐平。** mem0 目前（上游）有 25 个 vector store 后端，按引擎原生支持的索引类型看：

| 索引能力 | 上游覆盖 | 说明 |
|---|---|---|
| 向量检索 | 25/25 | base 接口强制，全部支持 |
| 全文检索（`keyword_search` / BM25）| **15/25** | base 默认实现返回 `None`，其余 10 个未 override，实际不支持 |
| 引擎内原生图（关系表 + 遍历）| 0/25 | 见下：这不是后端“缺陷”，而是 mem0 的架构选择 |

**关于图：mem0 把图记忆内置在 core 层、而非交给后端。** mem0 的 Graph Memory 是 always-on、provider-agnostic 的——由 core 的 `entity_store` + entity boost 实现，对所有后端一视同仁，因此**后端层本就不被要求提供图能力**（上表 0/25 是设计使然，不是集体缺陷）。但这份内置图在**存储层只借用到后端的向量接口**：实体存成向量行，实体→memory 的连接用 `payload.linked_memory_ids`（一个 list 字段）倒排表达；它只影响 ranking（entity_boost 融入 score），不返回图结构、无 typed/labeled 关系。

**这带来的现实约束：**
- 想要向量 + 全文的 hybrid 检索，只能在 15 个支持 `keyword_search` 的后端里选，且其中多数是外部服务（qdrant / elasticsearch / milvus / pinecone …）；嵌入式本地方案（faiss / chroma）连全文都缺。
- 内置图虽 always-on，但因承载在“向量接口 + payload list 倒排”上：热门实体（连接数千条 memory）的 `linked_memory_ids` 会膨胀，boost 计算需遍历整个 list，用不上任何原生图索引或遍历。

**NeuG 的独特之处：单引擎原生同时支持向量 + 全文 + 图三者。** NeuG 是一个**嵌入式 HTAP 图数据库**（Cypher），在**单一引擎内原生提供向量检索（HNSW）、全文检索（FTS/BM25）与图（关系表 + 变长遍历）**。作为 mem0 后端：
- 向量 + 全文由 neug 原生索引直接承载（hybrid 检索无需外部服务）；
- mem0 内置图的 `entity_store` 复用后端 provider、会自动落在 neug——**即便是内置图记忆系统，其实体数据也可以由 neug 单引擎承载**（当前经标准 base 接口存储、与其余后端一致，本 PR 不改 entity-boost 逻辑），memory 向量与实体同在一个嵌入式库、零额外基础设施；
- 且因为 neug 在引擎层就有原生图能力（这是其余 25 个后端都不具备的），未来还可把内置图的 entity→memory 连接从“payload list 倒排”升级为真正的 typed 图关系、用 Cypher 遍历计算（见 Alternatives Considered #2，本次 PR 不做）。

我们在 LongMemEval-M 记忆检索基准上做了本地离线实测（51,661 sessions，dim=1024，top_k=10，每类 50 查询，recall 按无条件 top-k 计；doc 侧统一 lemmatized-text，还原 mem0 core `add()` 对所有 backend 都存 lemmatized text 的真实行为）。为给出**可辩护的公平对比**，我们选两个**都建了真索引的生产级后端**作基线——qdrant **server 模式**（docker 真 qdrant，建 HNSW + BM25 稀疏倒排 + payload 索引）与 **pgvector**（docker 真 Postgres 16 + pgvector 0.8，建 HNSW + GIN 全文索引），均按 mem0 文档默认接法、**不做针对性调优**：

| 指标 (p50 ms / recall) | **mem0-neug** | mem0-qdrant-server | mem0-pgvector |
|---|---|---|---|
| vector_topk | **2.639 / 0.996** | 21.476 / 0.976 | 1092.99 / 0.998 |
| fts_keyword | 6.621 / **0.9487** | 5.052 / 0.8974 | **3.589** / 0.8462 |
| hybrid | **27.155 / 0.646** | 36.977 / 0.654 | 1110.75 / 0.554 |
| graph_multihop | **5.859 / 1.0** | N/A | N/A |
| load（背景成本，非评分项）| 175.6 s | 102.2 s | 194.3 s |

**诚实解读（对两个建了真索引的生产级基线）：**
- **vector_topk（三者 recall 均 ≥0.976）**：neug 2.639ms、qdrant-server 21.476ms（neug 快约 8×）、pgvector 1092.99ms（与 neug 相差约 414×）。pgvector 的数量级差异**源于其 HNSW 不 filterable、mem0 的过滤查询退化成全表扫，非引擎绝对优劣**——根因见下条 EXPLAIN 实证。
  - **pgvector 的 1093ms 不是“没建索引”，而是 mem0 的过滤查询打不中它的 HNSW**（EXPLAIN ANALYZE 实证）：pgvector 建了 HNSW（无过滤纯 ANN 仅 1.1ms），但 mem0 `search()` 永远带 `WHERE payload->>'user_id'=?`；该过滤列无索引、无表达式统计，planner 误估选择性（估 258 行、实际 51661 行），放弃 HNSW、退回全表 Seq Scan + top-N heapsort 暴力算距离；连 pgvector 0.8 官方 `hnsw.iterative_scan=relaxed_order` 也因这个误估不触发。**这暴露一个真实能力差异：neug / qdrant-server 的 ANN 是 filterable（带 user_id 过滤仍走索引），pgvector 的 HNSW 不 filterable。**
  - *诚实边界*：本基准是单 user_id（一整个用户的 51,661 条 memory），是 planner 误估的最坏情况；多租户生产场景（每 user_id 行数少、选择性接近通用估算）绝对延迟会不同，但 pgvector HNSW 不 filterable 这一结构性限制仍在。我们**保持 mem0 开箱默认、不手动加过滤索引去“救”它**（那超出 mem0 默认接法、对其他臂不对等），如实报告开箱数字。
- **独占 graph_multihop**（5.859ms/1.0）：qdrant、pgvector 结构上都无原生图遍历能力，N/A。
- **fts_keyword 三套机制、各有取舍**：pgvector 3.589ms 最快但 recall 0.8462 最低（Postgres `plainto_tsquery` 是 **AND** 语义，要求查询词全部命中）；neug 6.621ms、recall 0.9487 最高（SQLite FTS5 **exact top-k**，OR 展开、精确全局前 k、无剪枝）；qdrant-server 5.052ms/0.8974（fastembed 稀疏向量 **剪枝 top-k**，近似早停）。neug 的 fts 延迟差是 exact vs 剪枝的策略取舍、非引擎劣势：实测 k=1000 时 neug（~28ms，flat）反比 qdrant（89–128ms）快 3–4×，benchmark 用 top_k=10 恰落在剪枝最优区。
- **hybrid：neug 27.155ms 最快**（qdrant-server 36.977ms 同量级；pgvector 1110.75ms 被 vector 分量拖垮）。
- 代价：NeuG eager（index-first）建索引使导入比 qdrant-server 慢（175.6s vs 102.2s）、但比 pgvector 快（194.3s），属“导入时建好索引换查询快”的取舍；benchmark 已定义 `load_seconds_bg` 为背景成本、不作评分项。

> 注：qdrant 还有 **local 模式**（纯 Python 模拟器、不建任何索引），若拿它当基线，neug vector 倍数会虚高到 ~280×——那是跟“未建索引的玩具模式”比、不代表生产，**我们不采用该口径**；上表两个基线都是建了真索引的生产级后端。

### Proposed Solution

把 NeuG 作为**标准 vector store adapter** 接入，遵循现有 base 接口，**不改动任何 core 逻辑**：

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "neug",
        "config": {
            "collection_name": "mem0",
            "db_path": "/path/to/neug.db",
            "embedding_model_dims": 1536,
            "distance": "cosine",
        },
    },
}
m = Memory.from_config(config)
```

实现已完成，可整理为 PR：
- `mem0/vector_stores/neug.py`：实现全部 base 接口（`create_col`/`insert`/`search`/`keyword_search`/`update`/`delete`/`get`/`list`/…）
- `mem0/configs/vector_stores/neug.py`：`NeuGConfig`
- `mem0/utils/factory.py`：注册 `"neug"`
- `tests/vector_stores/test_neug.py`：mock 单测（bindings 全 mock，CI 无需真实 native 库）
- `tests/vector_stores/test_neug_smoke.py`：真实引擎冒烟（`importorskip` 跳过）
- `docs/components/vectordbs/dbs/neug.mdx`（已登记 `llms.txt`）
- 依赖：`neug` 0.2.0 已发布 PyPI，将加入 `pyproject.toml` 的 `vector-stores` **optional group**（不进核心依赖）

**内置 Graph Memory 自动受益（无需改 core）**：选择 neug 后，内置图的 `entity_store` 会自动落在 neug（`{collection}_entities`）。本 PR **不触碰 entity-boost 逻辑**——它完全通过标准 base 接口运行。

**已知约束（如实说明）**：NeuG 是嵌入式引擎，同一 db 目录在同进程内只能打开一次（error 1004），一个 Database 同时只允许一个读写连接（error 4001）。adapter 已用连接引用计数 + 锁在 `Memory`/`AsyncMemory` 共享场景下正确处理。

### Alternatives Considered

1. **现有后端**：qdrant / pgvector / milvus 等都是成熟的生产级后端，但多为外部服务（需独立部署与运维）；faiss / chroma 是嵌入式方案、但缺原生全文与图。NeuG 的差异化定位是“嵌入式单引擎同时原生支持向量 + 全文 + 图 + 低延迟”这一组合，供偏好零外部依赖、本地嵌入的用户多一个选择。

2. **用 NeuG 图引擎承载内置图逻辑（未来方向，本次不做）**：由于 NeuG 是完整图引擎，内置图的实体连接（当前 `payload.linked_memory_ids` 倒排）未来或可更原生地建模为图关系（Entity→Memory 边），用 Cypher 遍历计算 boost，缓解热门实体膨胀。graphiti 的 neug driver 已用 NeuG 原生关系表（`RELATES_TO`/`MENTIONS`）建模实体图，可作先例。**但这需改动 mem0 core 检索逻辑、影响 provider-agnostic 设计，刻意不在本 PR 范围**。若维护者对此方向开放，我们乐意在后续独立 RFC 中探讨——本 issue 仅请求“NeuG 作为标准 vector store 后端”。

**AI Assistance**：`AI-assisted, but the need is mine`
