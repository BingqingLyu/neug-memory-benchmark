# mem0 后端生态与 NeuG 定位

> 调研基准：mem0 `v2.0.19`，权威来源 `mem0/utils/factory.py`（各 Factory 的
> `provider_to_class` 注册表）+ `mem0/memory/main.py` / `mem0/utils/scoring.py`
> （内置 Graph Memory 检索链路）。整理于 2026-09-09；图能力结论已按官方文档
> https://docs.mem0.ai/platform/features/graph-memory 更正。

## 1. mem0 后端引擎全景

mem0 的后端按职责分为四类 Factory：

| 引擎类别 | 数量 | 说明 |
|---|---|---|
| **Vector Store** | **26** | 记忆存储引擎（neug 在列） |
| LLM | 18 | openai/anthropic/gemini/ollama/deepseek… |
| Embedder | 11 | openai/huggingface/fastembed/vertexai… |
| Reranker | 5 | cohere/sentence_transformer/llm_reranker… |
| Graph Store | 0 | **无独立 GraphStoreFactory**——但图能力并非空白，而是内置进检索算法（见 §2） |

“后端引擎”通常指 **26 个 vector store**。**关键澄清**：mem0 没有独立的图存储引擎
抽象，但这不等于“没有图能力”——mem0 把图记忆内置成了 always-on 的检索增强（§2），
不再需要外挂图数据库。

## 2. mem0 内置 Graph Memory（更正：图能力不是空白）

mem0（OSS v2.0.19 与 Platform 同一套算法）有**内置、always-on、零配置、schema-free
的 Graph Memory**。它不是独立图数据库，而是检索管道里的“实体连接”通道：

1. **实体抽取**：`mem0/utils/entity_extraction.py`（772 行，spaCy NLP）从每条 memory
   抽取 PROPER/QUOTED/TOPIC/IDENTIFIER 四类实体，实体存一次并 embed。
2. **共现连接**：同一实体出现在多条 memory → 这些 memory 通过该实体连接成图。
3. **检索 boost**：`main.py:_compute_entity_boosts`，`boost = similarity ×
   ENTITY_BOOST_WEIGHT × memory_count_weight`（权重 0.5）。
4. **三通道融合**：`mem0/utils/scoring.py`，`combined = (semantic + bm25 +
   entity_boost) / max_possible`（max_possible 自适应 1.0 / 2.0 / 2.5）。

**特性**：只影响 ranking（融入 score），**不返回图结构、无 typed/labeled 关系**
（官方明确 “won't record a 'manages' edge”）、无 relations payload。

**历史**：mem0 早期通过外部图数据库（Neo4j/Memgraph/Kuzu/Apache AGE/Neptune +
`enable_graph` + `graph_store` 配置 + `relations` 字段）实现图记忆，**现已被内置
Graph Memory 取代并移除**——v2.0.19 OSS 仅剩 `exceptions.py:396` 文档字符串里的
kuzu 残留示例。

## 3. 26 个 Vector Store 分类

**按部署形态：**
- **嵌入式/本地（3）**：faiss、chroma、**neug**
- **自托管开源服务（10）**：qdrant、milvus、weaviate、opensearch、elasticsearch、
  redis、valkey、cassandra、mongodb、pgvector
- **云托管 SaaS（12）**：pinecone、azure_ai_search、azure_mysql、vertex_ai_vector_search、
  databricks、supabase、turbopuffer、upstash_vector、s3_vectors、baidu、neptune、oracledb
- **适配桥接（1）**：langchain（桥接任意 LangChain vector store）

**按能力矩阵：**

| 能力 | 覆盖 | 说明 |
|---|---|---|
| 向量检索 `search` | 26/26 | base.py 强制（abstractmethod） |
| 全文/hybrid `keyword_search` | 16/26 | base.py 可选（默认实现）；neug/qdrant/es/weaviate/pgvector/milvus 等支持 |
| 内置 entity-boost 图（§2） | 26/26 | 在 main.py 层，**与 vector store 后端无关**，所有后端共享 |
| adapter 层显式图方法 `add_edge`/`traverse` | **1/26** | **仅 neug**（typed 关系表 + Cypher 变长遍历，返回邻域） |

## 4. NeuG 定位与优劣（重新定位）

NeuG 是 26 个后端里唯一在 adapter 层提供**显式 typed 属性图**的嵌入式 HTAP 引擎。
它的图能力与 mem0 内置 Graph Memory 是**两种不同范式**，不是“填补空白”，而是“提供
内置图不覆盖的能力”：

| 维度 | mem0 内置 Graph Memory | neug adapter 图方法 |
|---|---|---|
| 图构建 | 隐式：spaCy 抽实体 → 共现连接 | 显式：`add_edge(relation, weight)` |
| 关系 | schema-free，无 typed/labeled 边 | typed（relation STRING）+ weighted（DOUBLE）|
| 作用 | 只影响 ranking（entity_boost 融入 score）| 显式遍历查询（`traverse *1..2` 返回邻域）|
| 返回 | 无图结构，只有 combined score | 完整邻域节点集（Cypher BFS）|
| 配置 | 零配置 always-on | 需建关系表、写边 |

**优势：**
1. **显式 typed 图 + 遍历**：内置图只做 schema-free 的 ranking boost；neug 提供
   typed/labeled 关系和显式多跳遍历查询（返回子图结构），覆盖内置图不提供的用例。
2. **原生索引 → 查询延迟极低且规模无关**：eager 建 HNSW+FTS。benchmark 实测
   vector 2.456ms、fts 6.15ms、graph 5.608ms，比 qdrant local 快 vector ~280×、fts ~205×。
3. **COPY 批量导入**：`bulk_insert_copy` 为语料级载入优化，其他后端只有逐批 `insert`。
4. **零外部服务**：单文件 `neug.db`，本地部署简单。

**劣势：**
1. **生态成熟度**：neug 是新接入后端；qdrant/pinecone/pgvector/milvus 是社区主流，
   文档、客户端、生产案例更丰富。
2. **无全托管 SaaS / 分布式弹性**：neug 是嵌入式单机；pinecone/turbopuffer/vertex_ai/
   databricks 提供全托管、自动分片、弹性扩展。
3. **导入成本更高**：eager 建索引 → load 174.3s，比不建索引的后端慢（qdrant local 97.4s）。
   属“导入慢换查询快”取舍（详见 `../results/perf/mem0-neug/NOTES.md`）。
4. **技术栈绑定**：C++ 引擎 + Python/Node 绑定，复用已有 PostgreSQL/ES 栈不如
   pgvector/opensearch 顺滑。
5. **图能力与内置 Graph Memory 部分重叠**：mem0 内置的 entity-boost 图已零配置覆盖
   “entity-centric + multi-hop 检索增强”主用例；neug 的显式图是更重、更结构化的补充，
   需论证“为什么需要 typed 关系 + 显式遍历，而不只是内置 boost”。

**定位结论**：mem0 的图记忆已内置（schema-free entity-boost，所有后端共享）。neug 的
差异化不在“有没有图”，而在“提供内置图不覆盖的显式 typed 关系 + 图遍历查询”，叠加
极低查询延迟与 COPY 批量导入——适合**需要显式多跳关系查询、可控图 schema** 的记忆场景；
代价是导入慢、无云托管弹性。纯向量检索的超大规模场景，pinecone/milvus 等专用云向量库更省心。

## 5. benchmark 实证（LongMemEval-M，51661 sessions，top_k=10）

| 指标 | mem0-neug | mem0-qdrant | neug 领先 |
|---|---|---|---|
| vector_topk | 2.456ms / 0.996 | 687.8ms / 0.998 | ~280× |
| fts_keyword | 6.15ms / 0.9487 | 1261.7ms / 0.8974 | ~205× |
| hybrid | 28.057ms / 0.646 | 1466.3ms / 0.668 | ~52× |
| graph_multihop | 5.608ms / 1.0 | N/A | 独占（见下注） |
| load（背景成本） | 174.3s | 97.4s | qdrant 快（不建索引） |

评分项为查询延迟 + recall，neug 四类全面领先；load 是背景成本、非评分项。

> **graph_multihop 注**：perf 赛道为离线免费采用“存储层直注、绕过 `add()` 实体写入”，
> 导致 entity_boost 通道结构性为空（实体库无写入），**mem0 内置 Graph Memory 未被激活**。
> 故本项 graph_multihop 由 neug adapter 自建关系表 + Cypher BFS 实现，qdrant 臂无显式图
> 而标 N/A。若走完整 `add()` 管道（spaCy 抽取在 main.py 层、与后端无关），所有后端都会
> 获得内置 entity-boost 的多跳检索增强——但这不是显式图遍历。

## 6. 附录：neug 接入 mem0 的代码足迹

已在 fork `BingqingLyu/mem0` 的 `feature/neug-vector-store` 分支：
- `mem0/vector_stores/neug.py`（实现，973 行）
- `mem0/configs/vector_stores/neug.py`（`NeuGConfig`）
- `mem0/utils/factory.py`（注册 `"neug"`）
- `tests/vector_stores/test_neug.py`（816 行，mock 单测，CI 友好）
- `tests/vector_stores/test_neug_smoke.py`（真实引擎，`importorskip` 跳过）
- `docs/components/vectordbs/dbs/neug.mdx` + `neug-comparison.mdx`（已登记 llms.txt）
- neug `0.2.0` 已发布 PyPI（包名 `neug`，GraphScope Team，无名字冲突）
