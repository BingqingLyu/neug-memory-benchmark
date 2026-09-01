# mem0-qdrant（性能赛道）实现说明

对照 `harness/perf/PERF-ADAPTER-CONTRACT.md` 的逐条落地记录。

## load：存储层直注（绕开 LLM 抽取）
- 入口：`Memory.vector_store.insert(vectors, payloads, ids)`（mem0 的存储层写入
  入口），分批 1000 行。向量/原文原样载入，未重新 embedding。
- 主键：qdrant point id 只接受 int/UUID 字符串，session_id 不是 UUID，故后端主键
  用随机 uuid，行号↔session_id↔主键映射由 adapter 维护（`_id2sid`），
  查询返回前映射回 session_id。
- payload：`data`=session 原文（qdrant 侧 BM25 稀疏向量即由该字段经 fastembed
  编码，等价于"建 BM25 索引"）、`user_id="perf-corpus"`（隔离/过滤键）、
  `session_id`。
- 全程 0 次 LLM 调用：LLM/embedder 仅占位配置满足 `Memory.from_config` 构造；
  查询侧 `embedding_model` 被替换为直注 stub（见下）。
- 环境：为让 qdrant 臂具备 BM25 关键词能力（mem0 的官方做法），在基准 .venv
  安装了 `fastembed`，BM25 模型 `Qdrant/bm25` 经 HF_ENDPOINT=hf-mirror.com
  拉取后已本地缓存。

## query：走 mem0 检索路径
- `query_vector` / `query_hybrid` → `Memory._search_vector_store`（`Memory.search`
  的检索主体，mem0/memory/main.py）：语义 ANN（internal_limit=max(4·top_k, 60)
  over-fetch）+ `keyword_search` BM25 + sigmoid 归一（`normalize_bm25`）+
  `score_and_rank` 加性融合。两处与公开 `Memory.search` 的差异：
  1. 查询 embedding 步骤换成数据集预计算的查询向量（`_InjectedEmbedder` stub
     注入）。语料与查询向量同为 text-embedding-v3（1024 维、已归一），检索数学
     上完全等价，只是避免每题重复调 embedding 服务（那会引入网络延时污染计时）。
  2. entity boost 通道：直注模式不写实体库，该通道结构性为空（不短路任何真实逻辑）。
- `query_fts` → `vector_store.keyword_search`（mem0 的关键词检索通道），查询词
  先过 `lemmatize_for_bm25`（与 `Memory.search` 同口径；本环境无 spaCy，按
  mem0 自身降级逻辑原样返回）。
- `threshold=0.0`：GT 是无条件 top-k / AND 集合，不用 mem0 默认 0.1 阈值丢候选。
- `graph_multihop`：mem0 v2 OSS 检索路径无图遍历 → N/A（契约 §3.5 事实表）。

## 口径备注
- fts GT 为 AND 语义；qdrant BM25 是 OR 打分排序，recall 反映其 AND 贴合度
  （契约 §6 已声明该口径）。
- hybrid GT 是全库暴力向量序与暴力 BM25 序的 RRF；mem0 的融合候选池仅为语义
  top-60（BM25 分只对池内候选加成），属于系统自身检索架构，如实被测。
- 遥测关闭（`MEM0_TELEMETRY=False`）；qdrant 本地模式对 >20k 点的官方警告
  属预期（两臂同为嵌入模式，条件对等）。
