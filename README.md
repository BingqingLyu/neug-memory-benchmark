# NeuG Memory Benchmark

NeuG 0.2.0（向量 + BM25 全文 + Cypher 图一体）在 agent 记忆场景的对比 benchmark。
对比方式：同一套记忆系统，仅替换存储后端——**with NeuG** vs **without NeuG**（各系统官方默认后端），
除后端外全链路一致，量化存储底座本身带来的差异。

## 两条赛道

- **质量赛道（LoCoMo）**：完整管线。摄入 10 段对话共 272 sessions，回答 1540 道
  cat1–4 问题；qwen-plus 答题、qwen-max 判分。指标：准确率（总体/分类）+ 检索延迟。
- **性能赛道（LongMemEval-M）**：检索层直注。把预计算的
  embedding / 文本 / 共现图（51,661 sessions）直接写入各后端的存储层（不走 LLM），
  只测系统自身检索路径。四类查询 × 50 题：向量 top-k、关键词、混合、图多跳。
  指标：延迟 p50/p95/mean + recall@10（对照 numpy 暴力检索基线）。

统一约束：所有 LLM/embedding 流量经内置缓存代理（`harness/locomo/llm_proxy.py`，
OpenAI 兼容，SQLite 缓存，key = 路径|模型|请求体），保证 with/without 两侧
除后端外输入逐字节一致；计时只取 harness 侧 `perf_counter`。

## 目录结构

```
harness/         统一 harness（两赛道 runner + 各系统 adapter + 契约文档）
  locomo/        质量赛道：run_eval 主入口、judge、llm 客户端、缓存代理
  perf/          性能赛道：run_perf 主入口、数据集构建脚本、bruteforce 参照实现
  prepare_locomo.py  LoCoMo 原始数据 → sessions.jsonl / qa_eval.jsonl
docs/            各后端集成笔记
data/            数据集（不入库，本地生成）
results/         实验结果与缓存（不入库）
```

## 运行

```bash
cp harness/config.example.yaml harness/config.yaml   # 填入自己的配置
export DASHSCOPE_API_KEY=...                          # 或写入 .env

# 质量赛道（--limit-sessions N 可做小规模冒烟）
python -m harness.locomo.llm_proxy --port 8787       # 先起缓存代理
python -m harness.locomo.run_eval --adapter mem0-neug --runs 1
python -m harness.locomo.run_eval --adapter mem0-qdrant --runs 1

# 性能赛道（数据在 data/processed/perf/，先跑 harness/perf/build_dataset1.py）
python -m harness.perf.run_perf --adapter mem0-neug
```

adapter 注册为容错式：某系统实现或依赖未就位时自动跳过，不影响其它系统。
接入新系统请先读对应契约：`harness/locomo/ADAPTER-CONTRACT.md`、
`harness/perf/PERF-ADAPTER-CONTRACT.md`。

## 环境要点

- Python ≥ 3.12（harness 与各系统 fork 的公共下限）。
- NeuG 0.2.0 未发布 PyPI，需从源码本地构建 wheel，构建时启用
  `vector_search` 与 `fts` 扩展；运行时扩展只 `LOAD`，**严禁 `INSTALL`**
  （会删除本地扩展二进制并尝试联网下载）。
- 各系统使用其官方上游 + NeuG 集成分支（见下），不改动其默认后端代码路径，
  保证 without-NeuG 侧为官方行为。

## 对比系统与集成位置

| 系统 | without NeuG（默认后端） | with NeuG |
|---|---|---|
| mem0 | qdrant | NeuG vector store（fork 分支 `feature/neug-vector-store`） |
| cognee | 内置 LanceDB 栈 | NeuG graph adapter（fork 分支 `feature/neug-graph-adapter`） |
| graphiti | Neo4j | NeuG driver（fork 分支 `neug-driver`） |
| semantica | 原生内存栈 | NeuG 后端（bm25 全文路仅 NeuG 提供） |

## 数据集

- **LoCoMo-10**：10 对话 / 272 sessions / 1540 题（cat1 单跳、cat2 时间、
  cat3 多跳、cat4 开放式），约 193k token。
- **LongMemEval-M**：清洗去重后 51,661 sessions / 约 1.3 亿 token，
  embedding 统一用 text-embedding-v3（1024 维），共现图 2.8M 边。

## 当前状态

- mem0（NeuG / qdrant 双后端）：两赛道全量已完成首版（单轮），见 `docs/`。
- 其余系统：adapter 开发/验收中，完成并通过验收后单独提交。
