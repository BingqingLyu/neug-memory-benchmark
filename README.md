# NeuG Memory Benchmark

NeuG v0.2.0 agent memory benchmark 的工程目录。设计文档：
`~/Documents/projects/neug-blogs/blogs/release-v0.2.0-index-ai/benchmark-plan.md`（v1.3）

## 目录结构

```
data/            数据集（locomo10.json 原始；processed/ 为生成的评测集）
harness/         统一 harness（LoCoMo 协议：DashScope Qwen 答题+judge）
probes/          前置探针（probe_neug_020.py 已验证三检索全链路）
results/         构建日志与实验结果
neug-build-312/  py3.12 wheel 独立构建目录（勿与 alibaba/neug 的 dev build 混用）
```

## 环境

- `.venv`：Python 3.12.14，benchmark 主环境（neug wheel + 各系统 adapter）
- NeuG 0.2.0 尚未发布 PyPI：wheel 从 `~/Documents/projects/alibaba/neug`（main，0.2.0 代码）
  本地构建，`NEUG_BUILD_DIR` 指向本目录的 neug-build-312，**不得**用 alibaba/neug 自带
  build/ 目录（那是 py3.9 dev 环境，ABI 不同且会被重建破坏）。
  构建要点：`CI_INSTALL_EXTENSIONS="vector_search;fts"`；CMAKE_ARGS 需带
  `-DBUILD_HTTP_SERVER=OFF -DCRYPTO_LIB=<deps/openssl/lib/libcrypto.a>
  -DPROTOC_LIB=<build/third_party/protobuf/libprotoc.a>`。
- NeuG dev 环境（探针/临时验证用）：`~/Documents/projects/alibaba/neug/.venv`（py3.9），
  扩展只 LOAD 不 INSTALL。

## 集成仓库

| 系统 | 仓库 | 分支 |
|---|---|---|
| mem0 fork (v2.0.19) | ~/Documents/projects/mem0 | feature/neug-vector-store |
| cognee fork (v1.5.3) | ~/Documents/projects/cognee | feature/neug-graph-adapter |
| graphiti | ~/Documents/projects/graphiti | （WS0 由另一 agent 负责） |
| graphify | ~/Documents/projects/graphify | with-neug = 上游 PR #2895 分支 |
| codegraph | ~/Documents/projects/codegraph | with-neug = 上游 PR #698 分支 |

## 数据集状态

- LoCoMo-10：已下载验证（10 对话 / 272 session / cat1-4 共 1540 题 / ~193k token）
- LongMemEval-M（性能赛道）：待下载
- 代码域仓库（性能赛道数据集 2）：待选型冻结
