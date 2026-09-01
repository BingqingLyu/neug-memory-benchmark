# mem0-neug（性能赛道）实现说明

对照 `harness/perf/PERF-ADAPTER-CONTRACT.md` 的逐条落地记录。写入路径、
查询路径、占位配置与遥测处理与 mem0-qdrant 臂一致（见该目录 NOTES.md），
此处只记录 NeuG 臂的差异。

## load：批量写入
- 入口仍是 `Memory.vector_store.insert`（契约要求走系统写入入口），但
  `mem0/vector_stores/neug.py::insert` 已改为批量写入：每批 128 行合成一条
  多行 `CREATE (:t {...}), (:t {...}), ...`（绑定参数带行号后缀）。
  NeuG 不支持 UNWIND 列表绑定，故不用 COPY FROM/UNWIND，而是多行 CREATE。
- 逐行 CREATE 实测 ~60ms/行（全量约 51 分钟）；多行 CREATE 实测 ~1ms/行，
  全量 51661 行 load 约 200 秒（含 HNSW/FTS 索引维护），见
  `summary.json: load_seconds_bg=200.7`。
- upsert 语义保留：批量 CREATE 遇主键冲突（error 1009）会整批原子回滚，
  随后逐行重放 `_upsert`（CREATE 冲突再退化为 SET + get 校验）。
- 引擎坑点：NeuG 的 SET 不接受 NULL（error 1011 ERR_NOT_SUPPORTED），
  因此 `update()` 只对 payload 中非 None 的 promoted 列下发 SET。
- benchmark adapter 侧未绕过 mem0 直写引擎；批量化在 mem0 适配器内部完成，
  对其他使用 `vector_store.insert` 的场景同样生效。

## 查询路径（与 qdrant 臂相同）
- `Memory._search_vector_store`：ANN（internal_limit over-fetch）+
  `keyword_search` BM25 + sigmoid 归一 + `score_and_rank` 加性融合；
  查询 embedding 用数据集预计算向量经 stub 直注；实体通道结构性为空。

## 结果口径
- graph_multihop：mem0 v2 OSS 检索路径无图遍历 → N/A。
- 首轮跑出 fts_keyword recall 仅 0.38（qdrant 臂 0.92），定位为测试语料含
  连字符词（如 dutch-made）：NeuG FTS 底层是 SQLite FTS5，未加引号的带
  连字符 token 会被解析成列表达式并报 "no such column"，导致整个关键词
  通道返回 None。修复：`keyword_search` 前对查询按空白切分并逐 token
  加双引号（`mem0/vector_stores/neug.py::_sanitize_fts_query`），保留
  多 token 的 AND 语义。修复后重跑：
  fts_keyword recall 0.974（反超 qdrant 0.923）、延迟不变。
- 与 qdrant 臂对比（修复后）：
  - vector_topk：recall 0.998 vs 0.998（持平）；延迟 5.4ms vs 740ms
  - fts_keyword：recall 0.974 vs 0.923；延迟 3.5ms vs 1312ms
  - hybrid：recall 0.564 vs 0.676；延迟 10.5ms vs 1547ms
- hybrid 归因（控制变量实验，同 50 题、同 RRF-GT）：
  - "hybrid 低于单通道"是 GT 口径错觉：三赛道 GT 不同（vec=暴力余弦
    top10、fts=精确匹配、hybrid=oracle RRF top10）。统一按 RRF-GT 评：
    纯向量 0.532、纯BM25 0.152、mem0 加性融合 0.562——融合有效且
    高于单通道，接入无误。
  - RRF-GT 中约一半文档靠 BM25 排名入 top10（与纯向量 GT 重叠仅 4%），
    纯向量结构天花板 ~0.53，两引擎一致（0.532 vs 0.530）。
  - 两臂差距（0.564 vs 0.676）归因：qdrant 臂 fastembed BM25 带 Porter
    词干化（run/running 同 token），与 oracle GT 口径吻合，关键词池对
    GT recall 0.394；NeuG FTS5 词面精确匹配无词干化，仅 0.152。实测
    NeuG CREATE INDEX USING FTS 不支持 tokenizer 选项（5001/3000），
    属引擎能力限制，非适配器问题。
  - mem0 加性融合与标准 RRF 近似等效（qdrant 0.662 vs 0.696；neug
    0.562 vs 0.532），候选池覆盖 0.946——瓶颈不在融合设计。
  - 可选改进：待引擎支持 FTS tokenizer 选项，或直注时写 text_lemmatized
    列对齐口径（需 load 预处理，超出当前契约，须与数据集侧确认）。
