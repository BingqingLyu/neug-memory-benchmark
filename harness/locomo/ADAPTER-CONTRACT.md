# LoCoMo 质量赛道：SystemAdapter 接口契约（给 Qoder）

目标：为 mem0 / graphiti / cognee 各写 with/without 两个 adapter（共 6 个），
注册进 harness，跑通小规模试跑。harness 侧（runner/判分/计时）已就绪，勿改。

## 1. 接口（harness/locomo/base.py，已固定）

```python
class SystemAdapter(ABC):
    name: str                    # 如 "mem0-neug" / "graphiti-neo4j"
    sub_scenario: str = "A"

    def setup(self, work_dir: str, cache_dir: str): ...
        # work_dir：本配置独占的工作目录（DB 文件放这里）
        # cache_dir：同系统跨后端共享的缓存目录

    def ingest_session(self, sample_id: str, session_idx: int,
                       date_time: str, text: str): ...
        # 按 session 增量摄入。调用顺序：sample_id 内 session_idx 升序
        # （graphiti 的时序边依赖 reference_time 顺序）

    def search(self, question: str, top_k: int = 20
               ) -> tuple[list[SearchHit], SearchTrace]: ...
        # 只返回检索结果，不含答题。harness 统一计时、统一答题判分

    def teardown(self): ...      # 必须正常 close（NeuG ghost PK / checkpoint 纪律）

# SearchHit(id: str, score: float, payload: str)   payload=送入答题模型的上下文文本
# SearchTrace(latency_ms, context_chars, extra)    latency_ms 由 harness 填，adapter 不管
```

便捷方法 `ingest_all(sessions)` 默认逐条调 ingest_session，可覆盖做批量优化。

## 2. 数据（已就绪，勿重新生成）

- `data/processed/sessions.jsonl`：272 session（10 段对话），字段
  `{sample_id, session_idx, date_time, text}`
- `data/processed/qa_eval.jsonl`：1540 题（harness 用，adapter 不碰）

## 3. 六条硬约束

1. **会话隔离**：按 sample_id 隔离（mem0=user_id、graphiti=group_id、cognee=dataset_name），
   search 时必须带同样隔离条件
2. **search 返回检索上下文，不返回系统的最终 LLM 答案**——答题由 harness 统一做。
   cognee 注意：HYBRID_COMPLETION 默认返回 LLM 补全答案，**必须取补全前的检索上下文**
   （retriever 层结果拼装），否则判分 unfair
3. **LLM 配置统一 DashScope**：抽取/答题用 qwen-plus，embedding 用 text-embedding-v3
   （1024 维）。mem0 默认 embedder 是 OpenAI，**必须改配**。
   graphiti 必须用 `OpenAIGenericClient`（不是 OpenAIClient，见
   results/GRAPHITI-ISSUES-FOR-QODER.md 解决确认节）
4. **抽取走缓存代理**：LLM base_url 指向 `http://127.0.0.1:8787/v1`
   （harness/locomo/llm_proxy.py，跑前启动）。同系统两后端共享抽取结果，
   第二个后端摄入近似零 LLM 成本
5. **后端切换是唯一变量**：with/without 两臂除存储后端外配置完全一致
   （mem0: neug adapter vs qdrant；graphiti: NeuGDriver vs Neo4jDriver；
   cognee: GRAPH/VECTOR_DB_PROVIDER=neug vs ladybug+lancedb 默认）
6. **teardown 正常关闭**：NeuG 连接必须 close（checkpoint 纪律）

## 4. 各系统映射参考

| 系统 | ingest | search |
|---|---|---|
| mem0 | `Memory.add(text, user_id=sample_id)` | `Memory.search(question, user_id=sample_id)` → memory 文本作 payload |
| graphiti | `add_episode(name=f"{sample_id}-{session_idx}", episode_body=text, source_description="locomo", reference_time=parse(date_time), group_id=sample_id)` | `search(question, group_ids=[sample_id])` → facts 作 payload |
| cognee | `cognee.add(text, dataset_name=sample_id)` + 适时 `cognify()` | retriever 层取 HYBRID 检索上下文（见约束 2），datasets=[sample_id] |

cognee 的 cognify 时机自定（每 session / 每 sample 批量），注意成本与管线语义。
date_time 格式：`"2023-04-10 17:50:00"` 风格，自行 parse（含时区处理，UTC）。

## 5. 交付物与验收

- 文件：`harness/locomo/adapters/{mem0,graphiti,cognee}_arm.py`，
  每个文件注册 with/without 两个 name（如 `mem0-neug`、`mem0-qdrant`）
- 注册方式：`run_eval.py` 的 `_load_adapters()` 里 import 即生效
- 每臂冒烟：
  ```bash
  cd ~/Documents/projects/neug-memory-benchmark
  set -a && source .env && set +a
  .venv/bin/python -m harness.locomo.run_eval --adapter <name> --limit 5 --runs 1
  ```
  通过标准：摄入无异常、search 返回非空 payload、summary.json 产出、
  两后端（with/without）都能跑通同一冒烟

## 6. 已知坑（各系统验收报告里核过的）

- mem0 过滤语法：扁平字段 + 操作符（`{"tag": {"eq": "bench"}}`），嵌套 dict 抛错
- graphiti：OpenAIGenericClient；自定义 entity_types 的 json_schema 路径未验证（用默认类型）
- cognee：CHUNKS_LEXICAL 已修复（OR-join）；cognify 有 E 级 schema 探测日志噪声，忽略
- NeuG 方言：probes/NEUG-DIALECT-NOTES 见 results/NEUG-DIALECT-NOTES.md
  （parameters 关键字传、bm25 紧跟 ORDER BY score ASC LIMIT、IN 绑参段错误用内联）
