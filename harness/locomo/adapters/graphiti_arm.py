"""graphiti LoCoMo adapter：with/without 两臂（graphiti-neug / graphiti-neo4j）。

契约：harness/locomo/ADAPTER-CONTRACT.md
- 会话隔离：group_id = sample_id，search 带同样隔离条件（约束 1）
- search 只返回检索上下文（边 facts + valid_at/invalid_at 时间标注），答题由 harness 统一做（约束 2）
- LLM 统一 DashScope：抽取 qwen-plus 走 OpenAIGenericClient（不是 OpenAIClient，
  见 results/GRAPHITI-ISSUES-FOR-QODER.md 解决确认节），
  embedding text-embedding-v3 1024 维（约束 3）
- LLM/embedding 的 base_url 指向缓存代理 127.0.0.1:8787（约束 4），
  with/without 两臂共享同一抽取缓存
- 后端切换是唯一变量：两臂除 graph driver 外配置完全一致（约束 5）
- teardown 正常 close：NeuG checkpoint 纪律（约束 6）
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.openai_reranker_client import (  # noqa: E402
    OpenAIRerankerClient,
)
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import (  # noqa: E402
    LLMConfig,
    OpenAIGenericClient,
)

from ..base import SearchHit, SearchTrace, SystemAdapter  # noqa: E402

EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-v3")
EMBED_DIM = int(os.environ.get("BENCH_EMBED_DIM", "1024"))
LLM_MODEL = os.environ.get("GRAPHITI_LLM", "qwen-plus")
# 抽取缓存代理（跑前：python -m harness.locomo.llm_proxy --port 8787）
PROXY_BASE_URL = os.environ.get("BENCH_LLM_PROXY_URL", "http://127.0.0.1:8787/v1")
# DashScope 兼容端点 embedding 单批上限（与 naive_rag adapter 同口径）
EMBED_BATCH = 10


class ChunkedOpenAIEmbedder(OpenAIEmbedder):
    """DashScope 限制单次 embedding 批量 <=10；graphiti 默认一次性整批提交。"""

    async def create_batch(self, input_data_list):
        out = []
        for i in range(0, len(input_data_list), EMBED_BATCH):
            out.extend(await super().create_batch(input_data_list[i:i + EMBED_BATCH]))
        return out


class GraphitiArmAdapter(SystemAdapter):
    """一个 (graphiti, 后端) 配置 = 一个实例。backend: 'neug' | 'neo4j'。"""

    sub_scenario = "A"

    def __init__(self, backend: str):
        assert backend in ("neug", "neo4j")
        self.backend = backend
        self.name = f"graphiti-{backend}"
        self._loop = None
        self._driver = None
        self.graphiti = None
        self._ingested: set[tuple[str, int]] = set()

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")

        # harness 是同步接口；graphiti 全异步——本 adapter 独占一个事件循环，
        # 各调用串行经它调度（run_eval 检索侧已有锁，不引入并发）。
        self._loop = asyncio.new_event_loop()

        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        if self.backend == "neug":
            from graphiti_core.driver.neug_driver import NeuGDriver

            # 库必须落在 work_dir 内：run_eval 的过期守卫靠 rmtree(work_dir)
            # 整库重建（摄入规模变了就不能新旧混用）；放父目录会逃过删除，
            # 再被 _sync_ingested_from_db 回填成已摄入 → 静默在旧数据上继续 QA。
            # 跨 run 复用不需要绕行：run_eval:105 对 run>0 直接传 work_run0。
            self._driver = NeuGDriver(
                db_path=str(wd / "graphiti.db"), embedding_dim=EMBED_DIM
            )
        else:
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            self._driver = Neo4jDriver(
                uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                user=os.environ.get("NEO4J_USER", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", "testpass"),
            )

        # LLM/embedding/reranker 全部指向缓存代理（约束 4）。
        # reranker 必须显式传入：Graphiti 默认 OpenAIRerankerClient() 读
        # OPENAI_API_KEY，不设直接报 Missing credentials。
        llm = OpenAIGenericClient(
            config=LLMConfig(api_key=api_key, model=LLM_MODEL, base_url=PROXY_BASE_URL)
        )
        reranker = OpenAIRerankerClient(
            config=LLMConfig(api_key=api_key, model=LLM_MODEL, base_url=PROXY_BASE_URL)
        )
        embedder = ChunkedOpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=api_key,
                base_url=PROXY_BASE_URL,
                embedding_model=EMBED_MODEL,
                embedding_dim=EMBED_DIM,
            )
        )
        self.graphiti = Graphiti(
            graph_driver=self._driver, llm_client=llm, embedder=embedder,
            cross_encoder=reranker,
        )
        self._run(self.graphiti.build_indices_and_constraints())
        # 防复发守卫（见 results/BENCHMARK 汇总 §四）：质量/性能共用同一 Neo4j
        # 实例，perf adapter 只在 setup 清库、teardown 不清；若上一轮性能语料
        # 残留，本 adapter 的 Episodic 幂等守卫看不到它 → 全量重摄入 LoCoMo →
        # 两批数据共存一库，全局 top-K 向量检索被拖到性能量级（1850ms 事故）。
        if self.backend == "neo4j":
            self._clear_foreign_data()
        self._sync_ingested_from_db()

    def _clear_foreign_data(self):
        """精确清掉性能赛道遗留语料（group_id='perf'），不触碰任何 LoCoMo
        sample_id 组。先按批删 perf 边（hub 节点单批 DETACH DELETE 会撑爆事务
        内存），再删 perf 孤立点。当前库若全为 perf 语料等价于清空；若已有
        LoCoMo 图则只摘除 perf 部分，保留可复用的质量数据。"""
        assert self.graphiti is not None

        async def _wipe():
            while True:
                recs, _, _ = await self.graphiti.driver.execute_query(
                    "MATCH ()-[r {group_id: 'perf'}]->() "
                    "WITH r LIMIT 100000 DELETE r RETURN count(*) AS c"
                )
                if not recs or int(recs[0]["c"]) == 0:
                    break
            while True:
                recs, _, _ = await self.graphiti.driver.execute_query(
                    "MATCH (n {group_id: 'perf'}) "
                    "WITH n LIMIT 50000 DELETE n RETURN count(*) AS c"
                )
                if not recs or int(recs[0]["c"]) == 0:
                    break

        t0 = time.time()
        self._run(_wipe())
        print(f"[{self.name}] cleared foreign perf data in {time.time()-t0:.1f}s",
              flush=True)

    def _sync_ingested_from_db(self):
        """跨进程幂等：从库里读已有 episode 名（{sample_id}-{session_idx}）回填守卫。
        否则重跑复用同一库时会全量重复摄入（graphiti 不按 episode 名去重）。"""
        assert self.graphiti is not None

        async def _q():
            recs, _, _ = await self.graphiti.driver.execute_query(
                "MATCH (e:Episodic) RETURN e.name AS name"
            )
            return [r["name"] for r in recs]

        for name in self._run(_q()):
            if name and "-" in name:
                sid, _, idx = name.rpartition("-")
                if idx.isdigit():
                    self._ingested.add((sid, int(idx)))

    def teardown(self):
        if self.graphiti is not None:
            try:
                self._run(self.graphiti.close())
            finally:
                self.graphiti = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # ---- 摄入 ----
    def ingest_all(self, sessions):
        # 幂等守卫（含 setup 时的库内回填）：防同进程/跨进程重复摄入。
        pending = [s for s in sessions
                   if (s["sample_id"], s["session_idx"]) not in self._ingested]
        # 全量摄入几个小时，逐 session 打进度（覆写 base 的静默循环）
        t0 = time.time()
        for i, s in enumerate(pending):
            self.ingest_session(s["sample_id"], s["session_idx"], s["date_time"], s["text"])
            self._ingested.add((s["sample_id"], s["session_idx"]))
            print(f"[{self.name}] session {i+1}/{len(pending)} "
                  f"({s['sample_id']}/{s['session_idx']}) "
                  f"{(time.time()-t0)/60:.1f}min elapsed", flush=True)
        self._ingested.update((s["sample_id"], s["session_idx"]) for s in pending)

    def reuse_ingested(self, sessions):
        """marker 存在也必须过 ingest_all 的幂等守卫，不能直接当"已摄入"。

        run_eval:117 在 marker 存在时调本方法而**不调** ingest_all，而基类
        实现是空的（base.py:62-67）。于是库一旦与 marker 失配就静默跑空图：
        setup() 在 work_dir 内新建一个空库，_sync_ingested_from_db() 回填 0
        条 Episodic，QA 全程 0 命中，run 却报成功。真实发生过——旧库落在
        work_run0 的**父目录**（results/locomo/graphiti-neug/graphiti.db），
        rmtree(work_dir) 删得掉 marker 删不掉库，两者就此脱钩；与 2026-09-09
        semantica 臂塌缩成纯 bm25、差值被误读成代码收益是同一族事故。

        委托给 ingest_all 即可自愈，不需要抛：它的 pending 过滤依赖
        _ingested，而 _ingested 是 setup 时从**库里真实的 Episodic 行**回填
        的，不看 marker。健康冻结库 -> pending 为空 -> 瞬间 no-op；空库或
        错库 -> pending 为全量 -> 重新摄入。重摄入不贵：llm_proxy 把抽取与
        嵌入缓存在 results/cache/llm_proxy/cache.db，重跑走缓存命中。
        """
        self.ingest_all(sessions)
        self._assert_edge_leg()

    def _assert_edge_leg(self):
        """复用路径的唯一不可信点：确认检索真正读的边确实在库里。

        与 semantica 的 _assert_vector_leg 同理——宁可炸，不要静默出数。
        graphiti 的检索回的是边 facts（见 search），Episodic 有行不代表边
        写进去了，而空边集只会让每题 0 命中，与"确实没召回"无法区分。
        """
        assert self.graphiti is not None

        async def _q():
            recs, _, _ = await self.graphiti.driver.execute_query(
                "MATCH (:Entity)-[r:RELATES_TO]->(:Entity) RETURN count(r) AS c"
            )
            return recs

        recs = self._run(_q())
        n = int(recs[0]["c"]) if recs else 0
        if n == 0:
            raise RuntimeError(
                f"{self.name}: store has {len(self._ingested)} episodes but ZERO "
                "RELATES_TO edges, so every question would retrieve nothing and "
                "the run would report a plausible-looking 0. The store and "
                ".ingest_complete are out of step -- delete work_run0 and "
                "re-ingest. The store must live INSIDE work_dir (see setup) so "
                "rmtree clears both together."
            )

    def ingest_session(self, sample_id: str, session_idx: int, date_time: str, text: str):
        assert self.graphiti is not None
        self._run(
            self.graphiti.add_episode(
                name=f"{sample_id}-{session_idx}",
                episode_body=text,
                source_description="locomo conversation session",
                reference_time=self._parse_dt(date_time),
                group_id=sample_id,
            )
        )

    # ---- 检索：只回边 facts，答题归 harness ----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        # 约束 1 会话隔离：group_id = 题目携带的 sample_id，必填。缺失就抛——
        # "最后一次摄入的样本"兜底在多样本全量跑时只会命中最后一个样本，而
        # 复用库路径下 ingest_session 根本不跑，它只会静默退化成 0 命中。
        if sample_id is None:
            raise ValueError("graphiti arm requires sample_id for group isolation")
        assert self.graphiti is not None
        edges = self._run(
            self.graphiti.search(
                question, group_ids=[sample_id], num_results=top_k
            )
        )
        hits = [
            SearchHit(
                id=e.uuid,
                # graphiti 的 search() 不回传相关度分数（EntityEdge 无 score 字段），
                # harness 也只按返回顺序拼上下文、从不按 score 排序。
                score=0.0,
                payload=self._payload(e),
            )
            for e in edges
        ]
        return hits, SearchTrace(extra={"group_id": sample_id})

    @staticmethod
    def _payload(edge) -> str:
        # fact 保留原文相对表达（"yesterday"），绝对时间在 valid_at/invalid_at；
        # 不拼进上下文，答题模型无法解析时序问题。
        fact = getattr(edge, "fact", "") or ""
        temporal = []
        valid_at = getattr(edge, "valid_at", None)
        invalid_at = getattr(edge, "invalid_at", None)
        if valid_at is not None:
            temporal.append(f"this fact happened on {valid_at.date().isoformat()}")
        if invalid_at is not None:
            temporal.append(f"no longer true after {invalid_at.date().isoformat()}")
        if not temporal:
            return fact
        return f"{fact} [{'; '.join(temporal)}]"

    # ---- 内部 ----
    def _run(self, coro):
        assert self._loop is not None
        return self._loop.run_until_complete(coro)

    @staticmethod
    def _parse_dt(s: str) -> datetime:
        # 实测格式："1:56 pm on 8 May, 2023"（ISO 格式做兜底）
        try:
            dt = datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
        except ValueError:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # 契约：统一 UTC
        return dt


class GraphitiNeuGAdapter(GraphitiArmAdapter):
    def __init__(self):
        super().__init__("neug")


class GraphitiNeo4jAdapter(GraphitiArmAdapter):
    def __init__(self):
        super().__init__("neo4j")
