"""cognee LoCoMo adapter：cognee-neug（with）/ cognee-default（without）两臂。

后端对照（契约约束 5：后端切换是唯一变量）——两臂除存储后端外配置完全一致：
  - cognee-neug:    GRAPH/VECTOR_DB_PROVIDER=neug（NeuG 单引擎，图+向量+FTS）
  - cognee-default: GRAPH=ladybug + VECTOR=lancedb（cognee 内置默认组合）

管线映射（契约 §4）：
  ingest:  cognee.add(text, dataset_name=sample_id)，按 sample_id 逻辑隔离
           （约束 1）；全部 272 session 入库后统一跑一次 cognify()
           （时机自定条款）。
  search:  SearchType.HYBRID_COMPLETION + only_context=True —— 只取补全前的
           检索上下文（retriever 层拼装），不做 LLM 补全；答题由 harness 统一
           完成（约束 2）。接口不透传 sample_id（harness 协议如此，与其余系统
           一致），故检索在共享库上全局进行，与 naive-rag 基线同口径。

LLM / embedding（约束 3、4）：DashScope qwen-plus / text-embedding-v3（1024 维），
base_url 全部指向缓存代理（http://127.0.0.1:8787/v1）。两臂的抽取/嵌入请求体
完全一致，代理按 sha256(path+body) 命中缓存，第二个后端摄入近似零成本——
因此共享的 cache_dir 不再另存抽取产物。

异步模型：cognee 全异步，其引擎/连接池与事件循环绑定，故 adapter 起一个常驻
后台 loop，所有 add/cognify/search 经 run_coroutine_threadsafe 提交，
全进程单 loop（harness 侧检索本来就有锁串行）。

配置顺序（坑）：cognee/__init__.py 在 import 时执行 load_dotenv(override=True)，
会把 CWD 可见的 .env（开发检出时是 cognee 仓库根的 .env）覆写进 os.environ。
所以本 adapter 先 import cognee 让 dotenv 跑完，再 update os.environ——
否则我们的后端/目录配置会被仓库 .env 顶掉。

多租户：两臂统一 ENABLE_BACKEND_ACCESS_CONTROL=false —— NeuG 未注册 dataset
database handler，per-dataset 物理库不可用；为保持约束 5（后端是唯一变量），
两臂都用单租户共享库 + dataset_name 逻辑隔离。
"""
import asyncio
import os
import sys
import threading
from pathlib import Path

from ..base import SearchHit, SearchTrace, SystemAdapter

PROXY_BASE_URL = os.environ.get("BENCH_LLM_PROXY", "http://127.0.0.1:8787/v1")
LLM_MODEL = "openai/qwen-plus"
EMBED_MODEL = "openai/text-embedding-v3"
# cognify 切块粒度（tokens）：LoCoMo 细粒度 QA 需要比默认 ~8k 更细的块
CHUNK_SIZE = 1024


class _CogneeArm(SystemAdapter):
    sub_scenario = "A"
    backend = ""  # "neug" | "default"

    def __init__(self):
        self._cognee = None
        self._SearchType = None
        self._loop = None
        self._thread = None

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        env = {
            # 数据目录：本次 run 的 work_dir 内隔离
            "DATA_ROOT_DIRECTORY": str(work / "data"),
            "SYSTEM_ROOT_DIRECTORY": str(work / "system"),
            "CACHE_ROOT_DIRECTORY": str(work / "cache"),
            "COGNEE_LOGS_DIR": str(work / "logs"),
            # 两臂统一单租户：NeuG 无 per-dataset handler（约束 5 要求两臂同构）
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            # LLM：qwen-plus 经缓存代理（约束 3/4）
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": LLM_MODEL,
            "LLM_ENDPOINT": PROXY_BASE_URL,
            "LLM_API_KEY": api_key,
            # embedding：text-embedding-v3（1024 维）经缓存代理
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": EMBED_MODEL,
            "EMBEDDING_ENDPOINT": PROXY_BASE_URL,
            "EMBEDDING_API_KEY": api_key,
            "EMBEDDING_DIMENSIONS": "1024",
            # DashScope v3 不接受 dimensions 参数，让 litellm 丢弃它
            "LITELLM_DROP_PARAMS": "true",
            "EMBEDDING_BATCH_SIZE": "10",
        }
        if self.backend == "neug":
            env.update({
                "GRAPH_DATABASE_PROVIDER": "neug",
                "VECTOR_DB_PROVIDER": "neug",
                "NEUG_DB_PATH": str(work / "neug_db"),
            })
        else:
            env.update({
                "GRAPH_DATABASE_PROVIDER": "ladybug",
                "VECTOR_DB_PROVIDER": "lancedb",
            })
        # 配置顺序三重防御（实测教训：仅"import 后再注入"不够——cognee 内
        # 有 lru_cache 配置单例可能在 import 链中提前快照 env，导致数据漏到
        # cognee 仓库 .env 指向的共享目录，两臂串库）：
        # (1) import 前注入，任何 import 期间构建的配置快照都读到本臂路径；
        # (2) import cognee：其 __init__ 执行 load_dotenv(override=True)，
        #     python-dotenv 从 cognee 源码目录向上找 .env（与 cwd 无关），
        #     可能覆写我们的值；
        # (3) import 后再注入一次顶掉覆写，并 cache_clear 已知配置单例，
        #     令懒构建者重读 env。
        os.environ.update(env)
        import cognee
        from cognee.modules.search.types import SearchType

        os.environ.update(env)
        for cfg_mod in (
            "cognee.base_config",
            "cognee.infrastructure.databases.relational.config",
        ):
            mod = sys.modules.get(cfg_mod)
            if mod is None:
                continue
            for obj in list(vars(mod).values()):
                if callable(getattr(obj, "cache_clear", None)):
                    obj.cache_clear()

        self._cognee = cognee
        self._SearchType = SearchType

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=f"cognee-{self.backend}-loop", daemon=True
        )
        self._thread.start()
        print(f"[cognee-{self.backend}] setup done (work_dir={work_dir})")

    def teardown(self):
        if self._loop is None:
            return
        try:
            self._run(self._close_engines(), timeout=120)
        except Exception as e:  # noqa: BLE001 - teardown must not mask results
            print(f"[cognee-{self.backend}] teardown close warning: {e}")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=15)

    async def _close_engines(self):
        """尽力关引擎：NeuG 必须正常 close（checkpoint 纪律，约束 6）。"""
        try:
            from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

            engine = await get_graph_engine()
            close = getattr(engine, "close", None)
            if close is not None:
                await close()
        except Exception as e:  # noqa: BLE001
            print(f"[cognee-{self.backend}] graph engine close warning: {e}")
        if self.backend == "neug":
            from cognee.infrastructure.databases.neug.connection_manager import (
                get_neug_connection_manager,
            )

            get_neug_connection_manager().shutdown()

    # ---- 摄入：add 逐 session，cognify 全量一次 ----
    def ingest_all(self, sessions):
        for s in sessions:
            self.ingest_session(s["sample_id"], s["session_idx"], s["date_time"], s["text"])
        print(f"[cognee-{self.backend}] cognify over all datasets ...")
        self._run(self._cognify_all())

    def ingest_session(self, sample_id: str, session_idx: int, date_time: str, text: str):
        self._run(self._add_session(sample_id, session_idx, date_time, text))

    async def _add_session(self, sample_id, session_idx, date_time, text):
        # 头部保留 session 序号与原始时间戳，切块后仍可溯源
        header = f"Session {session_idx} on {date_time}\n"
        await self._cognee.add(header + text, dataset_name=sample_id)

    async def _cognify_all(self):
        await self._cognee.cognify(chunk_size=CHUNK_SIZE)

    # ---- 检索：HYBRID_COMPLETION + only_context（补全前的检索上下文）----
    def search(self, question: str, top_k: int = 20):
        hits = self._run(self._search(question, top_k), timeout=600)
        return hits, SearchTrace(extra={"backend": self.backend})

    async def _search(self, question, top_k):
        results = await self._cognee.search(
            query_text=question,
            query_type=self._SearchType.HYBRID_COMPLETION,
            only_context=True,
            top_k=top_k,
        )
        hits = []
        # 返回形状随访问控制/结果条数变化：str / list[str] / list[dict]，逐一兼容
        if isinstance(results, str):
            results = [results]
        for index, item in enumerate(results or []):
            if isinstance(item, dict):
                context = item.get("search_result") or item.get("context_result")
                dataset = item.get("dataset_name") or "cognee"
            elif isinstance(item, str):
                context, dataset = item, "cognee"
            elif isinstance(item, list):
                context = "\n".join(str(c) for c in item if c)
                dataset = "cognee"
            else:
                context = getattr(item, "search_result", None) or getattr(item, "context", None)
                dataset = getattr(item, "dataset_name", None) or "cognee"
            if not context:
                continue
            if isinstance(context, list):
                context = "\n".join(str(c) for c in context if c)
            hits.append(SearchHit(id=f"{dataset}#{index}", score=0.0, payload=str(context)))
        return hits

    # ---- 常驻 loop 提交 ----
    def _run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)


class CogneeNeuGAdapter(_CogneeArm):
    name = "cognee-neug"
    backend = "neug"


class CogneeDefaultAdapter(_CogneeArm):
    name = "cognee-default"
    backend = "default"
