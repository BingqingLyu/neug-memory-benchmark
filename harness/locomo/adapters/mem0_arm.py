"""mem0 系统 LoCoMo adapter：with/without NeuG 两臂（mem0-neug / mem0-qdrant）。

协议对齐（ADAPTER-CONTRACT.md）：
- 会话隔离：摄入按 user_id=sample_id；检索优先用题目携带的 sample_id 只查该
  样本（约束1，与 graphiti/semantica 同口径），避免逐样本 N× 检索与跨样本合并
  污染；仅当 harness 未传 sample_id 时才退回逐 sample 查再按分合并取 top-k
- 抽取与 embedding 的 base_url 都指向缓存代理 → 换后端时抽取结果共享
- embedding 用 DashScope text-embedding-v3（1024 维），抽取 LLM 用 qwen-plus
- 后端切换是唯一变量：两臂除 vector_store 配置外完全一致
- 关 telemetry（MEM0_TELEMETRY=False），避免重复连接打开同一 NeuG DB（Error 1004）
"""
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 必须在 import mem0 前

from mem0 import Memory  # noqa: E402

from ..base import SearchHit, SearchTrace, SystemAdapter  # noqa: E402

PROXY_BASE_URL = os.environ.get("BENCH_LLM_PROXY_URL", "http://127.0.0.1:8787/v1")
LLM_MODEL = os.environ.get("BENCH_MEM0_LLM_MODEL", "qwen-plus")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-v3")
EMBED_DIMS = 1024


class Mem0ArmAdapter(SystemAdapter):
    sub_scenario = "A"

    def __init__(self):
        self.memory = None
        self.sample_ids: list[str] = []

    def _vector_store_config(self, work_dir: str) -> dict:
        raise NotImplementedError

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {
                "model": LLM_MODEL,
                "api_key": api_key,
                "openai_base_url": PROXY_BASE_URL,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": EMBED_MODEL,
                "api_key": api_key,
                "openai_base_url": PROXY_BASE_URL,
                "embedding_dims": EMBED_DIMS,
            }},
            "vector_store": self._vector_store_config(work_dir),
            "history_db_path": os.path.join(work_dir, "history.db"),
        })

    def teardown(self):
        # NeuG 连接必须正常 close（checkpoint 纪律）；qdrant 无 close 则跳过
        store = getattr(self.memory, "vector_store", None)
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - teardown 不抛错
                pass

    # ---- 摄入：按 session 增量，隔离键 user_id=sample_id ----
    def ingest_session(self, sample_id, session_idx, date_time, text):
        if sample_id not in self.sample_ids:
            self.sample_ids.append(sample_id)
        # OSS SDK 的 add 不支持 timestamp 参数，会话时间随 metadata 落盘
        self.memory.add(
            [{"role": "user", "content": text}],
            user_id=sample_id,
            metadata={"sample_id": sample_id, "session_idx": session_idx,
                      "date_time": date_time},
        )

    def reuse_ingested(self, sessions):
        # search() 逐 sample 带 user_id 隔离；恢复 ingest_session 期间
        # 收集的样本列表（保持摄入顺序去重）。
        self.sample_ids = list(dict.fromkeys(s["sample_id"] for s in sessions))

    # ---- 检索：优先按题目 sample_id 隔离只查该样本；缺省才逐 sample 合并 ----
    def search(self, question, top_k=20, sample_id=None):
        # 约束1 会话隔离：run_eval 逐题传 sample_id（timed_search 探测签名后转发），
        # 用它只查题目所属样本——与 graphiti/semantica 同口径。不声明 sample_id 会
        # 退化成遍历全部样本各查一次再跨样本合并：检索开销 ×N（LoCoMo N=10），且
        # 其他样本的记忆会挤进 top-k 污染隔离。缺省（旧调用方）才保留逐样本合并。
        sids = [sample_id] if sample_id else self.sample_ids
        hits: list[SearchHit] = []
        for sid in sids:
            res = self.memory.search(
                question, top_k=top_k, threshold=0.0,
                filters={"user_id": sid},
            )
            for r in res.get("results", []):
                hits.append(SearchHit(
                    id=str(r.get("id", "")),
                    score=float(r.get("score") or 0.0),
                    payload=str(r.get("memory") or ""),
                ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k], SearchTrace(extra={"n_samples": len(sids)})


class Mem0NeuGAdapter(Mem0ArmAdapter):
    name = "mem0-neug"

    def _vector_store_config(self, work_dir):
        return {"provider": "neug", "config": {
            "collection_name": "mem0",
            "db_path": os.path.join(work_dir, "neug.db"),
            "distance": "cosine",
            "embedding_model_dims": EMBED_DIMS,
        }}


class Mem0QdrantAdapter(Mem0ArmAdapter):
    name = "mem0-qdrant"

    def _vector_store_config(self, work_dir):
        return {"provider": "qdrant", "config": {
            "collection_name": "mem0",
            "path": os.path.join(work_dir, "qdrant"),
            "embedding_model_dims": EMBED_DIMS,
        }}
