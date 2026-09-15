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
import time

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 必须在 import mem0 前
# qdrant 臂的 BM25 关键词检索依赖 fastembed 的 Qdrant/bm25 模型，首次使用需从
# HuggingFace 拉取；本环境 huggingface.co 不可达，默认走 hf-mirror 镜像（与 perf
# 赛道 mem0_arm 同策略）。必须在 import mem0（→fastembed→hf_hub）前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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


class Mem0QdrantServerAdapter(Mem0ArmAdapter):
    """qdrant server 模式臂（质量赛道）：连本机 docker qdrant（host/port），建真索引。

    与性能赛道 `Mem0QdrantServerPerfAdapter` 同口径，作为质量赛道的**生产级公平基线**
    （替代 local 模式）——让质量赛道的 compared systems 与性能赛道一致（neug /
    qdrant-server / pgvector）。关键差异（vs `Mem0QdrantAdapter` local）：
    - server 模式 is_local=False → create_col 额外走 _create_filter_indexes（payload
      过滤索引），点数超 indexing_threshold 后异步建 HNSW；
    - 数据在容器里、不在 work_dir，故 drop/复用判据用 work_dir 的 .ingest_complete
      marker：全新摄入才 drop 外部 collection，复用 run（marker 存在）保留容器数据。
    连接信息用 BENCH_QDRANT_HOST / BENCH_QDRANT_PORT 覆盖（默认 localhost:6333）。
    """
    name = "mem0-qdrant-server"

    def _host_port(self):
        host = os.environ.get("BENCH_QDRANT_HOST", "localhost")
        port = int(os.environ.get("BENCH_QDRANT_PORT", "6333"))
        return host, port

    def _vector_store_config(self, work_dir):
        host, port = self._host_port()
        return {"provider": "qdrant", "config": {
            "collection_name": "mem0",
            "host": host,
            "port": port,
            "embedding_model_dims": EMBED_DIMS,
        }}

    def setup(self, work_dir, cache_dir):
        # create_col 遇已存在集合会跳过不重建 → 全新摄入前先 drop，保证从空库开始。
        # 复用 run（marker 存在）不 drop，否则会把容器内已摄入数据清掉、search 查空。
        if not os.path.exists(os.path.join(work_dir, ".ingest_complete")):
            self._drop_server_collections()
        super().setup(work_dir, cache_dir)

    def _drop_server_collections(self):
        from qdrant_client import QdrantClient
        host, port = self._host_port()
        client = QdrantClient(host=host, port=port)
        try:
            for name in ("mem0", "mem0_entities"):
                try:
                    if client.collection_exists(name):
                        client.delete_collection(name)
                        print(f"[{self.name}] dropped stale collection {name}",
                              flush=True)
                except Exception as e:  # noqa: BLE001 - 不存在/已删都容忍
                    print(f"[{self.name}] drop {name} skipped: {e}", flush=True)
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    def ingest_all(self, sessions):
        super().ingest_all(sessions)
        # 摄入后等 qdrant 后台优化器收敛（status=green），与性能赛道同口径，免得
        # 查询与优化器抢 CPU 测到失真延迟。LoCoMo 记忆量通常 < indexing_threshold
        # （qdrant 默认对小集合延迟建 HNSW、走精确暴力扫），green 会很快到达。
        self._wait_hnsw_index()

    def _wait_hnsw_index(self, timeout_s=600, poll_s=3, stable_polls=3):
        store = self.memory.vector_store
        client = store.client
        t0 = time.time()
        last = None
        stable = 0
        while True:
            info = client.get_collection(store.collection_name)
            points = getattr(info, "points_count", 0) or 0
            indexed = getattr(info, "indexed_vectors_count", 0) or 0
            status = str(getattr(info, "status", "")).lower()
            elapsed = time.time() - t0
            sig = (points, indexed, status)
            if sig != last:
                print(f"[{self.name}] qdrant indexing: points={points} "
                      f"indexed={indexed} status={status} ({elapsed:.0f}s)", flush=True)
                last = sig
                stable = 0
            else:
                stable += 1
            if status == "green" and stable >= stable_polls:
                print(f"[{self.name}] qdrant converged: points={points} "
                      f"indexed={indexed} status=green in {elapsed:.0f}s", flush=True)
                return
            if elapsed > timeout_s:
                print(f"[{self.name}] WARN qdrant indexing timed out "
                      f"(status={status}) after {timeout_s}s; proceeding", flush=True)
                return
            time.sleep(poll_s)


class Mem0PgvectorAdapter(Mem0ArmAdapter):
    """pgvector server 模式臂（质量赛道）：连本机 docker postgres+pgvector，建 HNSW+GIN。

    与性能赛道 `Mem0PgvectorPerfAdapter` 同口径，作为质量赛道第二个**生产级公平基线**。
    关键机制：
    - vector：pgvector HNSW（vector_cosine_ops）；fts：Postgres 原生全文 to_tsvector/
      plainto_tsquery + GIN（AND 语义）。LoCoMo 走真实 memory.add()，mem0 core
      （main.py:1031）对所有 backend 都写 text_lemmatized，故 pgvector fts 的 doc 侧
      口径天然对齐（无需像性能赛道直注那样手动补该字段）。
    - entity_store 并发建表撞 pg_type 的修复（与性能赛道同）：mem0 _compute_entity_boosts
      用 ThreadPoolExecutor(max_workers=4) 并发搜实体（main.py:1778），PGVector.__init__
      不建表、建表推迟到首次 search 的 _ensure_collection（无锁守卫）→ 4 线程竞争
      create_col 撞 pg_type UniqueViolation。setup 单线程强制建表 + 置位，query 期跳过。
    - 数据在容器里，drop/复用判据同 qdrant-server 臂（work_dir 的 .ingest_complete marker）。
    连接信息用 BENCH_PGVECTOR_{HOST,PORT,USER,PASSWORD,DB} 覆盖（默认 localhost:5432 mem0/mem0/mem0）。
    """
    name = "mem0-pgvector"

    def _conn_params(self):
        return {
            "host": os.environ.get("BENCH_PGVECTOR_HOST", "localhost"),
            "port": int(os.environ.get("BENCH_PGVECTOR_PORT", "5432")),
            "user": os.environ.get("BENCH_PGVECTOR_USER", "mem0"),
            "password": os.environ.get("BENCH_PGVECTOR_PASSWORD", "mem0"),
            "dbname": os.environ.get("BENCH_PGVECTOR_DB", "mem0"),
        }

    def _conninfo(self):
        p = self._conn_params()
        return (f"postgresql://{p['user']}:{p['password']}@"
                f"{p['host']}:{p['port']}/{p['dbname']}")

    def _vector_store_config(self, work_dir):
        p = self._conn_params()
        return {"provider": "pgvector", "config": {
            "collection_name": "mem0",
            "dbname": p["dbname"],
            "user": p["user"],
            "password": p["password"],
            "host": p["host"],
            "port": p["port"],
            "embedding_model_dims": EMBED_DIMS,
            "diskann": False,
            "hnsw": True,
        }}

    def setup(self, work_dir, cache_dir):
        # create_col 用 CREATE TABLE IF NOT EXISTS，遇已存在表跳过不重建 → 全新摄入前
        # 先 drop（HNSW/GIN 索引随表一并 drop）。复用 run（marker 存在）不 drop。
        if not os.path.exists(os.path.join(work_dir, ".ingest_complete")):
            self._drop_table()
        super().setup(work_dir, cache_dir)
        # 预建 entity_store 的 mem0_entities 表，规避 query 期 4 线程并发 create_col 撞
        # pg_type UniqueViolation（机制见类 docstring）。
        try:
            self.memory.entity_store._ensure_collection()
            print(f"[{self.name}] entity_store pre-created (mem0_entities)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] entity_store pre-create skipped: {e}", flush=True)

    def _drop_table(self):
        import psycopg
        from psycopg import sql
        try:
            with psycopg.connect(self._conninfo(), autocommit=True) as conn:
                with conn.cursor() as cur:
                    for tbl in ("mem0", "mem0_entities"):
                        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(
                            sql.Identifier(tbl)))
            print(f"[{self.name}] dropped stale tables mem0/mem0_entities", flush=True)
        except Exception as e:  # noqa: BLE001 - 表不存在/连接未就绪都容忍
            print(f"[{self.name}] drop table skipped: {e}", flush=True)

    def ingest_all(self, sessions):
        super().ingest_all(sessions)
        # insert 完 ANALYZE 刷新 planner 统计，免得查询计划沿用空表统计而失真。
        self._analyze()

    def _analyze(self):
        from psycopg import sql
        try:
            store = self.memory.vector_store
            with store._get_cursor(commit=True) as cur:
                cur.execute(sql.SQL("ANALYZE {}").format(store._col()))
            print(f"[{self.name}] ANALYZE mem0 done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] ANALYZE skipped: {e}", flush=True)
