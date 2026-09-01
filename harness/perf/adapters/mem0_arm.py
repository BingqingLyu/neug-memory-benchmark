"""mem0 性能赛道 adapter：with/without NeuG 两臂（mem0-neug / mem0-qdrant）。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕过 LLM 抽取：corpus 的预计算向量/原文经 Memory.vector_store.insert
  （mem0 的存储层写入入口）原样写入；全程 0 次 LLM/网络调用——LLM 与 embedder 只给
  占位配置满足构造，embedding_model 随后替换为直注查询向量的 stub。
- query 走 mem0 检索路径：Memory._search_vector_store——向量 ANN（internal_limit
  over-fetch）+ keyword_search BM25 + sigmoid 归一 + score_and_rank 加性融合，
  即 Memory.search 的检索主体；查询 embedding 步骤换成数据集预计算向量，
  entity boost 通道在直注模式下结构性为空（实体库无写入）。
- graph_multihop：mem0 v2 OSS 检索路径无图遍历 → N/A（契约 §3.5）。
- 关遥测（MEM0_TELEMETRY=False）：避免 NeuG 同进程重复开同一 DB（Error 1004）。

实现说明详见 results/perf/<arm>/NOTES.md。
"""
import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 必须在 import mem0 前

import shutil  # noqa: E402
import uuid  # noqa: E402

from mem0 import Memory  # noqa: E402
from mem0.utils.lemmatization import lemmatize_for_bm25  # noqa: E402

from ..base import FTS_KEYWORD, HYBRID, PerfAdapter, VECTOR_TOPK  # noqa: E402

EMBED_DIMS = 1024
USER_ID = "perf-corpus"      # 隔离键：写入/查询统一按该 user_id 过滤
INSERT_BATCH = 1000          # vector_store.insert 分批大小（qdrant 端批量编码受益）
THRESHOLD = 0.0              # GT 为无条件 top-k/AND，不按阈值丢候选（mem0 默认 0.1）


class _InjectedEmbedder:
    """查询 embedding stub：注入数据集预计算的查询向量，零网络、零模型调用。

    数据集的 query_vectors.npy 与语料向量同为 text-embedding-v3（1024 维、已归一），
    直注后 mem0 检索路径与真实调用完全等价，只是省掉重复的 embedding 计算。
    """

    def __init__(self):
        self.current = None

    def embed(self, text, memory_action=None):
        return list(self.current)

    def embed_batch(self, texts, memory_action=None):
        return [list(self.current) for _ in texts]


class Mem0PerfAdapter(PerfAdapter):
    #: mem0 v2 OSS 检索路径无图遍历 → graph_multihop 由 runner 标 N/A
    supported_classes = frozenset({VECTOR_TOPK, FTS_KEYWORD, HYBRID})

    def __init__(self):
        self.memory = None
        self._embedder = _InjectedEmbedder()
        self._id2sid = {}                          # 后端主键 -> session_id
        self._filters = {"user_id": USER_ID}

    def _vector_store_config(self, work_dir):
        raise NotImplementedError

    def _backend_dirs(self, work_dir):
        raise NotImplementedError

    # ---- 生命周期 ----
    def setup(self, work_dir):
        # 清理上次运行的后端数据，保证每次 load 从空库开始（幂等）
        for path in self._backend_dirs(work_dir) + [os.path.join(work_dir, "history.db")]:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {
                "model": "qwen-plus",
                "api_key": "sk-perf-no-llm-calls",  # 占位：性能赛道全程不调 LLM
                "openai_base_url": "http://127.0.0.1:9/v1",
            }},
            "embedder": {"provider": "openai", "config": {
                "model": "text-embedding-v3",
                "api_key": "sk-perf-no-llm-calls",
                "openai_base_url": "http://127.0.0.1:9/v1",
                "embedding_dims": EMBED_DIMS,
            }},
            "vector_store": self._vector_store_config(work_dir),
            "history_db_path": os.path.join(work_dir, "history.db"),
        })
        # 查询向量由数据集预计算：替换 embedding 模型，检索路径其余部分不变
        self.memory.embedding_model = self._embedder

    def teardown(self):
        # NeuG 连接必须正常 close（checkpoint 纪律）；实体库若被懒加载也一并关
        for store in (getattr(self.memory, "vector_store", None),
                      getattr(self.memory, "_entity_store", None)):
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - teardown 不抛错
                    pass

    # ---- load：预计算语料经 mem0 存储层写入入口直注，不经 LLM ----
    def load(self, corpus):
        store = self.memory.vector_store
        n = len(corpus.session_ids)
        for start in range(0, n, INSERT_BATCH):
            end = min(start + INSERT_BATCH, n)
            ids, payloads = [], []
            for i in range(start, end):
                # qdrant 主键只接受 int/UUID 字符串，两臂统一用 uuid，
                # 行号↔session_id↔后端主键 映射在 adapter 内维护
                mid = str(uuid.uuid4())
                ids.append(mid)
                self._id2sid[mid] = corpus.session_ids[i]
                payloads.append({
                    # data 原文即 FTS 索引字段（NeuG text 列 / qdrant bm25 稀疏向量）
                    "data": corpus.texts[i],
                    "user_id": USER_ID,
                    "session_id": corpus.session_ids[i],
                })
            store.insert(vectors=[list(v) for v in corpus.embeddings[start:end]],
                         payloads=payloads, ids=ids)
            if (start // INSERT_BATCH) % 10 == 9 or end == n:
                print(f"[{self.name}] loaded {end}/{n}", flush=True)

    # ---- 四类查询 ----
    def query_vector(self, qvec, top_k):
        # mem0 检索路径（纯向量分支）：_search_vector_store 空查询文本 →
        # 词法/实体通道自然为空，退化为 vector_store.search + score_and_rank
        self._embedder.current = qvec
        results = self.memory._search_vector_store("", self._filters, top_k, THRESHOLD)
        return self._to_session_ids(results)

    def query_fts(self, keywords, top_k):
        # mem0 关键词通道：keyword_search（NeuG 原生 BM25 / qdrant bm25 稀疏向量），
        # 查询词与 Memory.search 同口径先过 lemmatize_for_bm25（无 spaCy 时原样）
        query = self._lemmatized(" ".join(keywords))
        hits = self.memory.vector_store.keyword_search(
            query=query, top_k=top_k, filters=self._filters) or []
        hits = sorted(hits, key=lambda h: float(getattr(h, "score", 0) or 0), reverse=True)
        return [self._id2sid[str(h.id)] for h in hits if str(h.id) in self._id2sid]

    def query_hybrid(self, qvec, keywords, top_k):
        # mem0 检索路径（融合分支）：_search_vector_store——
        # 语义 ANN + BM25 + sigmoid 归一 + 加性融合（与 Memory.search 同一实现）
        self._embedder.current = qvec
        results = self.memory._search_vector_store(
            " ".join(keywords), self._filters, top_k, THRESHOLD)
        return self._to_session_ids(results)

    # graph_multihop 保持 NotImplementedError（supported_classes 未含）

    # ---- 内部工具 ----
    def _to_session_ids(self, results):
        return [self._id2sid[str(r["id"])] for r in results
                if str(r["id"]) in self._id2sid]

    @staticmethod
    def _lemmatized(text):
        try:
            out = lemmatize_for_bm25(text)
            return out or text
        except Exception:  # noqa: BLE001 - 无 spaCy 环境时退回原文
            return text


class Mem0NeuGPerfAdapter(Mem0PerfAdapter):
    name = "mem0-neug"

    def _vector_store_config(self, work_dir):
        return {"provider": "neug", "config": {
            "collection_name": "mem0",
            "db_path": os.path.join(work_dir, "neug.db"),
            "distance": "cosine",
            "embedding_model_dims": EMBED_DIMS,
        }}

    def _backend_dirs(self, work_dir):
        return [os.path.join(work_dir, "neug.db")]


class Mem0QdrantPerfAdapter(Mem0PerfAdapter):
    name = "mem0-qdrant"

    def _vector_store_config(self, work_dir):
        return {"provider": "qdrant", "config": {
            "collection_name": "mem0",
            "path": os.path.join(work_dir, "qdrant"),
            "embedding_model_dims": EMBED_DIMS,
        }}

    def _backend_dirs(self, work_dir):
        return [os.path.join(work_dir, "qdrant")]
