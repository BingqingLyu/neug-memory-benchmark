"""mem0 性能赛道 adapter：with/without NeuG 多臂（mem0-neug / mem0-qdrant / mem0-qdrant-server）。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕过 LLM 抽取：corpus 的预计算向量/原文经 Memory.vector_store.insert
  （mem0 的存储层写入入口）原样写入；全程 0 次 LLM/网络调用——LLM 与 embedder 只给
  占位配置满足构造，embedding_model 随后替换为直注查询向量的 stub。
- query 走 mem0 检索路径：Memory._search_vector_store——向量 ANN（internal_limit
  over-fetch）+ keyword_search BM25 + sigmoid 归一 + score_and_rank 加性融合，
  即 Memory.search 的检索主体；查询 embedding 步骤换成数据集预计算向量，
  entity boost 通道在直注模式下结构性为空（实体库无写入）。
- graph_multihop：qdrant 臂无图遍历能力 → N/A（契约 §3.5）；NeuG 臂经 neug
  vector store 的原生关系表（mem0_mem0_links）承载规范共现图，query 走变长
  Cypher BFS（*1..2），与 graphiti-neug 同口径——见 Mem0NeuGPerfAdapter。
- 关遥测（MEM0_TELEMETRY=False）：避免 NeuG 同进程重复开同一 DB（Error 1004）。

实现说明详见 results/perf/<arm>/NOTES.md。
"""
import csv  # noqa: E402
import os
import time  # noqa: E402

os.environ.setdefault("MEM0_TELEMETRY", "False")  # 必须在 import mem0 前
# qdrant 臂的 BM25 关键词检索（fts_keyword/hybrid）依赖 fastembed 的 Qdrant/bm25
# 模型，首次使用需从 HuggingFace 拉取。本环境 huggingface.co 不可达，默认走 hf-mirror
# 镜像；setdefault 保留外部覆盖能力。必须在 import mem0（→fastembed→hf_hub）前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import shutil  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402
from mem0 import Memory  # noqa: E402
from mem0.utils.lemmatization import lemmatize_for_bm25  # noqa: E402

from ..base import (  # noqa: E402
    FTS_KEYWORD,
    GRAPH_MULTIHOP,
    HYBRID,
    PerfAdapter,
    VECTOR_TOPK,
)

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
    #: 基类=qdrant 臂口径：无图遍历 → graph_multihop N/A。NeuG 臂在子类里
    #: 覆盖 supported_classes 补上 GRAPH_MULTIHOP（neug store 有原生关系表）。
    supported_classes = frozenset({VECTOR_TOPK, FTS_KEYWORD, HYBRID})
    #: vector_store.insert 分批大小。qdrant server 臂走 REST，单批 payload 受
    #: 32MB 请求体上限约束（见 Mem0QdrantServerPerfAdapter.insert_batch 覆写）。
    insert_batch = INSERT_BATCH

    def __init__(self):
        self.memory = None
        self._embedder = _InjectedEmbedder()
        self._id2sid = {}                          # 后端主键 -> session_id
        self._sid2mid = {}                         # session_id -> 后端主键（建边/BFS 用）
        self._filters = {"user_id": USER_ID}
        self._work_dir = None
        # graph 邻域缓存：warmup 与计时重放同一种子，病态高度数种子只算一次
        self._graph_cache = {}

    def _vector_store_config(self, work_dir):
        raise NotImplementedError

    def _backend_dirs(self, work_dir):
        raise NotImplementedError

    # ---- 生命周期 ----
    def setup(self, work_dir):
        self._work_dir = work_dir
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
        lem = getattr(corpus, "text_lemmatized", None)
        for start in range(0, n, self.insert_batch):
            end = min(start + self.insert_batch, n)
            ids, payloads = [], []
            for i in range(start, end):
                # qdrant 主键只接受 int/UUID 字符串，两臂统一用 uuid，
                # 行号↔session_id↔后端主键 映射在 adapter 内维护
                mid = str(uuid.uuid4())
                ids.append(mid)
                self._id2sid[mid] = corpus.session_ids[i]
                self._sid2mid[corpus.session_ids[i]] = mid
                payload = {
                    # data 原文（向量/图之外的原始文本载荷）
                    "data": corpus.texts[i],
                    "user_id": USER_ID,
                    "session_id": corpus.session_ids[i],
                }
                # mem0 core add() 对所有 backend 都存 text_lemmatized（main.py:1031），fts
                # 的 doc 侧本应 lemmatized。三臂统一写此字段：neug/qdrant 的 `text_lemmatized
                # or data` fallback 会优先取它（doc 从 raw 变 lemmatized，还原真实口径），
                # pgvector 无 fallback、本就只认它。优先用预计算缓存（移出 load 计时），
                # 无缓存才现场 lemmatize 兜底。
                if lem is not None and lem[i] is not None:
                    payload["text_lemmatized"] = lem[i]
                else:
                    payload["text_lemmatized"] = self._lemmatized(corpus.texts[i])
                payloads.append(payload)
            store.insert(vectors=[list(v) for v in corpus.embeddings[start:end]],
                         payloads=payloads, ids=ids)
            if (start // self.insert_batch) % 10 == 9 or end == n:
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
    #: NeuG 臂补上 graph_multihop：neug vector store 有原生关系表 + 变长 BFS。
    #: qdrant 臂无此能力，保持基类的 N/A（能力完整性矩阵，与 semantica 对称：
    #: 原生栈缺的能力由 NeuG 后端补齐）。
    supported_classes = frozenset({VECTOR_TOPK, FTS_KEYWORD, HYBRID, GRAPH_MULTIHOP})

    def _vector_store_config(self, work_dir):
        return {"provider": "neug", "config": {
            "collection_name": "mem0",
            "db_path": os.path.join(work_dir, "neug.db"),
            "distance": "cosine",
            "embedding_model_dims": EMBED_DIMS,
        }}

    def _backend_dirs(self, work_dir):
        return [os.path.join(work_dir, "neug.db")]

    # ---- load：节点走 COPY 批量直注（JSONL），再把共现图 COPY 进关系表 ----
    def load(self, corpus):
        # 覆写基类的 store.insert（多行 CREATE）为单次 COPY 批量载入：
        # COPY 免除逐行 parse/plan/commit，JSONL 又绕开 CSV 反斜杠转义 bug。
        # 表是 index-first（create_col 建表即建 HNSW+FTS），COPY 进已索引表
        # 与 semantica-neug 同路径。id↔session_id 映射与基类逐条一致。
        store = self.memory.vector_store
        n = len(corpus.session_ids)
        ids, payloads = [], []
        for i in range(n):
            mid = str(uuid.uuid4())
            ids.append(mid)
            self._id2sid[mid] = corpus.session_ids[i]
            self._sid2mid[corpus.session_ids[i]] = mid
            payloads.append({
                "data": corpus.texts[i],
                "user_id": USER_ID,
                "session_id": corpus.session_ids[i],
            })
        t0 = time.time()
        store.bulk_insert_copy(vectors=[list(v) for v in corpus.embeddings],
                               payloads=payloads, ids=ids)
        print(f"[{self.name}] nodes COPY done ({n} rows) in {time.time() - t0:.0f}s",
              flush=True)
        self._load_graph_edges(corpus)

    def _load_graph_edges(self, corpus):
        """2.8M 无向共现边按双向各写一条 COPY 进 mem0_mem0_links 关系表。

        不用 neug.add_edge（每条边 2 次 MATCH，2.8M 边不可行）；走 COPY 批量导入
        （与 graphiti-neug 的 RELATES_TO COPY 同路径，探针 probe_mem0_graph.py
        已验证 COPY 进 REL 表 + 变长 BFS 正确）。双向写是为了用有向变长模式
        `*1..2 ->` 覆盖无向 2 跳邻域（graphiti-neug 同款做法）。
        """
        store = self.memory.vector_store
        ids = np.asarray(corpus.session_ids)
        src = ids[corpus.graph_edges[:, 0]]
        dst = ids[corpus.graph_edges[:, 1]]
        sid2mid = self._sid2mid
        csv_dir = os.path.join(self._work_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        path = os.path.join(csv_dir, "graph_edges.csv")
        t0 = time.time()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["from", "to", "relation", "weight"])
            for a, b in zip(src, dst):
                ma, mb = sid2mid[str(a)], sid2mid[str(b)]
                w.writerow([ma, mb, "co_occurs", 1.0])
                w.writerow([mb, ma, "co_occurs", 1.0])
        print(f"[{self.name}] edge csv written ({len(src) * 2} rows) "
              f"in {time.time() - t0:.0f}s, COPY ...", flush=True)
        store._execute(
            f'COPY {store._edge_table} FROM "{path}" (HEADER=true, DELIMITER=",", ESCAPE="")'
        )
        print(f"[{self.name}] graph edges COPY done in {time.time() - t0:.0f}s",
              flush=True)

    def query_graph(self, seed_session_id, max_nodes):
        """变长 Cypher BFS（*1..2）返回完整 2 跳邻域的 session_id 集合。

        契约 §6：真实臂返回完整邻域即 ⊇ GT（GT 已按 hop 独立截断到
        hop1≤50 + hop2≤100），故 max_nodes 形参此处不适用、不做全局截断。
        只 RETURN DISTINCT n.id（不序列化 payload），与 graphiti-neug 的
        driver 级 BFS 同口径；绕开 neug.traverse 的逐节点单跳迭代（高度数
        种子会退化成上千次查询）。双向边会让种子自身经回流边出现在结果里，
        属超集，对 recall_full 无害。
        """
        cached = self._graph_cache.get(seed_session_id)
        if cached is not None:
            return cached
        store = self.memory.vector_store
        mid = self._sid2mid.get(seed_session_id)
        if mid is None:
            self._graph_cache[seed_session_id] = []
            return []
        rows = store._execute(
            f"MATCH (origin:{store.table_name} {{id: $src}})"
            f"-[:{store._edge_table}*1..2]->(n:{store.table_name}) "
            # 先按节点身份去重再投影 id 字符串（WITH DISTINCT n RETURN n.id）：
            # 直接 RETURN DISTINCT n.id 会在整条展开路径多重集上对 36 字符 uuid
            # 字符串做 hash 去重；改写后 NeuG 实测 p50 15.65ms->2.53ms（6.2×），
            # 50 个真实种子结果集逐一致（0 处不匹配）。
            f"WITH DISTINCT n RETURN n.id AS id",
            {"src": mid},
        )
        sids = [self._id2sid[r["id"]] for r in rows if r["id"] in self._id2sid]
        self._graph_cache[seed_session_id] = sids
        return sids


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

    def load(self, corpus):
        super().load(corpus)
        # 载入后自检 BM25 能力：insert 已触发编码器 lazy-load，此时状态已确定。
        self._verify_bm25_encoder()

    def _verify_bm25_encoder(self):
        """qdrant FTS 能力自检，不可用则 fail loudly（而非静默记 recall=0）。

        keyword_search 依赖 fastembed 的 Qdrant/bm25 稀疏向量。若编码器加载失败
        （fastembed 缺失，或 HF 不可达且模型未缓存），mem0 只记 warning 不抛错——
        insert 会静默跳过 bm25 稀疏向量、keyword_search 查空，fts_keyword/hybrid 的
        recall 被误记成 0.0000（本赛道曾因此产出 qdrant fts=0 的假结果）。这里主动
        探测：既然声明支持 FTS_KEYWORD，就必须真具备 BM25 能力，否则报错并给指引。
        """
        store = self.memory.vector_store
        has_slot = getattr(store, "_has_bm25_slot", False)
        encoder = store._get_bm25_encoder() if has_slot else None
        if has_slot and encoder is not None:
            return
        reason = ("collection 无 bm25 稀疏槽（pre-v3 结构）" if not has_slot
                  else "fastembed 的 Qdrant/bm25 编码器加载失败")
        raise RuntimeError(
            f"[{self.name}] BM25 关键词能力不可用（{reason}）：fts_keyword/hybrid 会被"
            f"静默记成 recall=0。Qdrant/bm25 模型首次使用需从 HuggingFace 拉取，本环境"
            f" huggingface.co 不可达——请设 HF_ENDPOINT=https://hf-mirror.com（本模块已"
            f"默认设置，若被外部覆盖请检查）并确保已安装 fastembed，然后重跑。"
        )


class Mem0QdrantServerPerfAdapter(Mem0QdrantPerfAdapter):
    """qdrant server 模式臂：连本机 docker qdrant（host/port），建真 HNSW 索引。

    与 mem0-qdrant（local 模式，纯 Python 暴力扫、默认不建 ANN 索引）的关键区别：
    - server 模式 is_local=False → create_col 额外走 _create_filter_indexes（payload
      过滤索引），且向量集合按 qdrant 默认 HNSW（实测 1.19.1：m=16 / ef_construct=100 /
      indexing_threshold=10000）在点数超阈值后异步建 ANN 索引——这才是“建了索引的
      生产级 qdrant”，堵住“neug 只赢在跟没建索引的 local 模式比”的公平性质疑。
    - load() 直注后阻塞等 status=green（优化器收敛、所有段合并且 HNSW 建完），把构建
      耗时计入 load_seconds_bg，与 NeuG index-first（COPY 进已建 HNSW 表）的 eager 成本同口径。
    - BM25 关键词能力与 local 臂同一套（fastembed Qdrant/bm25 稀疏向量 + HF_ENDPOINT
      镜像），继承 _verify_bm25_encoder 自检；无图遍历 → graph_multihop 仍 N/A。
    连接信息用 BENCH_QDRANT_HOST / BENCH_QDRANT_PORT 覆盖（默认 localhost:6333）。
    """
    name = "mem0-qdrant-server"
    #: qdrant server 走 REST upsert，单请求体默认上限 32MB(service.max_request_size_mb)。
    #: 每点约 42.5KB（1024维 dense 的 JSON + bm25 稀疏向量 + text payload），基类的
    #: 1000/批 ≈ 42.5MB 会被 400 Bad Request 拒。降到 256/批（≈11MB）留足余量。
    #: load 是背景成本、非对比指标（run_perf.py 明载），批大小不影响查询延时/recall 的
    #: 公平性；HNSW 由 qdrant 优化器按总点数异步构建，与 upsert 批大小无关。
    insert_batch = 256

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

    def _backend_dirs(self, work_dir):
        # server 模式数据在容器里，无本地目录可清；history.db 仍由基类 setup 清理。
        return []

    def setup(self, work_dir):
        # create_col 遇已存在集合会跳过不重建（qdrant.py:137）→ 每次跑前先 drop，
        # 保证从空库开始（幂等）。entity_store 惰性且直注模式不触发，防御性一并 drop。
        self._drop_server_collections()
        super().setup(work_dir)

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

    def load(self, corpus):
        super().load(corpus)   # 直注 + BM25 自检（继承 Mem0QdrantPerfAdapter）
        self._wait_hnsw_index(len(corpus.session_ids))

    def _wait_hnsw_index(self, expected_points, timeout_s=900, poll_s=3,
                         stable_polls=3):
        """阻塞等 qdrant 后台优化器把 HNSW 建完并收敛（status=green）。

        收敛判据用 status==green 而非 indexed_vectors_count>=points_count：后者在
        qdrant 里会被 dense+bm25 双向量各计一次而虚高（实测 green 时 indexed=2×points），
        一插入就恒成立、会 0s 误判“建完”。green 表示所有段已合并且 HNSW 构建完成，
        才是“建了索引的生产级 qdrant”的可靠信号。51661 点 > indexing_threshold(10000)
        会触发异步建 HNSW；不等建完就计时查询会测到“半建索引”的失真延迟。等待计入
        load_seconds_bg，与 NeuG index-first eager 构建同口径对比（两者都在 load 阶段
        付清 ANN 索引成本）。
        """
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
                print(f"[{self.name}] HNSW indexing: points={points} "
                      f"indexed={indexed} status={status} ({elapsed:.0f}s)", flush=True)
                last = sig
                stable = 0
            else:
                stable += 1
            if status == "green" and points >= expected_points and stable >= stable_polls:
                print(f"[{self.name}] HNSW converged: points={points} indexed={indexed} "
                      f"status=green in {elapsed:.0f}s", flush=True)
                return
            if elapsed > timeout_s:
                print(f"[{self.name}] WARN HNSW indexing timed out (points={points} "
                      f"indexed={indexed} status={status}) after {timeout_s}s; proceeding",
                      flush=True)
                return
            time.sleep(poll_s)


class Mem0PgvectorPerfAdapter(Mem0PerfAdapter):
    """pgvector server 模式臂：连本机 docker postgres+pgvector（host/port），建 HNSW + GIN 索引。

    与 mem0-qdrant-server 同为“生产级服务型后端”公平基线（都建了真索引），堵住“neug 只赢在
    跟没建索引的 local 模式比”的质疑。关键机制：
    - vector_topk：pgvector HNSW（vector_cosine_ops；create_col 时 CREATE INDEX 建空索引，
      insert 逐行增量维护）；查询走 `vector <=> %s::vector ORDER BY distance LIMIT k`。
    - fts_keyword：**第三套 fts 实现**——Postgres 原生全文检索 to_tsvector/plainto_tsquery/
      ts_rank_cd + GIN 索引。语义是 plainto_tsquery 的 **AND**（区别于 neug FTS5 的 OR 展开、
      qdrant fastembed 的 soft-OR 稀疏向量），且 'simple' 配置不再 stem/去停用词（doc 侧已由
      text_lemmatized 预 lemmatize、query 侧 query_fts 亦 lemmatize，两侧口径一致）。
    - **payload 必须带 text_lemmatized**：pgvector.keyword_search 直接查 payload->>'text_lemmatized'
      （GIN 索引亦建于此表达式），**无 neug/qdrant 的 `or data` fallback**。直注绕过 mem0 core
      add()（main.py:1031 才写 text_lemmatized），故本臂 load 显式补该字段，否则 fts recall=0；
      这也让 doc 侧口径对齐 mem0 真实行为（core add 对所有 backend 都存 text_lemmatized）。
    - 无图遍历 → graph_multihop 保持基类 N/A。
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

    def _backend_dirs(self, work_dir):
        # server 模式数据在容器里，无本地目录可清；history.db 仍由基类 setup 清理。
        return []

    def setup(self, work_dir):
        # create_col 用 CREATE TABLE IF NOT EXISTS，遇已存在表跳过不重建 → 每次跑前先 drop，
        # 保证从空库开始（幂等）。HNSW/GIN 索引随表一并 drop。
        self._drop_table()
        super().setup(work_dir)
        # 预建 entity_store 的 mem0_entities 表：mem0 _compute_entity_boosts 用
        # ThreadPoolExecutor(max_workers=4) 并发搜实体（main.py:1778）。PGVector.__init__ 不建表
        # （_collection_ensured=False，pgvector.py:188），建表推迟到首次 search 的
        # _ensure_collection→create_col（pgvector.py:216/350）；该守卫无锁、CREATE TABLE IF NOT
        # EXISTS 检查-创建又非原子 → query 期 4 线程首次并发 search 各自 create_col 竞争，撞
        # pg_type UniqueViolation（手动单线程 drop→create→re-create 全程不撞，坐实是并发竞争
        # 而非 orphan type）。故 setup 单线程强制 _ensure_collection 建表 + 置位，query 期 4
        # 线程直接跳过 create。直注模式 entity_store 无写入，entity boost 结构性为空。
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
                    # mem0 entity_store 惰性建 mem0_entities 表（entity boost 通道），与
                    # qdrant-server 臂同口径一并 drop：否则残留表会让下次 create_col 撞
                    # pg_type UniqueViolation（冒烟实测 mem0_entities duplicate key）。
                    for tbl in ("mem0", "mem0_entities"):
                        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(
                            sql.Identifier(tbl)))
            print(f"[{self.name}] dropped stale tables mem0/mem0_entities", flush=True)
        except Exception as e:  # noqa: BLE001 - 表不存在/连接未就绪都容忍
            print(f"[{self.name}] drop table skipped: {e}", flush=True)

    def load(self, corpus):
        # base load 已统一写 text_lemmatized（lemmatized-doc，三臂同口径，pgvector fts 唯一
        # 索引列）；pgvector 额外在 insert 后 ANALYZE 刷新 planner 统计，免得查询计划沿用
        # 空表统计而失真。HNSW/GIN 在 create_col 建好、随 insert 增量维护（实测 ~2ms/row，
        # 全量 ~100s，与 qdrant-server 同量级），计入 load_seconds_bg（背景成本，非对比指标）。
        super().load(corpus)
        self._analyze(self.memory.vector_store)

    def _analyze(self, store):
        from psycopg import sql
        try:
            with store._get_cursor(commit=True) as cur:
                cur.execute(sql.SQL("ANALYZE {}").format(store._col()))
            print(f"[{self.name}] ANALYZE mem0 done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] ANALYZE skipped: {e}", flush=True)
