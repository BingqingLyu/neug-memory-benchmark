"""cognee 性能赛道 adapter：with/without NeuG 两臂（cognee-neug / cognee-lancedb）。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕过 LLM 抽取：预计算向量经 ``vector_engine.upsert_raw_vectors``
  （NeuG 臂内部走 JSONL COPY FROM 批量写入；LanceDB 臂走 lance 批量 add）原样写入
  ``DocumentChunk_text`` 集合；规范共现图经 ``COPY ... FROM`` 批量写入图库
  （NeuG：JSONL COPY；Ladybug/Kuzu：CSV COPY），全程 0 次 LLM/网络调用。
  查询 embedding 步骤替换为直注数据集查询向量的 stub（patch 向量引擎工厂的
  ``get_embedding_engine``），检索路径其余部分不变。
- query 走 cognee 检索路径：
  * vector  → ChunksRetriever（vector_engine.search "DocumentChunk_text"）
  * fts     → SearchType.CHUNKS_LEXICAL 工厂：NeuG 臂为原生 bm25
              （NeuGFTSChunksRetriever），LanceDB 臂为内存 BM25
              （BM25ChunksRetriever，从图引擎全量拉 DocumentChunk）
  * hybrid  → HybridRetriever（chunk 向量通道 + entity/summary 通道；直注模式下
              Entity/TextSummary/EdgeType 集合结构性为空，通道 fail-open）
  * graph   → graph_engine.query 变长路径（[:EDGE*1..2] 无向 2 跳邻域；完整邻域 ⊇ GT）
- 四类查询全支持（契约 §3.5）。
- teardown 正常 close 向量/图引擎（NeuG 引用计数归零时 checkpoint 关库）。

重要（环境变量隔离）：``cognee/__init__.py`` 在 import 时执行
``dotenv.load_dotenv(override=True)``，会用 CWD 向上找到的第一个 .env **覆盖**
os.environ（运行目录向上会捞到 cognee 仓库根 .env，其中指向真实 debug 库）。
因此 __init__ 必须先触发一次性 ``import cognee`` 把 dotenv 副作用消化掉，
紧接着再用 ``_apply_env`` 覆盖回本臂隔离路径；所有引擎/配置构建都在其后发生。
启动时另有 ``_verify_env_isolation`` 自检，路径逃逸 work_dir 即报错。

实现说明详见 results/perf/<arm>/NOTES.md。
"""
import asyncio
import csv
import json
import os
import shutil
import threading
import uuid as _uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from ..base import ALL_CLASSES, PerfAdapter

EMBED_DIMS = 1024
# NeuG 集合表 text/payload 列为 VARCHAR(65535)，语料最长文本 78135 字符；
# 双臂统一截断（留 payload JSON 余量），保证 with/without 数据一致可比。
MAX_STORED_TEXT_CHARS = 65000
VECTOR_BATCH = 4096          # upsert_raw_vectors 分批（NeuG 侧即一批一次 COPY）
NEUG_EDGE_BATCH = 200_000    # NeuG EDGE JSONL COPY 分批
LADYBUG_EDGE_SHARD = 500_000  # Ladybug EDGE CSV 分片（每片一次 COPY）
GRAPH_DEPTH = 2              # graph_multihop GT = 2 跳邻域


def _chunk_id(i: int) -> str:
    """确定性 UUID 形式的 chunk 主键。

    检索结果经 ``ScoredResult(id=parse_id(...))`` 构造，其 ``id`` 是 pydantic
    UUID 硬校验，``"perf-{i}"`` 过不了；改用 ``uuid.UUID(int=i)`` 既确定性可复现，
    又兼容两臂全部 id 解析路径。回映射表存同形式的字符串。
    """
    return str(_uuid.UUID(int=i))


_COGNEE_IMPORT_GUARD = False


def _consume_cognee_dotenv():
    """一次性 ``import cognee``，消化其 load_dotenv(override=True) 副作用。
    必须在 _apply_env 之前调用（幂等）；之后再导入不会再跑 dotenv。"""
    global _COGNEE_IMPORT_GUARD
    if _COGNEE_IMPORT_GUARD:
        return
    import cognee  # noqa: F401

    _COGNEE_IMPORT_GUARD = True


_PAYLOAD_SCHEMA_CLS = None


def _payload_schema_cls():
    """DocumentChunk_text 集合的 payload schema（upsert_raw_vectors 要求）。
    懒构建且全局单例：不能在顶层触碰 cognee/pydantic，且 LanceDB 侧按
    schema 类缓存 DataPoint 模型，避免逐批铸新类。"""
    global _PAYLOAD_SCHEMA_CLS
    if _PAYLOAD_SCHEMA_CLS is not None:
        return _PAYLOAD_SCHEMA_CLS
    from typing import List

    from pydantic import BaseModel

    class PerfChunkPayload(BaseModel):
        id: str
        text: str
        session_id: str = ""
        document_id: str = ""
        chunk_index: int = 0
        importance_weight: float = 0.5
        # 裸 ``list`` 无法转 Arrow 类型（LanceDB 臂会 TypeError），必须参数化；
        # NeuG 臂不受影响。
        belongs_to_set: List[str] = []

    _PAYLOAD_SCHEMA_CLS = PerfChunkPayload
    return PerfChunkPayload


class _EmbedderStub:
    """查询 embedding stub：注入数据集预计算的查询向量，零网络、零模型调用。

    数据集的 query_vectors.npy 与语料向量同为 text-embedding-v3（1024 维、已归一），
    直注后检索路径与真实调用完全等价，只是省掉重复的 embedding 计算。
    """

    def __init__(self):
        self.current = None

    async def embed_text(self, texts, *args, **kwargs):
        # numpy 标量不是 Python float，NeuG 参数序列化会拒绝（Unsupported
        # parameter type），这里显式转成原生 float 列表。
        vector = [float(x) for x in self.current]
        return [vector for _ in texts]

    def get_vector_size(self):
        return EMBED_DIMS

    def get_batch_size(self):
        return 64


def _patch_embedding_factory(stub):
    """把向量引擎工厂的 embedding 引擎替换为 stub。

    ``_create_vector_engine`` 通过模块级名字绑定调用 ``get_embedding_engine()``，
    替换该名字即可让任何新建的向量适配器拿到 stub（与 mem0 arm 同模式）。
    """
    from cognee.infrastructure.databases.vector import create_vector_engine as cve_module

    cve_module.get_embedding_engine = lambda: stub


class CogneePerfAdapter(PerfAdapter):
    #: cognee 四类查询全支持（契约 §3.5）
    supported_classes = frozenset(ALL_CLASSES)

    def __init__(self):
        self._loop = None
        self._thread = None
        self._embedder = _EmbedderStub()
        self._chunk2sid = {}            # chunk 主键（UUID 串，见 _chunk_id）-> session_id
        self._sid2chunk = {}            # session_id -> chunk 主键
        self._lexical = None            # CHUNKS_LEXICAL retriever 实例（缓存复用）
        # 顺序严格：先消化 cognee 的 load_dotenv(override=True)，再覆盖回
        # 隔离环境变量；任何配置单例/连接都在其后才实例化。
        self._work_dir = None
        _consume_cognee_dotenv()
        self._apply_env(None)

    # ---- 子类提供：后端选型与目录 ----
    def _provider_env(self, work_dir):
        raise NotImplementedError

    def _backend_dirs(self, work_dir):
        raise NotImplementedError

    # ---- 生命周期 ----
    def _apply_env(self, work_dir):
        """设置（或刷新）全部环境变量。

        必须在 ``import cognee``（其 load_dotenv(override=True) 会覆盖
        os.environ）之后、任何配置实例化之前执行，且显式覆盖所有路径/
        后端开关，不给外部 .env 留可污染的键。进程 env 优先于 .env 文件。
        """
        env = {
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",   # 单租户，走共享库
            "CACHING": "false",                         # 关会话缓存层
            "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
            "VECTOR_DB_SUBPROCESS_ENABLED": "false",
            "TELEMETRY_DISABLED": "true",
            "OTEL_TRACES_ENABLED": "false",
            "LOG_COLORS": "false",
        }
        if work_dir is not None:
            roots = os.path.join(work_dir, "cognee_roots")
            env.update({
                "DATA_ROOT_DIRECTORY": roots,
                "SYSTEM_ROOT_DIRECTORY": roots,
                "CACHE_ROOT_DIRECTORY": roots,
                "COGNEE_LOGS_DIR": os.path.join(roots, "logs"),
                "NEUG_DB_PATH": os.path.join(work_dir, "neug_db"),  # 兜底防回落
            })
            env.update(self._provider_env(work_dir))
        for key, value in env.items():
            os.environ[key] = str(value)

    def setup(self, work_dir):
        self._work_dir = work_dir
        # 清理上次运行的后端数据，保证每次 load 从空库开始（幂等）
        for path in self._backend_dirs(work_dir):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

        self._apply_env(work_dir)
        self._reset_cognee_config_caches()

        # cognee 全异步：专用 event loop 线程桥接同步接口（此刻才允许 import）
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="cognee-perf-loop", daemon=True
        )
        self._thread.start()

        _patch_embedding_factory(self._embedder)
        self._verify_env_isolation()

    @staticmethod
    def _reset_cognee_config_caches():
        """清空配置单例的 lru_cache。

        ``import cognee`` 链路上可能已有配置实例定型（读的是被 dotenv
        污染的 env），覆盖环境变量后必须强制重建，否则目录/后端选型
        仍用旧值。
        """
        from cognee.base_config import get_base_config
        from cognee.infrastructure.databases.graph.config import get_graph_config
        from cognee.infrastructure.databases.vector.config import get_vectordb_config

        for getter in (get_base_config, get_graph_config, get_vectordb_config):
            cache_clear = getattr(getter, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    def _verify_env_isolation(self):
        """启动时自检：确认配置解析到 work_dir，防止误写真实库。"""
        from cognee.base_config import get_base_config

        base = get_base_config()
        work_dir = os.path.abspath(self._work_dir)
        for path in (
            base.data_root_directory,
            base.system_root_directory,
            base.cache_root_directory,
        ):
            resolved = os.path.abspath(path)
            if os.path.commonpath([resolved, work_dir]) != work_dir:
                raise RuntimeError(
                    f"[{self.name}] cognee root escapes work_dir: {resolved} "
                    f"(check DATA/SYSTEM/CACHE_ROOT_DIRECTORY env overrides)"
                )
        if self.name == "cognee-neug":
            from cognee.infrastructure.databases.neug.connection_manager import (
                resolve_neug_db_path,
            )

            resolved = os.path.abspath(resolve_neug_db_path())
            if os.path.commonpath([resolved, work_dir]) != work_dir:
                raise RuntimeError(f"[{self.name}] NEUG_DB_PATH escapes work_dir: {resolved}")

    def teardown(self):
        # NeuG 连接必须正常 close（checkpoint 纪律）：向量/图引擎依次关闭，
        # NeuG 连接管理器引用计数归零时真正关库
        async def _close_all():
            from cognee.infrastructure.databases.graph import get_graph_engine
            from cognee.infrastructure.databases.vector import get_vector_engine_async

            for getter in (get_vector_engine_async, get_graph_engine):
                try:
                    engine = await getter()
                    close = getattr(engine, "close", None)
                    if close is not None:
                        await close()
                except Exception:  # noqa: BLE001 - teardown 不抛错
                    pass

        try:
            self._run(_close_all())
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=10)

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---- load：预计算语料存储层直注，全程无 LLM ----
    def load(self, corpus):
        n = len(corpus.session_ids)
        for i, sid in enumerate(corpus.session_ids):
            chunk_id = _chunk_id(i)
            self._chunk2sid[chunk_id] = sid
            self._sid2chunk[sid] = chunk_id

        texts = [(text or "")[:MAX_STORED_TEXT_CHARS] for text in corpus.texts]

        async def _load_all():
            from cognee.infrastructure.databases.unified import get_unified_engine

            unified = await get_unified_engine()
            # 先触图：确保图库连接与 Node/EDGE 表 schema 就绪
            # （NeuG 向量集合与图共享同一数据库，表必须先由图适配器建好）
            await unified.graph.is_empty()
            await self._load_vectors(unified.vector, corpus, texts, n)
            await self._load_graph(unified.graph, corpus, texts, n)

        self._run(_load_all())

    async def _load_vectors(self, vector_engine, corpus, texts, n):
        """向量 + 原文直注 DocumentChunk_text 集合（不触发 embedding）。

        NeuG 臂内部即 JSONL COPY FROM 批量写入；payload 带 session_id 供回映射。
        """
        for start in range(0, n, VECTOR_BATCH):
            end = min(start + VECTOR_BATCH, n)
            points = [
                {
                    "id": _chunk_id(i),
                    "vector": [float(x) for x in corpus.embeddings[i]],
                    "payload": {
                        "id": _chunk_id(i),
                        "text": texts[i],
                        "session_id": corpus.session_ids[i],
                        "document_id": f"perf-doc-{i}",
                        "chunk_index": 0,
                        "importance_weight": 0.5,
                        "belongs_to_set": [],
                    },
                }
                for i in range(start, end)
            ]
            await vector_engine.upsert_raw_vectors(
                "DocumentChunk_text", points, payload_schema=_payload_schema_cls()
            )
            if (start // VECTOR_BATCH) % 5 == 4 or end == n:
                print(f"[{self.name}] vectors {end}/{n}", flush=True)

    async def _load_graph(self, graph_engine, corpus, texts, n):
        """DocumentChunk 节点 + 规范共现边批量写入图库（COPY FROM）。

        无向边表按 (min, max) 去重去自环：邻域语义不变，写入量减半。
        """
        await self._copy_nodes(graph_engine, corpus, texts, n)
        edges = corpus.graph_edges
        lo = np.minimum(edges[:, 0], edges[:, 1])
        hi = np.maximum(edges[:, 0], edges[:, 1])
        mask = lo != hi
        canonical = np.unique(np.stack([lo[mask], hi[mask]], axis=1), axis=0)
        print(
            f"[{self.name}] graph edges: {edges.shape[0]} raw -> "
            f"{canonical.shape[0]} canonical",
            flush=True,
        )
        await self._copy_edges(graph_engine, canonical)

    async def _copy_nodes(self, graph_engine, corpus, texts, n):
        raise NotImplementedError

    async def _copy_edges(self, graph_engine, canonical_edges):
        raise NotImplementedError

    # ---- 四类查询：均走 cognee 检索路径，返回 session_id ----
    def query_vector(self, qvec, top_k):
        # ChunksRetriever：vector_engine.search("DocumentChunk_text")，
        # 查询向量由 stub 直注（query 文本仅为占位）
        self._embedder.current = qvec

        async def _q():
            from cognee.modules.retrieval.chunks_retriever import ChunksRetriever

            retriever = ChunksRetriever(top_k=top_k)
            return await retriever.get_retrieved_objects("perf vector query")

        return self._payloads_to_session_ids(self._run(_q()))

    def query_fts(self, keywords, top_k):
        # CHUNKS_LEXICAL 工厂：NeuG 臂走原生 bm25，LanceDB 臂走内存 BM25。
        # 实例缓存复用：BM25 臂首次 initialize 从图引擎全量拉语料（暖机成本，
        # 由 runner warm-up 阶段吸收），逐次重建会重复全量加载。
        query = " ".join(keywords)

        async def _q():
            from cognee.modules.search.methods.get_search_type_retriever_instance import (
                get_search_type_retriever_instance,
            )
            from cognee.modules.search.types import SearchType

            if self._lexical is None:
                self._lexical = await get_search_type_retriever_instance(
                    SearchType.CHUNKS_LEXICAL, query, top_k=top_k
                )
            return await self._lexical.get_retrieved_objects(query)

        return self._payloads_to_session_ids(self._run(_q()))

    def query_hybrid(self, qvec, keywords, top_k):
        # HybridRetriever 检索主体：chunk 向量通道 + entity/summary 通道 RRF 融合。
        # 直注模式下 Entity/TextSummary/EdgeType 集合为空，通道 fail-open，
        # 排序完全由 chunk 向量通道驱动（use_truth_weight=False，personalization off）。
        self._embedder.current = qvec

        async def _q():
            from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(chunks_top_k=top_k, entities_top_k=2)
            retrieved = await retriever.get_retrieved_objects(" ".join(keywords))
            return retrieved.get("chunks", [])

        return self._payloads_to_session_ids(self._run(_q()))

    def query_graph(self, seed_session_id, max_nodes):
        # 图遍历：无向 2 跳完整邻域，走引擎 Cypher 直通（CYPHER 检索路同一路径）。
        # 不用 get_neighborhood：它逐跳拼 OR 链 + 全量拉节点属性/诱导边，
        # 高度数种子（邻域 ~1.1万、诱导边 ~43万）实测单查询 10+ 分钟；
        # 变长路径单语句引擎侧毫秒级（实测 0.2s），检索语义不变。
        # 返回完整邻域（⊇ GT，recall_full 口径自洽；契约 §6 已知坑）。
        chunk_id = self._sid2chunk.get(seed_session_id)
        if chunk_id is None:
            return []

        async def _q():
            from cognee.infrastructure.databases.graph import get_graph_engine

            graph_engine = await get_graph_engine()
            return await graph_engine.query(
                f"MATCH (n:Node)-[:EDGE*1..{GRAPH_DEPTH}]-(m:Node) "
                "WHERE n.id = $sid RETURN DISTINCT m.id",
                {"sid": chunk_id},
            )

        sids = [seed_session_id]   # 种子自身属邻域（与 get_neighborhood 语义对齐）
        seen = {seed_session_id}
        for row in self._run(_q()):
            node_id = row[0] if isinstance(row, (list, tuple)) else row
            sid = self._chunk2sid.get(str(node_id))
            if sid and sid not in seen:
                seen.add(sid)
                sids.append(sid)
        return sids

    # ---- 内部工具 ----
    @staticmethod
    def _payload_of(obj):
        payload = getattr(obj, "payload", None)
        return payload if isinstance(payload, dict) else obj

    def _payloads_to_session_ids(self, objects):
        sids, seen = [], set()
        for obj in objects:
            payload = self._payload_of(obj)
            if not isinstance(payload, dict):
                continue
            sid = self._chunk2sid.get(str(payload.get("id", "")))
            if sid and sid not in seen:
                seen.add(sid)
                sids.append(sid)
        return sids


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


class CogneeNeuGPerfAdapter(CogneePerfAdapter):
    """with NeuG：向量 + 图全走 NeuG（同一嵌入式库），COPY 用 JSONL。"""

    name = "cognee-neug"

    def _provider_env(self, work_dir):
        return {
            "VECTOR_DB_PROVIDER": "neug",
            "GRAPH_DATABASE_PROVIDER": "neug",
            "NEUG_DB_PATH": os.path.join(work_dir, "neug_db"),
        }

    def _backend_dirs(self, work_dir):
        return [os.path.join(work_dir, "neug_db")]

    async def _copy_nodes(self, graph_engine, corpus, texts, n):
        # NeuG Node 表 6 列全量；add_nodes 的成员检查（OR 链）对 5 万节点
        # 无收益，空库直写 COPY FROM（批量导入路径）
        from cognee.infrastructure.databases.neug.copy_batch import copy_jsonl_rows

        now = _now_iso()
        execute = graph_engine.connection_manager.execute
        for start in range(0, n, VECTOR_BATCH):
            end = min(start + VECTOR_BATCH, n)
            rows = [
                {
                    "id": _chunk_id(i),
                    "name": corpus.session_ids[i],
                    "type": "DocumentChunk",
                    "created_at": now,
                    "updated_at": now,
                    "properties": json.dumps({"id": _chunk_id(i), "text": texts[i]}),
                }
                for i in range(start, end)
            ]
            await copy_jsonl_rows(execute, "Node", rows)
        print(f"[{self.name}] nodes {n}/{n} (COPY FROM)", flush=True)

    async def _copy_edges(self, graph_engine, canonical_edges):
        from cognee.infrastructure.databases.neug.copy_batch import copy_jsonl_rows

        now = _now_iso()
        execute = graph_engine.connection_manager.execute
        total = canonical_edges.shape[0]
        for start in range(0, total, NEUG_EDGE_BATCH):
            chunk = canonical_edges[start : start + NEUG_EDGE_BATCH]
            # REL 表 JSON 前两键为 from/to 端点主键
            rows = [
                {
                    "from": _chunk_id(int(a)),
                    "to": _chunk_id(int(b)),
                    "relationship_name": "co_occurs",
                    "created_at": now,
                    "updated_at": now,
                    "properties": "{}",
                }
                for a, b in chunk
            ]
            await copy_jsonl_rows(execute, "EDGE", rows, "(from='Node', to='Node')")
            print(
                f"[{self.name}] edges {min(start + NEUG_EDGE_BATCH, total)}/{total}",
                flush=True,
            )


class CogneeBuiltInPerfAdapter(CogneePerfAdapter):
    """without NeuG：cognee 内置后端（LanceDB 向量 + Ladybug/Kuzu 图），
    COPY 用 CSV（Kuzu 方言）。"""

    name = "cognee-lancedb"

    def _provider_env(self, work_dir):
        return {
            "VECTOR_DB_PROVIDER": "lancedb",
            "VECTOR_DB_URL": os.path.join(work_dir, "cognee.lancedb"),
            "GRAPH_DATABASE_PROVIDER": "ladybug",
            "GRAPH_FILE_PATH": os.path.join(work_dir, "graph"),
        }

    def _backend_dirs(self, work_dir):
        return [
            os.path.join(work_dir, "cognee.lancedb"),
            os.path.join(work_dir, "graph"),
        ]

    async def _copy_nodes(self, graph_engine, corpus, texts, n):
        # 5 万节点走 add_nodes 的 UNWIND MERGE 内置分批写
        now = _now_iso()
        for start in range(0, n, VECTOR_BATCH):
            end = min(start + VECTOR_BATCH, n)
            nodes = [
                SimpleNamespace(
                    id=_chunk_id(i),
                    name=corpus.session_ids[i],
                    type="DocumentChunk",
                    created_at=now,
                    updated_at=now,
                    text=texts[i],
                )
                for i in range(start, end)
            ]
            await graph_engine.add_nodes(nodes)
        print(f"[{self.name}] nodes {n}/{n}", flush=True)

    async def _copy_edges(self, graph_engine, canonical_edges):
        # Kuzu CSV COPY FROM：EDGE 表全 10 列（from + to + 4 载荷列 + 4 provenance
        # 列，COPY 不接受缺列，provenance 列置空串）。边值纯整数/固定串，无转义需求。
        now = _now_iso()
        header = [
            "from", "to", "relationship_name", "created_at", "updated_at", "properties",
            "source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs",
        ]
        total = canonical_edges.shape[0]
        shard_no = 0
        for start in range(0, total, LADYBUG_EDGE_SHARD):
            chunk = canonical_edges[start : start + LADYBUG_EDGE_SHARD]
            path = os.path.join(os.environ["GRAPH_FILE_PATH"], f"perf_edges_{shard_no}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(header)
                for a, b in chunk:
                    writer.writerow(
                        [
                            _chunk_id(int(a)), _chunk_id(int(b)), "co_occurs", now, now, "{}",
                            "", "", "", "",
                        ]
                    )
            await graph_engine.query(f"COPY EDGE FROM '{path}' (header=true)")
            os.remove(path)
            shard_no += 1
            print(
                f"[{self.name}] edges {min(start + LADYBUG_EDGE_SHARD, total)}/{total}",
                flush=True,
            )
