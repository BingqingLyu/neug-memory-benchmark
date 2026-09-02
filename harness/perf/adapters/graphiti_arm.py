"""graphiti 性能赛道 adapter：with/without NeuG 两臂（graphiti-neug / graphiti-neo4j）。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕过 LLM 抽取：
  * NEUG 臂：经 COPY FROM CSV 批量写入 graphiti 标准表结构（列集与
    graphiti_core/driver/neug/bulk_import.py 完全一致），一条语句整表导入；
  * Neo4j 臂：无 COPY FROM，用 UNWIND 批量 CREATE（后端差异，非数据差异）。
  全程 0 次 LLM/网络调用：向量直接用数据集预计算值，无 embedding 请求。
- 数据映射：1 session = 1 个 Entity 节点——
  uuid = session_id（行号↔session_id↔主键 三合一）、summary = session 原文
  （FTS 索引字段）、name_embedding = 预计算向量（HNSW/cosine 索引字段）、
  group_id 固定 'perf'（graphiti BFS 分支强制 n.group_id = origin.group_id）。
  规范共现图 = RELATES_TO 边：无向边表按双向各写一条（graphiti 的节点 BFS
  是有向模式 origin-[:RELATES_TO*1..2]->）。边不带 fact/向量（EdgeDoc 镜像
  不参与本赛道任何查询路径）。
- 索引走 graphiti 自身建索引路径：NeuGDriver bootstrap 建
  HNSW(Entity.name_embedding, cosine) + FTS(Entity.name/summary)；
  Neo4j 臂调 build_indices_and_constraints（fulltext node_name_and_summary
  + range 索引），载入后 CALL db.awaitIndexes 等 population。
- query 走 graphiti 检索路径：graphiti_core.search.search.node_search——
  Graphiti.search 的节点检索编排器（方法扇出 + 2×limit 候选超取 + RRF 融合），
  查询 embedding 步骤以数据集预计算向量替代（不经任何模型）。例外：NEUG 臂
  graph 类因全字段序列化成本不可行而短路为 driver 级 BFS（同遍历语义，仅取
  uuid；契约 §4 允许短路并要求记录，见 NOTES.md）；Neo4j 臂走真实
  node_search bfs 路径。
- 四类查询对应：
  vector  = NodeSearchMethod.cosine_similarity 单路；
  fts     = NodeSearchMethod.bm25 单路（NEUG 词项隐式 AND；Neo4j fulltext
            默认 OR，recall 反映其 AND 贴合度——契约 §6 已声明该口径）；
  hybrid  = cosine + bm25 双路 + NodeReranker.rrf 融合；
  graph   = driver 级 BFS（RELATES_TO*1..2，同 node_search bfs 分支语义），
            种子节点 uuid 即 seed_session_id，返回完整邻域
            （⊇ 按 hop 截断的 GT，契约 §6）。
- sim_min_score 置 0：graphiti 默认 0.6 会按相似度阈值丢候选，而暴力 GT 是
  无条件 top-k，阈值会造成不可比截断（与 mem0 臂 THRESHOLD=0 同一理由）。

实现说明详见 results/perf/<arm>/NOTES.md。
"""
import asyncio
import csv
import os
import shutil
import time

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")  # import graphiti 前

import numpy as np  # noqa: E402

from graphiti_core.search.search_config import (  # noqa: E402
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
)
from graphiti_core.search.search_filters import SearchFilters  # noqa: E402
from graphiti_core.search.search import node_search  # noqa: E402

from ..base import ALL_CLASSES, PerfAdapter  # noqa: E402

GROUP_ID = "perf"          # 单语料单组；两臂一致
SIM_MIN_SCORE = 0.0        # GT 为无条件 top-k，不按阈值丢候选（默认 0.6）
BFS_MAX_DEPTH = 2          # graph_multihop GT = 2 跳邻域
NODE_BATCH = 1000          # Neo4j UNWIND 节点批（1024 维向量，批太大超载荷上限）
EDGE_BATCH = 20000         # Neo4j UNWIND 边批
# NEUG COPY 列集：与 graphiti_core/driver/neug/bulk_import.py 保持一致
# （COPY 要求 CSV 列集与表结构完全匹配）
NEUG_ENTITY_COLUMNS = [
    "uuid", "name", "name_embedding", "group_id", "summary",
    "labels", "attributes", "created_at",
]
NEUG_RELATES_TO_COLUMNS = [
    "from", "to", "uuid", "name", "fact", "fact_embedding", "episodes",
    "group_id", "attributes", "created_at", "expired_at", "valid_at",
    "invalid_at", "reference_time",
]
CREATED_AT = "2026-01-01T00:00:00.000000+00:00"


def _vector_literal(vec) -> str:
    """FLOAT[N] 字面量：每个元素必须带小数点（NeuG 不做隐式类型转换）。"""
    parts = []
    for x in vec:
        s = repr(float(x))
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        parts.append(s)
    return "[" + ", ".join(parts) + "]"


class GraphitiPerfAdapter(PerfAdapter):
    #: graphiti 四类查询全支持（契约 §3.5 事实表）
    supported_classes = frozenset(ALL_CLASSES)

    def __init__(self):
        self._loop = None
        self._driver = None
        self._filters = SearchFilters()
        # graph 类邻域结果缓存：warmup 与计时重放同一种子，病态种子
        # （25k 邻域、276MB summary 序列化）只计费一次
        self._graph_cache = {}

    # ---- 后端差异点（子类实现）----
    def _open_driver(self, work_dir):
        raise NotImplementedError

    def _clear_backend(self, work_dir):
        raise NotImplementedError

    async def _load_nodes(self, corpus):
        raise NotImplementedError

    async def _load_edges(self, corpus):
        raise NotImplementedError

    # ---- 生命周期 ----
    def setup(self, work_dir):
        self._loop = asyncio.new_event_loop()
        self._clear_backend(work_dir)
        self._driver = self._run(self._open_driver(work_dir))

    def teardown(self):
        # NeuG 连接必须正常 close（checkpoint 纪律）
        if self._driver is not None:
            try:
                self._run(self._driver.close())
            except Exception:  # noqa: BLE001 - teardown 不抛错
                pass
            self._driver = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # ---- load：预计算语料存储层直注，不经 LLM ----
    def load(self, corpus):
        async def _load():
            await self._load_nodes(corpus)
            await self._load_edges(corpus)
            await self._post_load()

        self._run(_load())

    async def _post_load(self):
        """载入完成后的后端收尾（默认无操作）。"""

    # ---- 四类查询：均走 graphiti node_search 检索编排 ----
    def query_vector(self, qvec, top_k):
        nodes, _ = self._node_search(
            query="",
            qvec=qvec,
            methods=[NodeSearchMethod.cosine_similarity],
            limit=top_k,
        )
        return [n.uuid for n in nodes]

    def query_fts(self, keywords, top_k):
        nodes, _ = self._node_search(
            query=" ".join(keywords),
            qvec=None,
            methods=[NodeSearchMethod.bm25],
            limit=top_k,
        )
        return [n.uuid for n in nodes]

    def query_hybrid(self, qvec, keywords, top_k):
        # graphiti hybrid：向量 + BM25 双路候选，RRF 融合（系统自身排序逻辑）
        nodes, _ = self._node_search(
            query=" ".join(keywords),
            qvec=qvec,
            methods=[NodeSearchMethod.cosine_similarity, NodeSearchMethod.bm25],
            limit=top_k,
        )
        return [n.uuid for n in nodes]

    def query_graph(self, seed_session_id, max_nodes):
        # 种子节点 uuid 即 session_id。契约 §6：真实臂返回完整邻域即 ⊇ GT
        # （GT 已按 hop 独立截断），不得用单一全局 top-N 截断——本语料 2 跳
        # 邻域可达 ~25k 节点，截断会丢 hop2 GT。max_nodes 形参此处不适用。
        #
        cached = self._graph_cache.get(seed_session_id)
        if cached is not None:
            return cached
        nodes, _ = self._node_search(
            query="",
            qvec=None,
            methods=[NodeSearchMethod.bfs],
            limit=10_000_000,
            bfs_origin_node_uuids=[seed_session_id],
        )
        result = [n.uuid for n in nodes]
        self._graph_cache[seed_session_id] = result
        return result

    # ---- 内部工具 ----
    def _node_search(self, query, qvec, methods, limit, bfs_origin_node_uuids=None):
        config = NodeSearchConfig(
            search_methods=methods,
            reranker=NodeReranker.rrf,
            sim_min_score=SIM_MIN_SCORE,
            bfs_max_depth=BFS_MAX_DEPTH,
        )
        return self._run(
            node_search(
                driver=self._driver,
                cross_encoder=None,       # rrf 重排不调 cross encoder
                query=query,
                query_vector=list(qvec) if qvec is not None else [0.0] * 1024,
                group_ids=None,
                config=config,
                search_filter=self._filters,
                bfs_origin_node_uuids=bfs_origin_node_uuids,
                limit=limit,
                reranker_min_score=0.0,
            )
        )

    def _run(self, coro):
        assert self._loop is not None
        return self._loop.run_until_complete(coro)


class GraphitiNeuGPerfAdapter(GraphitiPerfAdapter):
    name = "graphiti-neug"

    def query_graph(self, seed_session_id, max_nodes):
        # NEUG 专属短路（契约 §4"短路该步并记录"条款，详见 NOTES.md）：
        # node_search 的 bfs 分支会序列化节点全部字段（每节点 ~10KB summary，
        # 最大邻域 25k 节点单次 ~7 分钟；50 查询合计 ~4 小时），而本类查询
        # 的口径只需要 uuid 集合。故此处直接走 driver 级 BFS，遍历语义与
        # node_search bfs 分支完全一致（RELATES_TO*1..2 + 同组约束 +
        # 引擎侧去重），仅 RETURN uuid。Neo4j 臂无此瓶颈，走真实
        # node_search 路径。
        cached = self._graph_cache.get(seed_session_id)
        if cached is not None:
            return cached
        rows, _, _ = self._run(self._driver.execute_query(
            "MATCH (origin:Entity)-[:RELATES_TO*1.." + str(BFS_MAX_DEPTH) + "]->(n:Entity) "
            "WHERE origin.uuid IN $origins AND n.group_id = origin.group_id "
            "RETURN DISTINCT n.uuid AS uuid",
            origins=[seed_session_id],
        ))
        result = [r["uuid"] for r in rows]
        self._graph_cache[seed_session_id] = result
        return result

    def _db_dir(self, work_dir):
        return os.path.join(work_dir, "graphiti.db")

    def _clear_backend(self, work_dir):
        shutil.rmtree(self._db_dir(work_dir), ignore_errors=True)

    async def _open_driver(self, work_dir):
        from graphiti_core.driver.neug_driver import NeuGDriver

        # bootstrap 同步建好表 + HNSW/FTS 索引（探针已验证：带索引 COPY 后
        # 向量/全文/变长路径检索均可用，见 probes/probe_graphiti_perf_copy.py）
        return NeuGDriver(db_path=self._db_dir(work_dir), embedding_dim=1024)

    async def _load_nodes(self, corpus):
        n = len(corpus.session_ids)
        csv_dir = os.path.join(os.path.dirname(self._driver._database), "csv")
        os.makedirs(csv_dir, exist_ok=True)
        path = os.path.join(csv_dir, "entity.csv")
        t0 = time.time()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(NEUG_ENTITY_COLUMNS)
            for i, sid in enumerate(corpus.session_ids):
                # corpus 向量已 L2 归一（与 driver 写前预归一约定等价）
                w.writerow([
                    sid, sid,
                    _vector_literal(corpus.embeddings[i].tolist()),
                    GROUP_ID, corpus.texts[i], "[]", "", CREATED_AT,
                ])
                if (i + 1) % 10000 == 0:
                    print(f"[{self.name}] entity csv {i + 1}/{n} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{self.name}] entity csv written in {time.time() - t0:.0f}s, "
              f"COPY ...", flush=True)
        await self._driver.execute_query(
            f'COPY Entity FROM "{path}" (HEADER true, DELIMITER ",")'
        )

    async def _load_edges(self, corpus):
        ids = np.asarray(corpus.session_ids)
        src = ids[corpus.graph_edges[:, 0]]
        dst = ids[corpus.graph_edges[:, 1]]
        csv_dir = os.path.join(os.path.dirname(self._driver._database), "csv")
        path = os.path.join(csv_dir, "relates_to.csv")
        t0 = time.time()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(NEUG_RELATES_TO_COLUMNS)
            # 无向边按双向各写一条：graphiti 节点 BFS 走有向模式
            for a, b in zip(src, dst):
                a, b = str(a), str(b)
                for u, v in ((a, b), (b, a)):
                    w.writerow([
                        u, v, f"{u}|{v}", "", "", "", "[]", GROUP_ID, "",
                        CREATED_AT, "", "", "", "",
                    ])
            print(f"[{self.name}] edge csv written ({len(src) * 2} rows) "
                  f"in {time.time() - t0:.0f}s, COPY ...", flush=True)
        await self._driver.execute_query(
            f'COPY RELATES_TO FROM "{path}" (HEADER true, DELIMITER ",")'
        )


class GraphitiNeo4jPerfAdapter(GraphitiPerfAdapter):
    name = "graphiti-neo4j"

    def _clear_backend(self, work_dir):
        pass  # 清图在打开驱动后异步执行（见 _open_driver）

    async def _open_driver(self, work_dir):
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        driver = Neo4jDriver(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "testpass"),
        )
        # 性能赛道要求每次 load 从空库开始：清掉共享服务端图库
        # （原内容为 LoCoMo 赛道 graphiti-neo4j 臂的图，可经 LLM 缓存再生）。
        # 分批删除：单事务全删会在 5.6M 边上撑爆事务内存池（8.4GB 上限）；
        # DETACH DELETE 连 hub 节点（度 ~25k）时单批边数仍会爆，故先按批
        # 删边、再删孤立点。
        while True:
            rows, _, _ = await driver.execute_query(
                "MATCH ()-[r]->() WITH r LIMIT 100000 DELETE r RETURN count(*) AS c"
            )
            if not rows or int(rows[0]["c"]) == 0:
                break
        while True:
            rows, _, _ = await driver.execute_query(
                "MATCH (n) WITH n LIMIT 50000 DELETE n RETURN count(*) AS c"
            )
            if not rows or int(rows[0]["c"]) == 0:
                break
        await driver.build_indices_and_constraints()
        return driver

    async def _post_load(self):
        # fulltext 索引 population 是异步的，查询前必须等其 ONLINE
        await self._driver.execute_query("CALL db.awaitIndexes(600)")

    async def _load_nodes(self, corpus):
        n = len(corpus.session_ids)
        t0 = time.time()
        for start in range(0, n, NODE_BATCH):
            end = min(start + NODE_BATCH, n)
            rows = [
                {
                    "uuid": corpus.session_ids[i],
                    "name": corpus.session_ids[i],
                    "summary": corpus.texts[i],
                    "embedding": [float(x) for x in corpus.embeddings[i]],
                }
                for i in range(start, end)
            ]
            await self._driver.execute_query(
                """
                UNWIND $rows AS r
                CREATE (n:Entity {
                    uuid: r.uuid, name: r.name, summary: r.summary,
                    group_id: $group_id, created_at: $created_at,
                    labels: $labels
                })
                WITH n, r
                CALL db.create.setNodeVectorProperty(n, 'name_embedding', r.embedding)
                """,
                rows=rows, group_id=GROUP_ID, created_at=CREATED_AT,
                labels=[],
            )
            if (start // NODE_BATCH) % 10 == 9 or end == n:
                print(f"[{self.name}] nodes {end}/{n} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    async def _load_edges(self, corpus):
        ids = np.asarray(corpus.session_ids)
        src = ids[corpus.graph_edges[:, 0]]
        dst = ids[corpus.graph_edges[:, 1]]
        m = len(src)
        t0 = time.time()
        for start in range(0, m, EDGE_BATCH // 2):
            end = min(start + EDGE_BATCH // 2, m)
            rows = []
            for i in range(start, end):
                a, b = str(src[i]), str(dst[i])
                rows.append({"f": a, "t": b, "u": f"{a}|{b}"})
                rows.append({"f": b, "t": a, "u": f"{b}|{a}"})
            await self._driver.execute_query(
                """
                UNWIND $rows AS r
                MATCH (a:Entity {uuid: r.f})
                MATCH (b:Entity {uuid: r.t})
                CREATE (a)-[:RELATES_TO {uuid: r.u, group_id: $group_id}]->(b)
                """,
                rows=rows, group_id=GROUP_ID,
            )
            if (end - start == EDGE_BATCH // 2 and (start // (EDGE_BATCH // 2)) % 20 == 19) or end == m:
                print(f"[{self.name}] edges {end * 2}/{m * 2} "
                      f"({time.time() - t0:.0f}s)", flush=True)
