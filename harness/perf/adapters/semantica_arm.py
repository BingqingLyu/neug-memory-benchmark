"""semantica 性能赛道 adapter：semantica-neug / semantica-native 双臂。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕 LLM 抽取：预计算向量/文本/规范共现图原样载入，全程 0 次
  LLM/embedding 调用（约束 1/2）。
- NEUG 臂：COPY FROM JSONL 批量冲刷（index-first：HNSW+FTS 先建后 COPY，
  第一步 P13 已验证路径）。重复 PK 的去重由 backend 的 copy_nodes/copy_edges
  在写 JSONL 前完成（semantica/graph_store/neug/store.py，引擎侧对重复 PK 静默
  跳过），adapter 不做第二遍。
- NATIVE 臂：Semantica 存储层直写——内存 ContextGraph（add_nodes/add_edges）
  + 原生 FAISS vector store（flat + inner_product，归一向量上等价余弦）。
- 四类查询走系统检索路径（约束 3）：
  vector = neug 原生 VectorStore(backend=neug).search_vectors（纯 HNSW，不传
           _neug_raw_query）/ native FAISS search_vectors
  fts    = neug bm25（决策 2a：门面无「纯 bm25」入口，孤立微基准直调 provider
           NeugFTS）；native 无全文路（原生栈源码事实），标 N/A（B3）
  hybrid = neug 向量+bm25 由后端 §1.4 内聚融合（search_vectors 传
           _neug_raw_query），再走 Semantica 原生 _rank_and_merge；native 纯向量
           （无词法成分，是能力差不是配置差）。native 侧分值必须取 FAISS 的
           distance（=余弦）而非 score，见 _faiss_ip_score
  graph  = neug get_neighbors varlen 服务端遍历（无向 *1..2）/
           native ContextGraph.get_neighbors(hops=2)；均返回完整邻域
           （⊇ 按 hop 截断的 GT，契约 §6，勿全局 top-N 截断）
- 行号↔session_id↔后端主键三合一：节点 id 直接 = session_id（约束 4）。
- teardown 正常 close（约束 6，NeuG checkpoint 纪律）。

实现说明详见 results/perf/<arm>/NOTES.md。
"""
import os
import shutil
import time

import numpy as np

# semantica（含 graph_store/neug provider）经 benchmark venv 的 editable 安装提供，
# 与 graphiti/mem0/cognee 三系统同惯例（裸导入，无 sys.path 注入）。

from ..base import (
    ALL_CLASSES,
    GRAPH_MULTIHOP,
    HYBRID,
    PerfAdapter,
    VECTOR_TOPK,
)

# NeuG 每次 COPY FROM 有一笔 ∝ 表内已有行数 的固定开销：每次 seal 都要把该表
# 在 checkpoint 里整表重写（引擎自己会 warn "Incremental checkpoint rewrites
# vertex table 'Entity' ... consider batching COPY statements"）。所以分片是净
# 亏损，不是防 OOM 的手段 —— 实测 51661 个 1024 维节点：
#   11 批(5000)  = 182.4s, peak RSS 12.9GB
#    3 批(20000) = 147.0s, peak RSS 13.3GB
#    1 批(51661) = 136.4s, peak RSS  9.2GB   ← 更快且更省内存
# 分批抬高 RSS 是因为每批的 checkpoint 副本叠加。真正的上限在 Python 侧：rows
# 把 1024 维向量摊成 float 列表（≈32KB/行），再加 _copy_jsonl 的文本副本，
# 全量单批约 2.7GB。故取一个 > 当前语料、又不至于撑爆 Python 的值。
NODE_BATCH = 60_000          # > 语料节点数（51661）→ 单次 COPY
NEUG_EDGE_COPY_BATCH = 5_000_000   # > 语料边数（2799244）→ 单次 COPY
EDGE_BATCH = 200_000       # native ContextGraph.add_edges 分片（无此惩罚，按内存分片）
FAISS_ADD_BATCH = 20_000   # native FAISS add_vectors 批
EDGE_TYPE = "COOCCURS"     # 规范共现边类型（两臂一致）
GRAPH_DEPTH = 2            # graph_multihop 的 BFS 跳数（三臂共用同一值）


def _faiss_ip_score(hit: dict) -> float:
    """把 FAISS inner_product 命中换算成与 neug 臂同标度的余弦分。

    FAISSSearch.search_similar 的 "score" 是 1/(1+max(0,dist))，而
    inner_product 下 FAISS 返回的 dist 本身就是余弦（语料与 query_vectors.npy
    均已 L2 归一，实测范数 1.0±1e-7）—— 该变换是余弦的严格递减函数（实测
    cos=1.0 -> 0.500、cos=0.0 -> 1.000）。直接当分值喂进 _rank_and_merge，
    min-max 归一化后整路排序被反转，返回的是"最不相关"的前 top_k。
    perf 的 recall_at_k 取 set(returned[:k])，对顺序不敏感，所以这个 bug 在
    指标上看不出来，但违反 base.py「均返回按相关度排序」的契约。
    neug 臂 NeugVectorStore.search 给的是 1 - cos_distance/2 = (1+cos)/2，
    这里取同一公式，两臂分值可直接互比。

    只适用于 FAISS 命中：两臂的 hit dict 都有 "distance" 键但语义相反
    （neug = 余弦距离，0 为相同；FAISS inner_product = 余弦相似度，1 为
    相同），把本函数用在 neug 命中上会得到反向分。neug 的 "score" 已经
    是正确的，直接用。
    """
    cos = float(hit["distance"])
    return max(0.0, min(1.0, (1.0 + cos) / 2.0))


class SemanticaNeuGPerfAdapter(PerfAdapter):
    """with NeuG：一个嵌入式库承载向量 + 全文 + 图三类检索。"""
    name = "semantica-neug"
    supported_classes = frozenset(ALL_CLASSES)

    def __init__(self):
        self._gs = None
        self._vs = None
        self._fts = None
        self._retriever = None

    def setup(self, work_dir):
        from semantica.graph_store.graph_store import GraphStore
        from semantica.graph_store.neug import NeugFTS
        from semantica.vector_store.vector_store import VectorStore
        from semantica.context.context_retriever import ContextRetriever

        db_dir = os.path.join(work_dir, "semantica.db")
        shutil.rmtree(db_dir, ignore_errors=True)
        # index-first：gs.connect() 作为 §1.2 注册表 owner 先建 HNSW + FTS 索引，
        # 再 COPY 冲刷；vs 随后 join 同一 engine（refcount==2），不重复建索引。
        self._gs = GraphStore(backend="neug", db_path=db_dir, vector_dim=1024)
        self._gs.connect()
        # Option A：向量走原生 VectorStore(backend=neug) 门面（§1.3 派发 → §1.4
        # 后端内聚融合），与 gs 经 §1.2 注册表共享单 engine，不再手工拼 store。
        self._vs = VectorStore(backend="neug",
                               config={"db_path": db_dir, "dimension": 1024})
        # 决策 2a：query_fts 是孤立 bm25 微基准，门面无「纯 bm25」入口（全文在
        # Semantica 里是 neug 独有能力），故仍直调 provider NeugFTS，构造自 gs 的
        # 后端（与 vs 同一 engine）。
        self._fts = NeugFTS(self._gs._store_backend)
        self._retriever = ContextRetriever(hybrid_alpha=0.5)

    def load(self, corpus):
        backend = self._gs._store_backend
        n = len(corpus.session_ids)
        t0 = time.time()
        for start in range(0, n, NODE_BATCH):
            end = min(start + NODE_BATCH, n)
            rows = [
                {"id": corpus.session_ids[i], "etype": "session",
                 "content": corpus.texts[i],
                 "vec": corpus.embeddings[i].tolist(), "extra": ""}
                for i in range(start, end)
            ]
            backend.copy_nodes(rows)
            print(f"[{self.name}] nodes {end}/{n} ({time.time() - t0:.0f}s)",
                  flush=True)

        ids = np.asarray(corpus.session_ids)
        edges = corpus.graph_edges
        m = len(edges)
        for start in range(0, m, NEUG_EDGE_COPY_BATCH):
            end = min(start + NEUG_EDGE_COPY_BATCH, m)
            # 单次 COPY 意味着这里一次性物化 2.8M 个 dict（≈600MB），随后
            # _copy_jsonl 再摊成 ~250MB 文本，两份同时在峰值上。合并成单次是
            # 值得的（分批 3674s vs 单次 2.6s），但内存账要认。
            rows = [
                {"from_id": str(ids[a]), "to_id": str(ids[b]),
                 "edge_type": EDGE_TYPE, "weight": 1.0}
                for a, b in edges[start:end]
            ]
            backend.copy_edges(rows)
            print(f"[{self.name}] edges {end}/{m} ({time.time() - t0:.0f}s)",
                  flush=True)

    def query_vector(self, qvec, top_k):
        # 纯向量路：走原生门面，不传 _neug_raw_query（后端只做 HNSW，不融合 bm25）。
        return [h["id"] for h in self._vs.search_vectors(list(qvec), k=top_k)]

    def query_fts(self, keywords, top_k):
        return [h["id"] for h in self._fts.search(" ".join(keywords),
                                                  limit=top_k)]

    def query_hybrid(self, qvec, keywords, top_k):
        from semantica.context.context_retriever import RetrievedContext

        # Option A：向量 + bm25 由后端 §1.4 内聚融合——单次 search_vectors，传
        # _neug_raw_query 触发 bm25 路，命中自带融合分值与 metadata['retrieval_routes']
        # 溯源（['vector'] / ['bm25'] / 两者=共识）。不再手工拼「向量路 + fts 路」。
        results = []
        for h in self._vs.search_vectors(list(qvec), k=top_k,
                                         _neug_raw_query=" ".join(keywords)):
            results.append(RetrievedContext(
                # content 直接下标：load() 给每个节点都写了 corpus.texts[i]，后端
                # 融合命中（向量 _metadata_from_row / bm25-only fts hit）都带 content。
                # 这里比 locomo 臂更不能容它默默变空：content 是 _rank_and_merge 的
                # content[:100] 去重键，全路空串会撞成同一个键、把整条路去重成 1 条，
                # 而 hybrid recall 只会看上去偏低，没任何异常信号。
                content=h["metadata"]["content"], score=h["score"],
                source=f"vector:{h['id']}",
                metadata={"node_id": h["id"],
                          "retrieval_routes": h["metadata"].get("retrieval_routes")}))
        # 与 native 臂对称：同样以 _rank_and_merge 收尾（此处单 vector 池，min-max
        # 保序、content 恒为真实 session 文本不会撞空键，故只归一不改变排序）。neug
        # 与 native 的 hybrid 差异是能力差（neug 池含 bm25 融合命中、native 纯向量），
        # 不是配置差，NOTES.md 里写明。
        merged = self._retriever._rank_and_merge(results, " ".join(keywords))
        return [r.metadata["node_id"] for r in merged[:top_k]]

    def query_graph(self, seed_session_id, max_nodes):
        # varlen 服务端遍历（无向 *1..2），返回完整邻域 ⊇ 按 hop 截断的 GT；
        # max_nodes 形参不适用（全局 top-N 截断会丢 hop2，契约 §6）。
        #
        # driver 级窄投影短路（契约 §4「短路该步并记录」），**不走门面**
        # `GraphStore.get_neighbors`。两个理由：
        #   1) 成本：门面按 semantica 的公开 API 契约返回完整节点记录
        #      （`neug/store.py:1092-1093` 投影 `t.id, t.etype, t.content,
        #      t.extra`），而本赛道 content 平均 10284 字符 ⇒ 单题 ~9538 行
        #      × ~10 KB ≈ 98 MB 纯投影开销；而本类查询口径只需要 id 集合
        #      （GT 与 recall 都定义在 id 集合上）。
        #   2) 对称性（更要紧）：本组的 neo4j 臂**本来就短路**——semantica 自己的
        #      neo4j 后端 `neo4j_store.py:895` 返回 `id(neighbor), neighbor,
        #      labels(neighbor)`（整个节点 + labels，比 neug 后端还宽），而 harness
        #      的 neo4j 臂绕过门面手写了只投影 id 的 Cypher。即改动前是「neo4j 臂
        #      短路、neug 臂不短路」，与 graphiti 组修复前正好镜像。现两臂都只
        #      投影 id，且 Cypher 逐字同形。
        # 遍历语义与门面逐条等价：同 MATCH 模式（无向 `*1..2`，表名/列名取自
        # semantica 的 schema 映射、不硬编码）、同 `WITH DISTINCT t` 引擎侧按节点
        # 身份去重、同「排除种子自身」（与门面一样在 Python 侧做，见下）；
        # 仅少投影三个字段。
        # ⚠️ 排除种子 **不能下推到 Cypher**：实测 `WHERE t.id <> $start` 会让 p50
        # 从 2.962ms 涨到 4.591ms（**1.55×**），引擎日志同时报
        # `path.cc:715 Currently only support path expand without vertex predicate`
        # ——NeuG 不支持变长展开上带顶点谓词，会退到退化计划。故按门面的原做法
        # 在 Python 侧过滤（实测该过滤只值 ~0.05ms，可忽）。
        # 走 `_run` 而非 `execute_query`：与门面 `get_neighbors` 内部同一条执行路径
        # （同 MODE_READ、无 progress_tracker 逐次开销），使测得的差值纯粹来自
        # 投影宽度。已实测 50 个真实种子的 id 集合与门面逐一一致
        # （**0/50 不匹配**，且长度也逐题相等）；宽投影 p50 41.108ms →
        # 窄投影 5.207ms（7.89×），每行成本 4.24 → 0.56 µs。
        from semantica.graph_store.neug import schema as neug_schema

        start = str(seed_session_id)
        rows = self._gs._store_backend._run(
            f"MATCH (s:{neug_schema.NODE_TABLE} {{id: $start}})"
            f"-[r:{neug_schema.REL_TABLE}*1..{GRAPH_DEPTH}]-(t:{neug_schema.NODE_TABLE}) "
            f"WITH DISTINCT t RETURN t.{neug_schema.PK_COL}",
            {"start": start},
        )
        return [r[0] for r in rows if r[0] != start]

    def teardown(self):
        # gs 与 vs 共享单 engine（§1.2 注册表 refcount==2），两个都 close 才把
        # refcount 归 0、真正释放引擎一次。fts 是 gs 后端的轻量包装，无需单独 close。
        if self._vs is not None:
            self._vs.close()
            self._vs = None
        if self._gs is not None:
            self._gs.close()
            self._gs = None


class SemanticaNativePerfAdapter(PerfAdapter):
    """without NeuG 对照：内存 ContextGraph + 原生 FAISS。fts 标 N/A。"""
    name = "semantica-native"
    supported_classes = frozenset({VECTOR_TOPK, HYBRID, GRAPH_MULTIHOP})

    def __init__(self):
        self._cg = None
        self._vs = None
        self._retriever = None

    def setup(self, work_dir):
        from semantica.context.context_graph import ContextGraph
        from semantica.context.context_retriever import ContextRetriever
        from semantica.vector_store.vector_store import VectorStore

        self._cg = ContextGraph()
        self._vs = VectorStore(backend="faiss", config={"dimension": 1024})
        self._retriever = ContextRetriever(hybrid_alpha=0.5)

    def load(self, corpus):
        t0 = time.time()
        n = len(corpus.session_ids)
        backend_store = self._vs._backend_store
        # inner_product + 已归一向量 == 余弦（精确暴力索引，与暴力 GT 同序）
        backend_store.create_index(index_type="flat", metric="inner_product")
        vecs = np.asarray(corpus.embeddings, dtype=np.float32)
        for start in range(0, n, FAISS_ADD_BATCH):
            end = min(start + FAISS_ADD_BATCH, n)
            backend_store.add_vectors(
                vecs[start:end], ids=corpus.session_ids[start:end],
                metadata=[{"node_id": sid}
                          for sid in corpus.session_ids[start:end]])
            print(f"[{self.name}] vectors {end}/{n} ({time.time() - t0:.0f}s)",
                  flush=True)

        # 无向边表按双向各写一条：ContextGraph 邻接是出边单向存储，
        # get_neighbors 走出边，双向写入才是无向遍历（与 graphiti 臂同法）。
        ids = np.asarray(corpus.session_ids)
        edges = corpus.graph_edges
        m = len(edges)
        for start in range(0, m, EDGE_BATCH):
            end = min(start + EDGE_BATCH, m)
            rows = []
            for a, b in edges[start:end]:
                sa, sb = str(ids[a]), str(ids[b])
                rows.append({"source_id": sa, "target_id": sb,
                             "type": EDGE_TYPE, "weight": 1.0})
                rows.append({"source_id": sb, "target_id": sa,
                             "type": EDGE_TYPE, "weight": 1.0})
            self._cg.add_edges(rows)
            print(f"[{self.name}] edges {end}/{m} ({time.time() - t0:.0f}s)",
                  flush=True)
        # 节点（边写入时端点已自动建为占位节点，这里补 content/type）
        self._cg.add_nodes([
            {"id": sid, "type": "session", "content": corpus.texts[i]}
            for i, sid in enumerate(corpus.session_ids)
        ])
        print(f"[{self.name}] loaded in {time.time() - t0:.0f}s", flush=True)

    def query_vector(self, qvec, top_k):
        # FAISS 返回序即相关度降序（IP 度量下 index.search 内部排好），按原序
        # 取 id 就是契约要的排序；不要按 r["score"] 重排，那个字段是倒的。
        res = self._vs.search_vectors(np.asarray(qvec, dtype=np.float32),
                                      k=top_k)
        return [r["id"] for r in res]

    # query_fts：原生栈无全文路（HANDOFF §1 源码事实），保持
    # NotImplementedError 且已从 supported_classes 去掉，runner 标 N/A。

    def query_hybrid(self, qvec, keywords, top_k):
        # native hybrid = 纯向量路（无词法成分）：与 neug 三路融合的差异是
        # 能力差（无 bm25 路），不是配置差，NOTES.md 里写明。
        from semantica.context.context_retriever import RetrievedContext

        results = []
        for r in self._vs.search_vectors(np.asarray(qvec, dtype=np.float32),
                                         k=top_k):
            # content 只被 _rank_and_merge 用来按 content[:100] 去重，取图里
            # 那一份即可（add_nodes 无条件覆盖 add_edges 建的占位节点，故这里
            # 一定是真实文本）；不再另存一份 id->text 字典。
            results.append(RetrievedContext(
                content=self._cg.nodes[r["id"]].content,
                score=_faiss_ip_score(r),
                source=f"vector:{r['id']}", metadata={"node_id": r["id"]}))
        merged = self._retriever._rank_and_merge(results, " ".join(keywords))
        return [r.metadata["node_id"] for r in merged[:top_k]]

    def query_graph(self, seed_session_id, max_nodes):
        # hops=2 BFS 出边（双向写入 == 无向），返回完整邻域 ⊇ GT。
        return [n["id"] for n in self._cg.get_neighbors(seed_session_id,
                                                        hops=GRAPH_DEPTH)]

    def teardown(self):
        """内存栈无需 close；空实现只为与接口对齐，语料与图随 runner 释放。"""


# ---- semantica-neo4j perf 臂（server 对照）----
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpass")
# 与 locomo 臂用不同 marker：两臂可能同 session 先后跑，各自 marker 互不清对方。
NEO4J_MARK = "SemanticaPerf"
NEO4J_NODE_BATCH = 1000      # 节点 UNWIND 批（同 graphiti perf-neo4j）
NEO4J_EDGE_BATCH = 10000     # 边 UNWIND 批


class SemanticaNeo4jPerfAdapter(PerfAdapter):
    """server 对照：原生 FAISS 向量 + 真实 neo4j 图服务端。fts 标 N/A（同 native）。

    能力面 = native（向量 + 图两路，无 bm25），唯一变量是图后端换成真实 neo4j
    server（bolt RPC）。四类里 vector/fts/hybrid 与 native 逐位同（都走内存 FAISS，
    neo4j 不参与），只有 graph_multihop 真正测 neo4j 服务端 BFS —— 这正是
    embedded(neug) vs server(neo4j) 的架构对比点（§2.3 口径标注：延时差异归因于
    架构而非引擎快慢）。

    ingest 走裸 UNWIND（不经 facade add_edges）：facade create_relationship 每边
    一次 `MATCH (a),(b) WHERE id(a)..id(b).. CREATE` 往返，2.8M 边不可行。与 neug
    臂的 copy_nodes/copy_edges、graphiti-neo4j 臂的 UNWIND 同属"后端专有批量
    ingest 路"（retrieval 侧仍尽量走 facade/裸 Cypher，见 query_graph）。节点建
    :SemanticaPerf{id,content} + id 属性索引，边**单写**（query_graph 用无向
    varlen）。向量不进 neo4j（FAISS 承载，与 native 臂同）。
    """
    name = "semantica-neo4j"
    supported_classes = frozenset({VECTOR_TOPK, HYBRID, GRAPH_MULTIHOP})

    def __init__(self):
        self._gs = None
        self._vs = None
        self._retriever = None

    def setup(self, work_dir):
        from semantica.graph_store.graph_store import GraphStore
        from semantica.vector_store.vector_store import VectorStore
        from semantica.context.context_retriever import ContextRetriever

        # work_dir 不用：neo4j 是服务端、FAISS 在内存，无本地库目录（同 native）。
        self._vs = VectorStore(backend="faiss", config={"dimension": 1024})
        self._retriever = ContextRetriever(hybrid_alpha=0.5)
        self._gs = GraphStore(backend="neo4j", uri=NEO4J_URI, user=NEO4J_USER,
                              password=NEO4J_PASSWORD)
        self._gs.connect()
        be = self._gs._store_backend
        # fresh start：分批清掉本臂 marker（scope 到 :SemanticaPerf，不碰 graphiti
        # 或 locomo-neo4j 臂的 :SemanticaBench）。
        self._clear(be)
        # id 属性索引：边 UNWIND 的 MATCH (n:SemanticaPerf {id}) 与 query_graph 的
        # 种子匹配都靠它；缺了则 2.8M 边 × 全表扫描不可行。
        be.execute_query(
            f"CREATE INDEX semantica_perf_id IF NOT EXISTS "
            f"FOR (n:{NEO4J_MARK}) ON (n.id)")
        be.execute_query("CALL db.awaitIndexes(600)")

    @staticmethod
    def _clear(be):
        # 先分批删边、再删点：单事务全删会在 2.8M 边上撑爆事务内存池（与 graphiti
        # perf-neo4j 臂同一手法，但 scope 到本臂 marker，不误删其它臂/系统的数据）。
        while True:
            recs = be.execute_query(
                f"MATCH (n:{NEO4J_MARK})-[e]-() WITH e LIMIT 100000 DELETE e "
                "RETURN count(*) AS c")["records"]
            if not recs or int(recs[0]["c"]) == 0:
                break
        while True:
            recs = be.execute_query(
                f"MATCH (n:{NEO4J_MARK}) WITH n LIMIT 50000 DELETE n "
                "RETURN count(*) AS c")["records"]
            if not recs or int(recs[0]["c"]) == 0:
                break

    def load(self, corpus):
        t0 = time.time()
        n = len(corpus.session_ids)

        # 向量：FAISS（与 native 臂同口径：flat + inner_product，归一向量上等价
        # 余弦）。metadata 额外带 content —— native 臂的 hybrid 从 self._cg.nodes
        # 取 content 做 _rank_and_merge 的 content[:100] 去重键，本臂图在 neo4j、
        # 逐命中回查太贵，故把同一份 corpus.texts 直接存进 FAISS payload（两臂
        # content 同源，去重行为逐位一致，约束 5 对称）。
        backend_store = self._vs._backend_store
        backend_store.create_index(index_type="flat", metric="inner_product")
        vecs = np.asarray(corpus.embeddings, dtype=np.float32)
        for start in range(0, n, FAISS_ADD_BATCH):
            end = min(start + FAISS_ADD_BATCH, n)
            backend_store.add_vectors(
                vecs[start:end], ids=corpus.session_ids[start:end],
                metadata=[{"node_id": corpus.session_ids[i],
                           "content": corpus.texts[i]}
                          for i in range(start, end)])
            print(f"[{self.name}] vectors {end}/{n} ({time.time() - t0:.0f}s)",
                  flush=True)

        be = self._gs._store_backend
        # 节点：UNWIND 批量 CREATE :SemanticaPerf{id,content}（无向量属性）
        for start in range(0, n, NEO4J_NODE_BATCH):
            end = min(start + NEO4J_NODE_BATCH, n)
            rows = [{"id": corpus.session_ids[i], "content": corpus.texts[i]}
                    for i in range(start, end)]
            be.execute_query(
                f"UNWIND $rows AS r "
                f"CREATE (n:{NEO4J_MARK} {{id: r.id, content: r.content}})",
                {"rows": rows})
            if (start // NEO4J_NODE_BATCH) % 10 == 9 or end == n:
                print(f"[{self.name}] neo4j nodes {end}/{n} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        # 边：UNWIND 批量、单写（query_graph 用无向 varlen，同 neug 臂的 copy_edges
        # 单写 + get_neighbors direction=both），MATCH 走 id 索引。
        ids = np.asarray(corpus.session_ids)
        edges = corpus.graph_edges
        m = len(edges)
        for start in range(0, m, NEO4J_EDGE_BATCH):
            end = min(start + NEO4J_EDGE_BATCH, m)
            rows = [{"f": str(ids[a]), "t": str(ids[b])}
                    for a, b in edges[start:end]]
            be.execute_query(
                f"UNWIND $rows AS r "
                f"MATCH (a:{NEO4J_MARK} {{id: r.f}}) "
                f"MATCH (b:{NEO4J_MARK} {{id: r.t}}) "
                f"CREATE (a)-[:{EDGE_TYPE}]->(b)",
                {"rows": rows})
            if (start // NEO4J_EDGE_BATCH) % 20 == 19 or end == m:
                print(f"[{self.name}] neo4j edges {end}/{m} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{self.name}] loaded in {time.time() - t0:.0f}s", flush=True)

    def query_vector(self, qvec, top_k):
        # FAISS 返回序即相关度降序（IP 度量），按原序取 id（与 native 臂逐位同）。
        res = self._vs.search_vectors(np.asarray(qvec, dtype=np.float32),
                                      k=top_k)
        return [r["id"] for r in res]

    # query_fts：neo4j 臂无全文路（同 native），不在 supported_classes，runner 标 N/A。

    def query_hybrid(self, qvec, keywords, top_k):
        from semantica.context.context_retriever import RetrievedContext

        # 纯向量路（无词法成分，同 native）：与 neug 三路融合的差异是能力差
        # （无 bm25 路），不是配置差，NOTES.md 里写明。content 取自 FAISS payload。
        results = []
        for r in self._vs.search_vectors(np.asarray(qvec, dtype=np.float32),
                                         k=top_k):
            results.append(RetrievedContext(
                content=r["metadata"]["content"],
                score=_faiss_ip_score(r),
                source=f"vector:{r['id']}", metadata={"node_id": r["id"]}))
        merged = self._retriever._rank_and_merge(results, " ".join(keywords))
        return [r.metadata["node_id"] for r in merged[:top_k]]

    def query_graph(self, seed_session_id, max_nodes):
        # 无向 *1..2 BFS，按 id 属性匹配（不走 facade get_neighbors：那要 neo4j
        # 内部整数 id，而本臂用裸 UNWIND 建点、无 _app_node_id_map 可桥接；且
        # semantica 自己的 neo4j 后端 `neo4j_store.py:895` 返回的是整个节点
        # `neighbor` + `labels`，比本类查询口径宽得多——本臂一直是短路口径）。
        # max_nodes 形参不适用（全局 top-N 截断会丢 hop2，契约 §6，同 neug/native）。
        # 与 neug 臂逐字同形（`WITH DISTINCT <node> RETURN <node>.id`，种子排除同样
        # 放在 Python 侧）：先按节点身份去重再投影 id。原写法 `RETURN DISTINCT nb.id`
        # 与本写法**结果集相同**（id 是主键），差别只在去重发生在投影前还是投影后；
        # neo4j 侧实测近中性（1.10×，见 graphiti-neo4j/NOTES.md §4.3），取同形是
        # 为了两臂口径逐字一致、杜绝漂移（同 cognee 三臂的做法）。
        # 种子排除不下推到 Cypher：neug 臂已实测下推会触发退化计划（1.55×），
        # 两臂同放 Python 侧既同形又避开引擎差异。
        seed = str(seed_session_id)
        recs = self._gs._store_backend.execute_query(
            f"MATCH (start:{NEO4J_MARK} {{id: $seed}})"
            f"-[:{EDGE_TYPE}*1..{GRAPH_DEPTH}]-(nb:{NEO4J_MARK}) "
            "WITH DISTINCT nb RETURN nb.id AS id",
            {"seed": seed})["records"]
        return [r["id"] for r in recs if r["id"] != seed]

    def teardown(self):
        # 共享服务端 courteous cleanup：分批清掉本臂 marker，避免 2.8M 边滞留拖累
        # graphiti-neo4j（下次 setup 也会清，双保险）；再关 bolt 连接。
        if self._gs is not None:
            try:
                self._clear(self._gs._store_backend)
            finally:
                self._gs.close()
                self._gs = None
