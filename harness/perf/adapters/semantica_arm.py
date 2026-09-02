"""semantica 性能赛道 adapter：semantica-neug / semantica-native 双臂。

契约对齐（PERF-ADAPTER-CONTRACT.md）：
- 存储层直注、绕 LLM 抽取：预计算向量/文本/规范共现图原样载入，全程 0 次
  LLM/embedding 调用（约束 1/2）。
- NEUG 臂：COPY FROM JSONL 批量冲刷（index-first：HNSW+FTS 先建后 COPY，
  第一步 P13 已验证路径），冲刷前客户端去重（重复 PK 静默跳过）。
- NATIVE 臂：Semantica 存储层直写——内存 ContextGraph（add_nodes/add_edges）
  + 原生 FAISS vector store（flat + inner_product，归一向量上等价余弦）。
- 四类查询走系统检索路径（约束 3）：
  vector = neug HNSW vector_distance_cosine / native FAISS search_vectors
  fts    = neug bm25；native 无全文路（原生栈源码事实），标 N/A（B3）
  hybrid = Semantica 原生 _rank_and_merge 融合（neug: 向量+bm25；
           native: 纯向量——无词法成分，是能力差不是配置差）
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

# semantica / semantica_patch 经 benchmark venv 的 editable 安装提供，
# 与 graphiti/mem0/cognee 三系统同惯例（裸导入，无 sys.path 注入）。

from ..base import (  # noqa: E402
    ALL_CLASSES,
    GRAPH_MULTIHOP,
    HYBRID,
    PerfAdapter,
    VECTOR_TOPK,
)

NODE_BATCH = 5000          # neug 节点 COPY 分片（节点带 1024 维向量，单批过大会 OOM）
# NeuG 每次 COPY FROM 调用有一笔 ∝ 表内已有行数 的固定开销（CSR/PK 重建），
# 批内插入本身极快。实测 2.8M 边：拆 14 批 = 3674s，单次 COPY = 2.6s（>1000x）。
# 边行不含向量（JSONL ~250MB），故 neug 臂边载入合并为单次 COPY。
NEUG_EDGE_COPY_BATCH = 5_000_000   # > 语料边数（2799244）→ 单次 COPY
EDGE_BATCH = 200_000       # native ContextGraph.add_edges 分片（无此惩罚，按内存分片）
FAISS_ADD_BATCH = 20_000   # native FAISS add_vectors 批
EDGE_TYPE = "COOCCURS"     # 规范共现边类型（两臂一致）


class SemanticaNeuGPerfAdapter(PerfAdapter):
    """with NeuG：一个嵌入式库承载向量 + 全文 + 图三类检索。"""
    name = "semantica-neug"
    supported_classes = frozenset(ALL_CLASSES)

    def __init__(self):
        self._gs = None
        self._vec = None
        self._fts = None
        self._retriever = None

    def setup(self, work_dir):
        from semantica.graph_store.graph_store import GraphStore
        from semantica_patch import NeugFTS, NeugVectorStore
        from semantica.context.context_retriever import ContextRetriever

        db_dir = os.path.join(work_dir, "semantica.db")
        shutil.rmtree(db_dir, ignore_errors=True)
        # index-first：connect() 先建 HNSW + FTS 索引，再 COPY 冲刷
        self._gs = GraphStore(backend="neug", db_path=db_dir, vector_dim=1024)
        self._gs.connect()
        store = self._gs._store_backend
        self._vec = NeugVectorStore(store=store)
        self._fts = NeugFTS(store)
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
            rows = [
                {"from_id": str(ids[a]), "to_id": str(ids[b]),
                 "edge_type": EDGE_TYPE, "weight": 1.0}
                for a, b in edges[start:end]
            ]
            backend.copy_edges(rows)
            print(f"[{self.name}] edges {end}/{m} ({time.time() - t0:.0f}s)",
                  flush=True)

    def query_vector(self, qvec, top_k):
        return [h["id"] for h in self._vec.search(list(qvec), top_k=top_k)]

    def query_fts(self, keywords, top_k):
        return [h["id"] for h in self._fts.search(" ".join(keywords),
                                                  limit=top_k)]

    def query_hybrid(self, qvec, keywords, top_k):
        from semantica.context.context_retriever import RetrievedContext

        results = []
        for h in self._vec.search(list(qvec), top_k=top_k):
            results.append(RetrievedContext(
                content=h["metadata"].get("content", ""), score=h["score"],
                source=f"vector:{h['id']}", metadata={"node_id": h["id"]}))
        for h in self._fts.search(" ".join(keywords), limit=top_k):
            results.append(RetrievedContext(
                content=h["content"], score=h["relevance"],
                source=f"vector:fts:{h['id']}", metadata={"node_id": h["id"]}))
        merged = self._retriever._rank_and_merge(results, " ".join(keywords))
        return [r.metadata["node_id"] for r in merged[:top_k]]

    def query_graph(self, seed_session_id, max_nodes):
        # varlen 服务端遍历（无向 *1..2），返回完整邻域 ⊇ 按 hop 截断的 GT；
        # max_nodes 形参不适用（全局 top-N 截断会丢 hop2，契约 §6）。
        return [n["id"] for n in self._gs.get_neighbors(seed_session_id,
                                                        depth=2)]

    def teardown(self):
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
        self._content: dict[str, str] = {}

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
        self._content = dict(zip(corpus.session_ids, corpus.texts))

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
            results.append(RetrievedContext(
                content=self._content.get(r["id"], ""), score=r["score"],
                source=f"vector:{r['id']}", metadata={"node_id": r["id"]}))
        merged = self._retriever._rank_and_merge(results, " ".join(keywords))
        return [r.metadata["node_id"] for r in merged[:top_k]]

    def query_graph(self, seed_session_id, max_nodes):
        # hops=2 BFS 出边（双向写入 == 无向），返回完整邻域 ⊇ GT。
        return [n["id"] for n in self._cg.get_neighbors(seed_session_id,
                                                        hops=2)]

    def teardown(self):
        # 内存栈无需 close；保持空实现与接口对齐。
        self._content = {}
