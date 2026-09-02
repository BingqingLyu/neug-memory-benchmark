"""semantica LoCoMo adapter：semantica-neug / semantica-native 双臂。

HANDOFF-semantica-benchmark.md（§2 质量赛道）：
- 接原生抽取管线：GraphBuilder.build（spacy NER + pattern 三元组，零 LLM），
  抽取产物按 (system, sample_id) 键控落 cache_dir，两配置共享同一份产物。
- semantica-neug：NeuG 单库承载三路检索——HNSW 向量 + bm25 全文 + 图遍历，
  融合用 Semantica 原生 ContextRetriever._rank_and_merge。
- semantica-native：内存 ContextGraph + 原生 FAISS vector store，检索走
  ContextRetriever.retrieve() 原生的向量+图两路；不接全文路（原生栈无
  全文，见 HANDOFF §1 源码事实），也不写客户端关键词兜底（决策 B3）。
- 唯一变量是后端：抽取产物、embedding（text-embedding-v3 内容哈希缓存）、
  hybrid_alpha、top_k、切块策略两配置完全一致。

契约：harness/locomo/ADAPTER-CONTRACT.md
- 会话隔离：每 sample_id 一个独立库（neug 文件库 / native 内存图）（约束 1）
- search 只返回检索上下文，答题由 harness 统一做（约束 2）
- teardown 正常 close（约束 6）。引擎多实例坑：同进程关闭一个实例会破坏
  其他实例的向量查询，故多样本库全程保活、仅在 teardown 统一关
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

from ..base import SearchHit, SearchTrace, SystemAdapter
from ..llm import DEFAULT_BASE_URL

# semantica / semantica_patch 经 benchmark venv 的 editable 安装提供，
# 与 graphiti/mem0/cognee 三系统同惯例（裸导入，无 sys.path 注入）。

EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-v3")
EMBED_DIM = int(os.environ.get("BENCH_EMBED_DIM", "1024"))
CHUNK_CHARS = 1000   # 与 naive-rag 同口径：按 turn 边界切，不拆发言
EMBED_BATCH = 10     # DashScope 兼容端点单次批量上限（保守值）
HYBRID_ALPHA = 0.5   # 0=vector only, 1=graph only（Semantica 原生融合权重）
GRAPH_SEEDS = 3      # 图扩展的种子数（向量 top 命中）


def _chunk_text(text: str) -> list[str]:
    """按行切块（与 naive-rag 同函数语义：目标长度、不拆发言）。"""
    lines = (text or "").split("\n")
    chunks, buf, size = [], [], 0
    for ln in lines:
        if size + len(ln) > CHUNK_CHARS and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _norm_vec(v: list[float]) -> list[float]:
    """客户端归一化双保险（NEUG-DIALECT-NOTES：余弦按未归一化向量计算）。"""
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else arr.tolist()


class _CachedEmbedder:
    """把 adapter 的哈希缓存 _embed 适配成 VectorStore.embedder 接口。"""

    def __init__(self, embed_fn):
        self._embed_fn = embed_fn

    def generate_embeddings(self, texts):
        if isinstance(texts, str):
            return np.asarray(self._embed_fn([texts])[0], dtype=np.float32)
        return [np.asarray(v, dtype=np.float32) for v in self._embed_fn(texts)]


class _SemanticaBase(SystemAdapter):
    """两臂共享：嵌入缓存、抽取产物、统一的摄入行构造。"""
    sub_scenario = "A"

    def __init__(self):
        self._work_dir: Path | None = None
        self._cache_dir: Path | None = None
        self._client = None
        self._embed_cache_path: Path | None = None
        self._embed_cache: dict[str, list[float]] = {}
        self._ingested: set[tuple[str, int]] = set()

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        self._client = OpenAI(
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key, timeout=120,
        )
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._embed_cache_path = self._cache_dir / f"semantica_embed_{EMBED_MODEL}.jsonl"
        self._load_embed_cache()

    # ---- embedding（内容哈希磁盘缓存，协议 3/6，两臂共享）----
    @staticmethod
    def _content_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _load_embed_cache(self):
        if not self._embed_cache_path.exists():
            return
        with self._embed_cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._embed_cache[rec["k"]] = rec["v"]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._content_key(t) for t in texts]
        out = [self._embed_cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(missing), EMBED_BATCH):
            batch_idx = missing[s:s + EMBED_BATCH]
            resp = self._client.embeddings.create(
                model=EMBED_MODEL, input=[texts[i][:8000] for i in batch_idx])
            for j, d in enumerate(resp.data):
                i = batch_idx[j]
                vec = [float(x) for x in d.embedding]
                out[i] = vec
                self._embed_cache[keys[i]] = vec
                with self._embed_cache_path.open("a") as f:
                    f.write(json.dumps({"k": keys[i], "v": vec}) + "\n")
        return out

    # ---- 抽取产物（按 (system, sample_id) 键控，两臂共享）----
    def _artifact_path(self, sample_id: str) -> Path:
        return self._cache_dir / f"semantica_extract_{sample_id}.json"

    def _get_extraction(self, sample_id: str, session_idx: int,
                        date_time: str, text: str) -> dict:
        """返回该 session 的抽取产物；缺失则跑原生管线并落盘共享。"""
        path = self._artifact_path(sample_id)
        artifact = {"sample_id": sample_id, "method": "spacy+pattern",
                    "sessions": {}}
        if path.exists():
            artifact = json.loads(path.read_text())
        key = str(session_idx)
        if key in artifact["sessions"]:
            return artifact["sessions"][key]

        # 原生抽取：spacy NER + pattern 三元组（GraphBuilder 默认，零 LLM）
        from semantica.kg.graph_builder import GraphBuilder
        graph = GraphBuilder(resolve_conflicts=False).build(
            text, ner_method="spacy", extract_relations=False,
            extract_triplets=True)
        sess = {
            "date_time": date_time or "",
            "entities": [
                {"id": e.get("id"), "name": e.get("name"),
                 "type": e.get("type", "UNKNOWN")}
                for e in graph["entities"] if e.get("id")
            ],
            "relationships": [
                {"source": r.get("source"), "target": r.get("target"),
                 "type": r.get("type", "RELATED_TO")}
                for r in graph["relationships"]
                if r.get("source") and r.get("target")
            ],
        }
        artifact["sessions"][key] = sess
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=1))
        os.replace(tmp, path)
        return sess

    # ---- 摄入：两臂统一的行构造（切块 + 抽取产物 -> 节点/边）----
    def _build_rows(self, sample_id: str, session_idx: int,
                    date_time: str, text: str, sess_art: dict,
                    prev_head: str | None):
        chunks = _chunk_text(text)
        node_rows, edge_rows = [], []
        for ci, chunk in enumerate(chunks):
            node_rows.append({
                "id": f"{sample_id}:{session_idx}:{ci}", "etype": "chunk",
                "content": chunk,
                "extra": json.dumps({"date_time": date_time or "",
                                     "session_idx": session_idx},
                                    ensure_ascii=False),
            })
        edge_rows.extend(
            {"from_id": f"{sample_id}:{session_idx}:{ci}",
             "to_id": f"{sample_id}:{session_idx}:{ci + 1}",
             "edge_type": "NEXT", "weight": 1.0}
            for ci in range(len(chunks) - 1))
        if prev_head and chunks:
            edge_rows.append({"from_id": prev_head,
                              "to_id": f"{sample_id}:{session_idx}:0",
                              "edge_type": "FOLLOWS", "weight": 1.0})

        # 抽取实体节点（跨 session 同名实体同 id 自然合并）
        texts_lower = [c.lower() for c in chunks]
        for ent in sess_art.get("entities", []):
            name = str(ent["name"])
            node_rows.append({
                "id": f"ent:{sample_id}:{name}", "etype": str(ent["type"]),
                "content": name,
                "extra": json.dumps({"session_idx": session_idx},
                                    ensure_ascii=False),
            })
            # 实体 -> 提及它的块（确定性字符串包含链接）
            nl = name.lower()
            for ci, tl in enumerate(texts_lower):
                if nl in tl:
                    edge_rows.append({
                        "from_id": f"ent:{sample_id}:{name}",
                        "to_id": f"{sample_id}:{session_idx}:{ci}",
                        "edge_type": "MENTIONED", "weight": 1.0})
        # 三元组边（实体-实体）
        for rel in sess_art.get("relationships", []):
            edge_rows.append({
                "from_id": f"ent:{sample_id}:{rel['source']}",
                "to_id": f"ent:{sample_id}:{rel['target']}",
                "edge_type": str(rel["type"]), "weight": 1.0})
        return node_rows, edge_rows, chunks

    def ingest_session(self, sample_id: str, session_idx: int,
                       date_time: str, text: str):
        if (sample_id, session_idx) in self._ingested:
            return
        sess_art = self._get_extraction(sample_id, session_idx, date_time, text)
        prev_head = self._prev_head_for(sample_id)
        node_rows, edge_rows, chunks = self._build_rows(
            sample_id, session_idx, date_time, text, sess_art, prev_head)
        if not node_rows:
            return
        # 节点向量：块与实体同一口径（内容哈希缓存，两臂共享零新增调用）
        vecs = self._embed([r["content"] for r in node_rows])
        for row, vec in zip(node_rows, vecs):
            row["vec"] = _norm_vec(vec)
        self._flush(sample_id, node_rows, edge_rows)
        if chunks:
            self._set_prev_head(sample_id, f"{sample_id}:{session_idx}:0")
        self._ingested.add((sample_id, session_idx))

    # ---- 子类钩子 ----
    def _prev_head_for(self, sample_id: str) -> str | None:
        raise NotImplementedError

    def _set_prev_head(self, sample_id: str, head: str):
        raise NotImplementedError

    def _flush(self, sample_id: str, node_rows: list[dict], edge_rows: list[dict]):
        raise NotImplementedError

    def reuse_ingested(self, sessions):
        raise NotImplementedError

    def teardown(self):
        pass


class SemanticaNeuGAdapter(_SemanticaBase):
    """with NeuG：一个嵌入式库承载向量 + 全文 + 图三路检索。"""
    name = "semantica-neug"

    def __init__(self):
        super().__init__()
        # sample_id -> {"gs": GraphStore, "vec": NeugVectorStore, "fts": NeugFTS}
        self._stores: dict[str, dict] = {}
        self._prev_head: dict[str, str] = {}
        self._retriever = None

    def setup(self, work_dir: str, cache_dir: str):
        super().setup(work_dir, cache_dir)
        from semantica.context.context_retriever import ContextRetriever
        self._retriever = ContextRetriever(hybrid_alpha=HYBRID_ALPHA)

    def teardown(self):
        # 引擎多实例坑：仅在此统一关闭，关闭后不再有任何查询。
        for st in self._stores.values():
            st["gs"].close()
        self._stores.clear()

    def reuse_ingested(self, sessions):
        for s in sessions:
            self._ensure_store(s["sample_id"])

    def _prev_head_for(self, sample_id):
        return self._prev_head.get(sample_id)

    def _set_prev_head(self, sample_id, head):
        self._prev_head[sample_id] = head

    def _ensure_store(self, sample_id: str) -> dict:
        st = self._stores.get(sample_id)
        if st is not None:
            return st
        from semantica.graph_store.graph_store import GraphStore
        from semantica_patch import NeugFTS, NeugVectorStore

        gs = GraphStore(
            backend="neug",
            db_path=str(self._work_dir / f"semantica_{sample_id}.db"),
            vector_dim=EMBED_DIM,
        )
        gs.connect()
        store = gs._store_backend
        st = {"gs": gs, "vec": NeugVectorStore(store=store), "fts": NeugFTS(store)}
        self._stores[sample_id] = st
        return st

    def _flush(self, sample_id, node_rows, edge_rows):
        backend = self._ensure_store(sample_id)["gs"]._store_backend
        backend.copy_nodes(node_rows)
        if edge_rows:
            backend.copy_edges(edge_rows)

    # ---- 检索：三路（向量 + bm25 + 图扩展）走原生融合 ----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        if sample_id is None:
            if len(self._stores) != 1:
                raise RuntimeError("sample_id required when multiple samples ingested")
            sample_id = next(iter(self._stores))
        st = self._stores.get(sample_id)
        if st is None:
            return [], SearchTrace(extra={"error": f"no store for {sample_id}"})

        from semantica.context.context_retriever import RetrievedContext

        qvec = self._embed([question])[0]
        results: list[RetrievedContext] = []

        # 路 1：HNSW 向量
        vec_hits = st["vec"].search(qvec, top_k=top_k)
        for h in vec_hits:
            results.append(RetrievedContext(
                content=h["metadata"].get("content", ""), score=h["score"],
                source=f"vector:{h['id']}", metadata={"node_id": h["id"]},
            ))

        # 路 2：bm25 全文（relevance 已在 hit 列表内归一到 [0,1]）
        fts_hits = st["fts"].search(question, limit=top_k)
        for h in fts_hits:
            results.append(RetrievedContext(
                content=h["content"], score=h["relevance"],
                source=f"vector:fts:{h['id']}", metadata={"node_id": h["id"]},
            ))

        # 路 3：图扩展（向量 top 种子的一跳邻居）
        for h in vec_hits[:GRAPH_SEEDS]:
            for n in st["gs"].get_neighbors(h["id"], depth=1):
                nb_id = n["id"]
                results.append(RetrievedContext(
                    content=n["properties"].get("content", ""), score=0.5,
                    source=f"graph:{nb_id}", metadata={"node_id": nb_id},
                    related_entities=[{"id": h["id"], "type": "seed"}],
                ))

        # Semantica 原生融合：分路归一化 + hybrid_alpha 加权 + 去重
        merged = self._retriever._rank_and_merge(results, question)
        hits = [
            SearchHit(
                id=str(r.metadata.get("node_id") or hash(r.content)),
                score=float(r.score), payload=r.content,
            )
            for r in merged[:top_k] if r.content
        ]
        trace = SearchTrace(extra={
            "vector_hits": len(vec_hits), "fts_hits": len(fts_hits),
            "graph_hits": sum(1 for r in results if r.source.startswith("graph:")),
            "merged": len(merged),
        })
        return hits, trace


class SemanticaNativeAdapter(_SemanticaBase):
    """without NeuG 对照：内存 ContextGraph + 原生 FAISS，检索走
    ContextRetriever.retrieve() 的向量+图两路（原生栈无全文，不接 fts）。"""
    name = "semantica-native"

    def __init__(self):
        super().__init__()
        # sample_id -> {"cg": ContextGraph, "vs": VectorStore, "rt": ContextRetriever}
        self._stores: dict[str, dict] = {}
        self._prev_head: dict[str, str] = {}

    def reuse_ingested(self, sessions):
        # 内存栈不跨进程持久：复用 run 时基于共享缓存（抽取产物 + 嵌入）
        # 零 API 调用重建，语义等价于读回已摄入的库。
        for s in sessions:
            if (s["sample_id"], s["session_idx"]) not in self._ingested:
                self.ingest_session(s["sample_id"], s["session_idx"],
                                    s.get("date_time", ""), s["text"])

    def _prev_head_for(self, sample_id):
        return self._prev_head.get(sample_id)

    def _set_prev_head(self, sample_id, head):
        self._prev_head[sample_id] = head

    def _ensure_store(self, sample_id: str) -> dict:
        st = self._stores.get(sample_id)
        if st is not None:
            return st
        from semantica.context.context_graph import ContextGraph
        from semantica.context.context_retriever import ContextRetriever
        from semantica.vector_store.vector_store import VectorStore

        cg = ContextGraph()
        vs = VectorStore(backend="faiss", config={"dimension": EMBED_DIM})
        # 查询/重排嵌入走同一份 text-embedding-v3 哈希缓存（与 neug 臂同口径）
        vs.embedder = _CachedEmbedder(self._embed)
        rt = ContextRetriever(knowledge_graph=cg, vector_store=vs,
                              hybrid_alpha=HYBRID_ALPHA)
        st = {"cg": cg, "vs": vs, "rt": rt}
        self._stores[sample_id] = st
        return st

    def _flush(self, sample_id, node_rows, edge_rows):
        st = self._ensure_store(sample_id)
        cg, vs = st["cg"], st["vs"]
        cg.add_nodes([
            {"id": r["id"], "type": r["etype"], "content": r["content"]}
            for r in node_rows
        ])
        if edge_rows:
            cg.add_edges([
                {"source_id": e["from_id"], "target_id": e["to_id"],
                 "type": e["edge_type"], "weight": e.get("weight", 1.0)}
                for e in edge_rows
            ])
        vecs = np.asarray([r["vec"] for r in node_rows], dtype=np.float32)
        backend_store = vs._backend_store
        if backend_store.index is None:
            # inner_product + 归一向量 == 余弦（与 neug HNSW cosine 同口径）
            backend_store.create_index(index_type="flat", metric="inner_product")
        backend_store.add_vectors(
            vecs, ids=[r["id"] for r in node_rows],
            metadata=[{"content": r["content"], "node_id": r["id"]}
                      for r in node_rows])

    # ---- 检索：原生两路（向量 + 图），无全文 ----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        if sample_id is None:
            if len(self._stores) != 1:
                raise RuntimeError("sample_id required when multiple samples ingested")
            sample_id = next(iter(self._stores))
        st = self._stores.get(sample_id)
        if st is None:
            return [], SearchTrace(extra={"error": f"no store for {sample_id}"})

        merged = st["rt"].retrieve(question, max_results=top_k)
        hits = [
            SearchHit(
                id=str(r.metadata.get("node_id") or hash(r.content)),
                score=float(r.score), payload=r.content,
            )
            for r in merged if r.content
        ]
        trace = SearchTrace(extra={
            "routes": "vector+graph", "merged": len(merged),
            "fts": "N/A (native stack has no full-text route)",
        })
        return hits, trace
