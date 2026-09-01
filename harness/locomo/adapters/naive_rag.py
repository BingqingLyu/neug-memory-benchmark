"""naive-rag 基线 adapter：session 切块 + DashScope embedding + Python 余弦 top-k。

定位（benchmark-plan.md）：不是对比对象，是 sanity anchor——
(1) 全量管线压测；(2) judge 链路校验（准确率应落在合理区间）；
(3) 文章参照线（Mem0 论文在 LOCOMO 上同样带 RAG baseline）。

无图、无 FTS、无索引——纯暴力向量检索，恰好是"暴力基线"的语义。
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

from ..base import SearchHit, SearchTrace, SystemAdapter
from ..llm import DEFAULT_BASE_URL

EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-v3")
CHUNK_CHARS = 1000   # 每块目标字符数（按 turn 边界切，不拆发言）
EMBED_BATCH = 10     # DashScope 兼容端点单次批量上限（保守值）


class NaiveRAGAdapter(SystemAdapter):
    name = "naive-rag"
    sub_scenario = "A"

    def __init__(self):
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.client = None
        self.cache_dir: Path | None = None

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        self.client = OpenAI(
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key, timeout=120,
        )

    def teardown(self):
        pass  # 纯内存态，无需 close

    # ---- 摄入：切块 + 嵌入（磁盘缓存，跨 run 复用）----
    def ingest_all(self, sessions):
        for s in sessions:
            self._chunk_session(s["sample_id"], s["session_idx"], s["text"])
        self.vectors = self._embed_all_cached()

    def ingest_session(self, sample_id, session_idx, date_time, text):
        self._chunk_session(sample_id, session_idx, text)

    def _chunk_session(self, sample_id, session_idx, text):
        lines = (text or "").split("\n")
        buf, size = [], 0
        ci = 0
        for ln in lines:
            if size + len(ln) > CHUNK_CHARS and buf:
                self._add_chunk(sample_id, session_idx, ci, "\n".join(buf))
                ci += 1
                buf, size = [], 0
            buf.append(ln)
            size += len(ln) + 1
        if buf:
            self._add_chunk(sample_id, session_idx, ci, "\n".join(buf))

    def _add_chunk(self, sample_id, session_idx, ci, text):
        self.chunks.append({
            "id": f"{sample_id}:{session_idx}:{ci}",
            "sample_id": sample_id,
            "text": text,
        })

    def _embed_all_cached(self) -> np.ndarray:
        ids = [c["id"] for c in self.chunks]
        key = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        cache_vec = self.cache_dir / f"embed_{EMBED_MODEL}.npy"
        cache_meta = self.cache_dir / f"embed_{EMBED_MODEL}.meta.json"
        if cache_vec.exists() and cache_meta.exists():
            meta = json.loads(cache_meta.read_text())
            if meta.get("key") == key and meta.get("model") == EMBED_MODEL:
                vecs = np.load(cache_vec)
                if len(vecs) == len(ids):
                    print(f"[naive-rag] embedding cache hit ({len(ids)} chunks)")
                    return vecs

        vecs = np.zeros((len(self.chunks), 0))
        out = []
        for i in range(0, len(self.chunks), EMBED_BATCH):
            batch = [c["text"][:8000] for c in self.chunks[i:i + EMBED_BATCH]]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            out.extend([d.embedding for d in resp.data])
            if (i // EMBED_BATCH) % 20 == 0:
                print(f"[naive-rag] embedded {min(i + EMBED_BATCH, len(self.chunks))}/{len(self.chunks)}")
        vecs = np.asarray(out, dtype=np.float32)
        # L2 归一化 -> 余弦即点积
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        np.save(cache_vec, vecs)
        cache_meta.write_text(json.dumps({"key": key, "model": EMBED_MODEL, "n": len(ids)}))
        return vecs

    # ---- 检索：暴力余弦 top-k ----
    def search(self, question: str, top_k: int = 20):
        if self.vectors is None or len(self.chunks) == 0:
            self.vectors = self._embed_all_cached()
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[question])
        q = np.asarray(resp.data[0].embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        scores = self.vectors @ q
        top = np.argsort(-scores)[:top_k]
        hits = [
            SearchHit(id=self.chunks[i]["id"], score=float(scores[i]),
                      payload=self.chunks[i]["text"])
            for i in top
        ]
        return hits, SearchTrace(extra={"n_chunks": len(self.chunks)})
