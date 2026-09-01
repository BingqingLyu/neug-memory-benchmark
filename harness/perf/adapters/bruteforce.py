"""参考实现：numpy 暴力后端（无索引、全量现算）。

两个作用：
1. 验证 run_perf 的 recall 计算正确——暴力解 == derive 的 ground truth，
   四类 query 的 recall 都应 ≈ 1.0；若偏离说明 runner 或 GT 有 bug。
2. 作为"暴力现算"延时基线，对照索引化后端（NeuG HNSW/FTS vs O(N) 扫描）。

注意：本 adapter 不代表任何真实系统，不进 with/without 对比表，只作 sanity + 基线。
"""
from collections import defaultdict

import numpy as np

from ..base import (PerfAdapter, VECTOR_TOPK, FTS_KEYWORD, HYBRID,
                    GRAPH_MULTIHOP)
from ..build_dataset1 import tokenize, build_index, bm25_scores, rrf


class BruteForceAdapter(PerfAdapter):
    name = "bruteforce-numpy"
    supported_classes = frozenset(
        {VECTOR_TOPK, FTS_KEYWORD, HYBRID, GRAPH_MULTIHOP})

    def setup(self, work_dir: str):
        pass

    def load(self, corpus):
        self.session_ids = corpus.session_ids
        self.mat = corpus.embeddings
        self.n = len(self.session_ids)
        self._sid2idx = {sid: i for i, sid in enumerate(self.session_ids)}
        # BM25 倒排索引（复用 derive 的同一实现，保证口径一致）
        self.df, self.postings, self.doc_tokens = build_index(
            [{"text": t} for t in corpus.texts])
        self._doc_toksets = [set(toks) for toks in self.doc_tokens]
        # 规范共现图邻接
        self.adj = defaultdict(set)
        for a, b in corpus.graph_edges:
            self.adj[int(a)].add(int(b))
            self.adj[int(b)].add(int(a))

    # ---- vector_topk：暴力余弦 ----
    def query_vector(self, qvec, top_k):
        sims = self.mat @ qvec
        order = np.argsort(-sims)[:top_k]
        return [self.session_ids[i] for i in order]

    # ---- fts_keyword：AND 语义 + BM25 排序 ----
    def query_fts(self, keywords, top_k):
        scores = bm25_scores(keywords, self.df, self.postings,
                             self.doc_tokens, self.n)
        kwset = set(keywords)
        # GT 为全含词集合（AND）：只保留含全部关键词的 session 再按 BM25 排
        mask = np.fromiter((kwset.issubset(ts) for ts in self._doc_toksets),
                           dtype=bool, count=self.n)
        scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)[:top_k]
        return [self.session_ids[i] for i in order if scores[i] >= 0]

    # ---- hybrid：向量粗筛→BM25 重排，RRF 融合（与 derive GT 同口径）----
    def query_hybrid(self, qvec, keywords, top_k):
        sims = self.mat @ qvec
        vec_rank = np.argsort(-sims)[:200]
        bm = bm25_scores(keywords, self.df, self.postings, self.doc_tokens, self.n)
        bm_rank = np.argsort(-bm)[:200]
        fused = rrf([vec_rank, bm_rank])
        order = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        return [self.session_ids[i] for i, _ in order]

    # ---- graph_multihop：规范共现图 2 跳 BFS（与 derive bfs2 同口径）----
    def query_graph(self, seed_session_id, max_nodes):
        start = self._sid2idx.get(seed_session_id)
        if start is None:
            return []
        d1 = self.adj.get(start, set())
        d2 = set()
        for x in d1:
            d2 |= self.adj.get(x, set())
        d2 -= d1 | {start}
        # GT 按 hop 独立截断（hop1[:50]、hop2[:100]）；参考实现逐 hop 对齐，
        # 保证 returned ⊇ GT、recall=1.0（校验 runner 用）。真实 adapter 返回
        # 完整邻域时同样 ⊇ GT，recall 口径一致。
        out = sorted(d1)[:50] + sorted(d2)[:100]
        return [self.session_ids[i] for i in out]

    def teardown(self):
        pass
