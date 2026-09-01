"""性能赛道统一 runner（LongMemEval-M，search 层直注）。

用法：
  export DASHSCOPE_API_KEY=...   # 本 runner 离线跑，不需要 LLM，仅占位
  python -m harness.perf.run_perf --adapter bruteforce-numpy
  python -m harness.perf.run_perf --list

流程（每 adapter）：setup -> load(直注预计算语料) -> warm-up ->
  4 类 query 逐条计时检索 -> 对比暴力 GT 算 recall -> teardown。
产出：results/perf/<adapter>/results.jsonl + summary.json

口径（benchmark-plan.md §3.4）：
- 延时一律 harness 侧 perf_counter 采集；load 耗时只作背景，不作对比指标。
- recall 对比同一份暴力 GT：vector=暴力余弦 top10、fts=AND 精确匹配、
  hybrid=RRF 融合 top10、graph=BFS 邻域（full-set recall，非 top-k）。
- 不支持的查询类标 N/A（能力完整性矩阵）。
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .base import (PerfAdapter, PerfCorpus, timed,
                   VECTOR_TOPK, FTS_KEYWORD, HYBRID, GRAPH_MULTIHOP, ALL_CLASSES)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "perf"
RESULTS = ROOT / "results" / "perf"
EMBED_MODEL = "text-embedding-v3"
TOP_K = 10
GRAPH_MAX_NODES = 150     # hop1(≤50)+hop2(≤100)
WARMUP = 3                # 每类正式计时前的预热查询数

ADAPTERS: dict[str, type[PerfAdapter]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        ADAPTERS[name] = cls
        return cls
    return deco


def _load_adapters():
    """延迟导入各 adapter。某系统实现尚未就位（文件缺失或依赖未装）时跳过注册。"""
    def _try(fn):
        try:
            fn()
        except ImportError:
            pass

    def _bruteforce():
        from .adapters.bruteforce import BruteForceAdapter
        ADAPTERS.setdefault(BruteForceAdapter.name, BruteForceAdapter)

    def _mem0():
        from .adapters.mem0_arm import Mem0NeuGPerfAdapter, Mem0QdrantPerfAdapter
        ADAPTERS.setdefault(Mem0NeuGPerfAdapter.name, Mem0NeuGPerfAdapter)
        ADAPTERS.setdefault(Mem0QdrantPerfAdapter.name, Mem0QdrantPerfAdapter)

    def _graphiti():
        from .adapters.graphiti_arm import GraphitiNeuGPerfAdapter, GraphitiNeo4jPerfAdapter
        ADAPTERS.setdefault(GraphitiNeuGPerfAdapter.name, GraphitiNeuGPerfAdapter)
        ADAPTERS.setdefault(GraphitiNeo4jPerfAdapter.name, GraphitiNeo4jPerfAdapter)

    def _cognee():
        from .adapters.cognee_arm import CogneeBuiltInPerfAdapter, CogneeNeuGPerfAdapter
        ADAPTERS.setdefault(CogneeNeuGPerfAdapter.name, CogneeNeuGPerfAdapter)
        ADAPTERS.setdefault(CogneeBuiltInPerfAdapter.name, CogneeBuiltInPerfAdapter)

    def _semantica():
        from .adapters.semantica_arm import SemanticaNativePerfAdapter, SemanticaNeuGPerfAdapter
        ADAPTERS.setdefault(SemanticaNeuGPerfAdapter.name, SemanticaNeuGPerfAdapter)
        ADAPTERS.setdefault(SemanticaNativePerfAdapter.name, SemanticaNativePerfAdapter)

    for fn in (_bruteforce, _mem0, _graphiti, _cognee, _semantica):
        _try(fn)


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_corpus() -> PerfCorpus:
    sessions = [json.loads(l) for l in (DATA / "corpus_sessions.jsonl").open()]
    session_ids = [s["session_id"] for s in sessions]
    texts = [s["text"] for s in sessions]
    embeddings = np.load(DATA / f"embed_{EMBED_MODEL}.npy")
    graph_edges = np.load(DATA / "graph_edges.npy")
    assert len(session_ids) == embeddings.shape[0], "session/向量行数不一致"
    return PerfCorpus(session_ids=session_ids, texts=texts,
                      embeddings=embeddings, graph_edges=graph_edges)


def load_queries() -> dict:
    def rd(name):
        return [json.loads(l) for l in (DATA / f"{name}.jsonl").open()]
    return {
        VECTOR_TOPK: rd("queries_vector"),
        FTS_KEYWORD: rd("queries_fts"),
        HYBRID: rd("queries_hybrid"),
        GRAPH_MULTIHOP: rd("queries_graph"),
    }


def recall_at_k(returned, gt_set, k=TOP_K):
    """top-k recall：返回前 k 中命中 GT 的比例（按 GT 规模归一）。"""
    if not gt_set:
        return None
    hits = len(set(returned[:k]) & gt_set)
    return hits / min(k, len(gt_set))


def recall_full(returned, gt_set):
    """full-set recall（graph 用）：返回邻域对 GT 邻域的覆盖。"""
    if not gt_set:
        return None
    return len(set(returned) & gt_set) / len(gt_set)


def _gt_for(cls, q):
    if cls == VECTOR_TOPK:
        return set(q["gt_top10_bruteforce"])
    if cls == FTS_KEYWORD:
        return set(q["gt_match_session_ids"])
    if cls == HYBRID:
        return set(q["gt_top10_rrf_bruteforce"])
    if cls == GRAPH_MULTIHOP:
        return set(q["gt_hop1_session_ids"]) | set(q["gt_hop2_session_ids"])
    raise ValueError(cls)


def _make_call(adapter, cls, q, qvecs, i):
    """返回无参 callable，供 timed() 计时。"""
    if cls == VECTOR_TOPK:
        qv = qvecs[i]
        return lambda: adapter.query_vector(qv, TOP_K)
    if cls == FTS_KEYWORD:
        kws = q["keywords"]
        return lambda: adapter.query_fts(kws, TOP_K)
    if cls == HYBRID:
        qv, kws = qvecs[i], q["keywords"]
        return lambda: adapter.query_hybrid(qv, kws, TOP_K)
    if cls == GRAPH_MULTIHOP:
        seed = q["seed_session_id"]
        return lambda: adapter.query_graph(seed, GRAPH_MAX_NODES)
    raise ValueError(cls)


def run_class(adapter, cls, qlist, qvecs):
    lats, recalls, detail = [], [], []
    # warm-up（不计时）：让后端索引/缓存进入稳态
    for w in range(min(WARMUP, len(qlist))):
        _make_call(adapter, cls, qlist[w], qvecs, w)()
    for i, q in enumerate(qlist):
        call = _make_call(adapter, cls, q, qvecs, i)
        returned, lat = timed(call)
        lats.append(lat)
        gt = _gt_for(cls, q)
        r = (recall_full(returned, gt) if cls == GRAPH_MULTIHOP
             else recall_at_k(returned, gt, TOP_K))
        if r is not None:
            recalls.append(r)
        detail.append({"qid": q["qid"], "latency_ms": round(lat, 3),
                       "recall": None if r is None else round(r, 4),
                       "n_returned": len(returned)})
    return {
        "class": cls, "status": "OK", "n": len(qlist),
        "p50_ms": round(percentile(lats, 50), 3),
        "p95_ms": round(percentile(lats, 95), 3),
        "mean_ms": round(statistics.mean(lats), 3),
        "recall_mean": round(statistics.mean(recalls), 4) if recalls else None,
        "detail": detail,
    }


def run_one(adapter: PerfAdapter, corpus: PerfCorpus, queries: dict, qvecs):
    out_dir = RESULTS / adapter.name
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    adapter.setup(str(work_dir))
    try:
        t0 = time.time()
        adapter.load(corpus)
        load_s = time.time() - t0
        print(f"[{adapter.name}] loaded {corpus.embeddings.shape[0]} sessions "
              f"in {load_s:.1f}s（背景成本，不作对比指标）")

        class_results = []
        for cls in ALL_CLASSES:
            if cls not in adapter.supported_classes:
                class_results.append({"class": cls, "status": "N/A"})
                print(f"[{adapter.name}] {cls}: N/A（不支持）")
                continue
            res = run_class(adapter, cls, queries[cls], qvecs)
            class_results.append(res)
            print(f"[{adapter.name}] {cls}: p50={res['p50_ms']}ms "
                  f"p95={res['p95_ms']}ms recall={res['recall_mean']}")

        # 落盘：逐 query 明细 + 汇总
        with (out_dir / "results.jsonl").open("w") as f:
            for res in class_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        summary = {
            "adapter": adapter.name,
            "n_sessions": corpus.embeddings.shape[0],
            "load_seconds_bg": round(load_s, 1),
            "top_k": TOP_K,
            "classes": {r["class"]: {k: v for k, v in r.items()
                                     if k not in ("detail", "class")}
                        for r in class_results},
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        adapter.teardown()


def main():
    _load_adapters()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="adapter name (see --list)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.adapter:
        print("registered adapters:", sorted(ADAPTERS) or "(none yet)")
        return 0 if args.list else 1

    cls = ADAPTERS.get(args.adapter)
    if cls is None:
        print(f"unknown adapter: {args.adapter}; available: {sorted(ADAPTERS)}")
        return 1

    print("loading corpus ...")
    corpus = load_corpus()
    queries = load_queries()
    qvecs = np.load(DATA / "query_vectors.npy")
    print(f"corpus: {corpus.embeddings.shape[0]} sessions, "
          f"dim={corpus.embeddings.shape[1]}, "
          f"edges={corpus.graph_edges.shape[0]}")

    adapter = cls()
    run_one(adapter, corpus, queries, qvecs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
