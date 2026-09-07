"""hybrid recall 归因实验（同一题目 50 道，GT=oracle RRF top10）。

变体：
  raw-vec/raw-fts      : mem0 臂同款的 50 候选池（原始表）
  lem-fts              : 词形归一表（qdrant 臂同款口径）的 BM25 池
  rrf(raw) / rrf(lem)  : 在池内做标准 RRF（与 GT 构造同算法族）
  additive             : mem0 score_and_rank 同款（候选=向量池，BM25 只加分，
                         threshold=0.1）
"""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ["MEM0_TELEMETRY"] = "False"

# 仓库根（本文件在 harness/perf/ 下，parents[2] 即根）：所有路径相对它派生，
# 可用 env 覆盖，不再硬编码任何机器本地绝对路径。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.scoring import get_bm25_params, normalize_bm25, score_and_rank
from mem0.vector_stores.neug import NeuG

DATA = os.environ.get("BENCH_PERF_DATA", str(ROOT / "data/processed/perf"))
WORK = os.environ.get("BENCH_MEM0_NEUG_WORK", str(ROOT / "results/perf/mem0-neug/work"))
TOP_K, POOL, TH = 10, 50, 0.1

text2sid = {}
for line in open(f"{DATA}/corpus_sessions.jsonl"):
    o = json.loads(line)
    text2sid[o["text"]] = o["session_id"]

store_raw = NeuG(collection_name="mem0", embedding_model_dims=1024, db_path=f"{WORK}/neug.db")
store_lem = NeuG(collection_name="lem", embedding_model_dims=1024, db_path=f"{WORK}/neug.db")
qvecs = np.load(f"{DATA}/query_vectors.npy")
hyb = [json.loads(l) for l in open(f"{DATA}/queries_hybrid.jsonl")]

def sid_of(row):
    p = row.payload or {}
    return p.get("session_id") or text2sid.get(p.get("data", ""))

def recall_at_k(returned, gt):
    return len(set(returned[:TOP_K]) & gt) / min(TOP_K, len(gt)) if gt else None

def rrf(pools):
    scores = {}
    for pool in pools:
        for rank, sid in enumerate(pool):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

R = {k: [] for k in ("vec", "fts_raw", "fts_lem", "rrf_raw", "rrf_lem", "additive", "cov_raw", "cov_lem")}
for i, q in enumerate(hyb):
    gt = set(q["gt_top10_rrf_bruteforce"])
    qv, kws = qvecs[i].tolist(), q["keywords"]
    kw = " ".join(kws)

    vrows = store_raw.search(q["query"], qv, top_k=POOL, filters={"user_id": "perf-corpus"})
    vpool = [sid_of(r) for r in vrows]
    frows_r = store_raw.keyword_search(kw, top_k=POOL, filters={"user_id": "perf-corpus"}) or []
    fpool_r = [sid_of(r) for r in frows_r]
    kw_lem = lemmatize_for_bm25(kw)
    frows_l = store_lem.keyword_search(kw_lem, top_k=POOL, filters={"user_id": "perf-corpus"}) or []
    fpool_l = [sid_of(r) for r in frows_l]

    R["cov_raw"].append(len((set(vpool) | set(fpool_r)) & gt) / len(gt))
    R["cov_lem"].append(len((set(vpool) | set(fpool_l)) & gt) / len(gt))
    R["vec"].append(recall_at_k(vpool, gt))
    R["fts_raw"].append(recall_at_k(fpool_r, gt))
    R["fts_lem"].append(recall_at_k(fpool_l, gt))
    R["rrf_raw"].append(recall_at_k(rrf([vpool, fpool_r]), gt))
    R["rrf_lem"].append(recall_at_k(rrf([vpool, fpool_l]), gt))

    # mem0 score_and_rank 同款：候选=向量池，BM25 归一后加分
    mid, steep = get_bm25_params(kw, lemmatized=kw_lem)
    bm = {str(r.id): normalize_bm25(r.score, mid, steep) for r in frows_l if r.score and r.score > 0}
    cands = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in vrows]
    ranked = score_and_rank(cands, bm, {}, TH, TOP_K)
    R["additive"].append(recall_at_k([sid_of(c) for c in [type('x', (), {'payload': c['payload']})() for c in ranked]], gt))

for k, vals in R.items():
    print(f"{k:10s}: {np.mean(vals):.4f}")
store_raw.close(); store_lem.close()
