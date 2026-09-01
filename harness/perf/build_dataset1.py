"""WS5 性能赛道数据集 1（A 类：agent 对话记忆数据）构建管线。

语料：LongMemEval-M（cleaned）——500 题 × 每题约 500 session 历史，
去重后得到共享 session 池（所有系统摄入同一份）。

四类 query × 50（benchmark-plan.md §3.2）：
  vector_topk   语义 top-k，GT = 暴力余弦 top-10 + answer_session 命中
  fts_keyword   关键词（来自 answer session 的高 IDF 词），GT = 全含词集合（AND）
  hybrid        向量粗筛→BM25 重排，GT = RRF(暴力向量序, 暴力 BM25 序) top-10
  graph_multihop 规范共现图上 2 跳扩展，GT = 精确 BFS 邻域
                （跨系统只比延时；recall 仅系统内 with/without 可比，见 manifest 声明）

阶段（可分步重跑，全部落盘缓存）：
  extract  流式解析 M json → corpus_sessions.jsonl + questions.jsonl
  embed    session 池向量化（DashScope text-embedding-v3，npy 缓存）
  derive   四类 query 派生 + ground truth 计算
"""
import argparse
import json
import math
import os
import re
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "LongMemEval" / "data"
OUT = ROOT / "data" / "processed" / "perf"
M_FILE = RAW / "longmemeval_m_cleaned.json"

SEED = 42
N_PER_CLASS = 50
EMBED_MODEL = "text-embedding-v3"
EMBED_BATCH = 10
RRF_K = 60
STOPWORDS = set("""a an the and or but if then else when at by for with about against
between into through during before after above below to from up down in out on off over
under again further once here there all any both each few more most other some such no
nor not only own same so than too very can will just don should now i you he she it we
they me him her us them my your his its our their what which who whom this that these
those am is are was were be been being have has had do does did doing would could might
must may shall of as""".split())
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")


def session_text(session) -> str:
    lines = []
    for t in session:
        role = t.get("role", "")
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------- stage: extract ----------------
def extract():
    import ijson
    OUT.mkdir(parents=True, exist_ok=True)
    pool = {}          # sid -> (date, text)
    questions = []
    with M_FILE.open("rb") as f:
        for i, item in enumerate(ijson.items(f, "item")):
            sids = item["haystack_session_ids"]
            dates = item.get("haystack_dates", [None] * len(sids))
            for sid, date, sess in zip(sids, dates, item["haystack_sessions"]):
                if sid not in pool:
                    pool[sid] = (date, session_text(sess))
            questions.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "question_date": item.get("question_date"),
                "answer_session_ids": item["answer_session_ids"],
                "n_haystack": len(sids),
            })
            if (i + 1) % 50 == 0:
                print(f"extracted {i+1} questions, pool={len(pool)}")
    with (OUT / "corpus_sessions.jsonl").open("w") as f:
        for sid, (date, text) in pool.items():
            f.write(json.dumps({"session_id": sid, "date": date, "text": text},
                               ensure_ascii=False) + "\n")
    with (OUT / "questions.jsonl").open("w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    total_chars = sum(len(t) for _, t in pool.values())
    print(f"DONE extract: {len(questions)} questions, {len(pool)} unique sessions, "
          f"~{total_chars//4} tokens")


# ---------------- stage: embed ----------------
def load_pool():
    sessions = [json.loads(l) for l in (OUT / "corpus_sessions.jsonl").open()]
    return sessions


def embed():
    from openai import OpenAI
    sessions = load_pool()
    ids = [s["session_id"] for s in sessions]
    key = ",".join(ids[:5]) + f"...n={len(ids)}"
    cache_vec = OUT / f"embed_{EMBED_MODEL}.npy"
    cache_meta = OUT / f"embed_{EMBED_MODEL}.meta.json"
    if cache_vec.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if meta.get("n") == len(ids) and meta.get("key") == key:
            print("embed cache hit")
            return
    client = OpenAI(base_url=os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.environ["DASHSCOPE_API_KEY"])
    vecs = []
    for i in range(0, len(sessions), EMBED_BATCH):
        batch = [s["text"][:6000] for s in sessions[i:i + EMBED_BATCH]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        if (i // EMBED_BATCH) % 50 == 0:
            print(f"embedded {min(i+EMBED_BATCH, len(sessions))}/{len(sessions)}")
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    np.save(cache_vec, mat)
    cache_meta.write_text(json.dumps({"n": len(ids), "key": key,
                                      "model": EMBED_MODEL, "dim": int(mat.shape[1])}))
    print(f"DONE embed: {mat.shape}")


# ---------------- helpers ----------------
def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_index(sessions):
    """token -> (doc_freq, {sid_idx: count})；返回 (df, postings, doc_tokens)"""
    df = Counter()
    postings = defaultdict(dict)
    doc_tokens = []
    for idx, s in enumerate(sessions):
        toks = tokenize(s["text"])
        doc_tokens.append(toks)
        for tok, c in Counter(toks).items():
            df[tok] += 1
            postings[tok][idx] = c
    return df, postings, doc_tokens


def bm25_scores(query_tokens, df, postings, doc_tokens, n_docs, k1=1.5, b=0.75):
    avgdl = sum(len(t) for t in doc_tokens) / max(n_docs, 1)
    scores = np.zeros(n_docs, dtype=np.float64)
    for tok in set(query_tokens):
        post = postings.get(tok)
        if not post:
            continue
        idf = math.log(1 + (n_docs - df[tok] + 0.5) / (df[tok] + 0.5))
        for idx, tf in post.items():
            dl = len(doc_tokens[idx])
            scores[idx] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    return scores


def rrf(rankings, k=RRF_K):
    """rankings: list of idx arrays (best-first). 返回融合分 dict。"""
    fused = Counter()
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[int(idx)] += 1.0 / (k + rank + 1)
    return fused


# ---------------- stage: derive ----------------
def derive():
    sessions = load_pool()
    sid2idx = {s["session_id"]: i for i, s in enumerate(sessions)}
    n = len(sessions)
    questions = [json.loads(l) for l in (OUT / "questions.jsonl").open()]
    mat = np.load(OUT / f"embed_{EMBED_MODEL}.npy")
    df, postings, doc_tokens = build_index(sessions)

    rng = random.Random(SEED)
    # 按 question_type 分层抽 50 题（vector/hybrid 共用）
    by_type = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)
    # 均匀分配 50 题到各类型，余数分配给前几个类型
    n_types = len(by_type)
    base_per_type = N_PER_CLASS // n_types
    remainder = N_PER_CLASS % n_types
    sampled = []
    for i, t in enumerate(sorted(by_type)):
        rng.shuffle(by_type[t])
        n_sample = base_per_type + (1 if i < remainder else 0)
        sampled.extend(by_type[t][:n_sample])
    assert len(sampled) == N_PER_CLASS, f"sampled {len(sampled)}, expected {N_PER_CLASS}"

    idf = {tok: math.log(1 + n / freq) for tok, freq in df.items()}

    def salient_tokens(text, topk=8):
        toks = [t for t in tokenize(text) if t not in STOPWORDS and t in idf]
        toks.sort(key=lambda t: -idf[t])
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= topk:
                break
        return out

    # ---- class 1: vector_topk ----
    vec_queries = []
    client_emb = None
    q_vecs = []
    from openai import OpenAI
    client_emb = OpenAI(base_url=os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.environ["DASHSCOPE_API_KEY"])
    for q in sampled:
        resp = client_emb.embeddings.create(model=EMBED_MODEL, input=[q["question"]])
        qv = np.asarray(resp.data[0].embedding, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        q_vecs.append(qv)
    q_mat = np.stack(q_vecs)
    # 持久化 query 向量（vec/hybrid 共用此 50 题顺序），供 run_perf 离线直注
    np.save(OUT / "query_vectors.npy", q_mat.astype(np.float32))
    sims = q_mat @ mat.T  # (50, n)
    for i, q in enumerate(sampled):
        order = np.argsort(-sims[i])
        top10 = [sessions[j]["session_id"] for j in order[:10]]
        gt_oracle = [s for s in q["answer_session_ids"] if s in sid2idx]
        vec_queries.append({
            "qid": f"vec-{i:03d}", "class": "vector_topk",
            "query": q["question"], "question_id": q["question_id"],
            "question_type": q["question_type"],
            "gt_top10_bruteforce": top10,
            "gt_oracle_session_ids": gt_oracle,
        })

    # ---- class 2: fts_keyword ----
    fts_queries = []
    fts_sample = []
    for i, t in enumerate(sorted(by_type)):
        offset = base_per_type + (1 if i < remainder else 0)
        n_fts = base_per_type + (1 if i < remainder else 0)
        fts_sample.extend(by_type[t][offset:offset + n_fts])
    fts_sample = fts_sample[:N_PER_CLASS]
    while len(fts_sample) < N_PER_CLASS:
        fts_sample.append(rng.choice(questions))
    for i, q in enumerate(fts_sample):
        ans_text = " ".join(
            sessions[sid2idx[s]]["text"] for s in q["answer_session_ids"] if s in sid2idx)
        kws = salient_tokens(ans_text, topk=6)[:3]
        if not kws:
            continue
        gt = [sessions[j]["session_id"] for j in range(n)
              if all(tok in set(doc_tokens[j]) for tok in kws)]
        fts_queries.append({
            "qid": f"fts-{i:03d}", "class": "fts_keyword",
            "keywords": kws, "question_id": q["question_id"],
            "question_type": q["question_type"],
            "gt_match_session_ids": gt,
            "gt_size": len(gt),
        })

    # ---- class 3: hybrid（与 vector 同 50 题）----
    hybrid_queries = []
    for i, q in enumerate(sampled):
        vec_rank = np.argsort(-sims[i])[:200]
        q_toks = [t for t in tokenize(q["question"]) if t not in STOPWORDS]
        bm = bm25_scores(q_toks, df, postings, doc_tokens, n)
        bm_rank = np.argsort(-bm)[:200]
        fused = rrf([vec_rank, bm_rank])
        top10 = [sessions[j]["session_id"]
                 for j, _ in sorted(fused.items(), key=lambda x: -x[1])[:10]]
        hybrid_queries.append({
            "qid": f"hyb-{i:03d}", "class": "hybrid",
            "query": q["question"], "keywords": q_toks,
            "question_id": q["question_id"], "question_type": q["question_type"],
            "gt_top10_rrf_bruteforce": top10,
            "gt_oracle_session_ids": [s for s in q["answer_session_ids"] if s in sid2idx],
        })

    # ---- class 4: graph_multihop（规范共现图）----
    # 边：共享 >=2 个 IDF>=阈值 的 token；token 文档频 <=50（排除泛化词）
    MAX_DF = 50
    MIN_IDF = math.log(1 + n / MAX_DF)
    adj = defaultdict(set)
    tok_sessions = defaultdict(list)
    for idx in range(n):
        for tok in set(doc_tokens[idx]):
            if idf.get(tok, 0) >= MIN_IDF and df[tok] <= MAX_DF:
                tok_sessions[tok].append(idx)
    print(f"graph: salient tokens={len(tok_sessions)}, building edges...")
    for tok, idxs in tok_sessions.items():
        if len(idxs) > 30:
            continue
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                a, b2 = idxs[a_pos], idxs[b_pos]
                adj[a].add(b2)
                adj[b2].add(a)
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"graph: nodes={n}, edges={n_edges}")
    # 持久化无向边表（a<b 去重），供 run_perf 直注各后端做 graph_multihop
    edge_pairs = np.asarray(
        sorted((a, b) for a, nbrs in adj.items() for b in nbrs if a < b),
        dtype=np.int32).reshape(-1, 2)
    np.save(OUT / "graph_edges.npy", edge_pairs)
    print(f"graph: saved edge list {edge_pairs.shape}")

    def bfs2(start):
        d1 = adj.get(start, set())
        d2 = set()
        for x in d1:
            d2 |= adj.get(x, set())
        return sorted(d1), sorted(d2 - d1 - {start})

    graph_queries = []
    gi = 0
    for q in fts_sample:
        for sid in q["answer_session_ids"]:
            if sid not in sid2idx:
                continue
            sidx = sid2idx[sid]
            if len(adj.get(sidx, set())) < 2:
                continue
            toks = salient_tokens(sessions[sidx]["text"], topk=3)
            if not toks:
                continue
            hop1, hop2 = bfs2(sidx)
            graph_queries.append({
                "qid": f"grp-{gi:03d}", "class": "graph_multihop",
                "seed_keyword": toks[0], "seed_session_id": sid,
                "question_id": q["question_id"], "question_type": q["question_type"],
                "gt_hop1_session_ids": [sessions[j]["session_id"] for j in hop1[:50]],
                "gt_hop2_session_ids": [sessions[j]["session_id"] for j in hop2[:100]],
                "note": "canonical co-occurrence graph; cross-system latency only, "
                        "recall comparable within with/without pair",
            })
            gi += 1
            break
        if gi >= N_PER_CLASS:
            break

    for name, data in [("queries_vector", vec_queries), ("queries_fts", fts_queries),
                       ("queries_hybrid", hybrid_queries), ("queries_graph", graph_queries)]:
        with (OUT / f"{name}.jsonl").open("w") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)}")

    manifest = {
        "source": "longmemeval_m_cleaned.json (hf xiaowu0162/longmemeval-cleaned)",
        "seed": SEED, "n_sessions_pool": n,
        "n_questions_total": len(questions),
        "question_types": {t: len(v) for t, v in by_type.items()},
        "embed_model": EMBED_MODEL,
        "graph_edges": n_edges, "graph_max_df": MAX_DF,
        "comparability": {
            "vector_topk": "recall@10 vs brute-force GT + oracle hit; cross-system OK",
            "fts_keyword": "recall@10 vs exact AND-match GT; cross-system OK",
            "hybrid": "overlap@10 vs RRF-of-brute-force GT; cross-system OK (GT 定义统一)",
            "graph_multihop": "latency cross-system; recall only within with/without pair",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("DONE derive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "embed", "derive", "all"])
    args = ap.parse_args()
    if args.stage in ("extract", "all"):
        extract()
    if args.stage in ("embed", "all"):
        embed()
    if args.stage in ("derive", "all"):
        derive()


if __name__ == "__main__":
    main()
