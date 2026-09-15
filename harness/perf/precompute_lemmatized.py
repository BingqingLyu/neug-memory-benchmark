"""预计算全量 corpus 的 text_lemmatized 缓存（多进程加速），移出 benchmark load 计时。

背景（实测）：lemmatize_for_bm25（spaCy en_core_web，完整 tagger→lemmatizer pipeline）
对本语料 92ms/row（每行 text 均 ~1006 词），单进程全量 51661 行 = 79min。这是 pgvector
臂 load 的真瓶颈（pgvector insert 本身仅 ~1s/500 行、HNSW+GIN 增量 ~2ms/row），也是
mem0 core add()（main.py:1026 lemmatize_for_bm25）对**所有 backend** 的固定预处理成本。

处理：与 corpus 的 embeddings（embed_*.npy）/graph_edges 预计算同理，把 lemmatize 移出
benchmark 的 load_seconds_bg——一次性多进程算好、存 data/processed/perf/text_lemmatized.jsonl
（按 corpus 行序，session_id↔lemma 对齐），各臂 load 直接读缓存，不再重复 lemmatize。

用法：PYTHONPATH=. python harness/perf/precompute_lemmatized.py
"""
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("MEM0_TELEMETRY", "False")

from harness.perf.run_perf import DATA, load_corpus  # noqa: E402
from mem0.utils.lemmatization import lemmatize_for_bm25  # noqa: E402

OUT = DATA / "text_lemmatized.jsonl"
# CPU 密集 + 每 worker 各加载一份 spaCy 模型（数百 MB）；cap 到 6 平衡速度与内存，
# 呼应本仓“并行度不超过物理核、避免内存打爆”的纪律（AGENTS.md）。
NPROC = min(os.cpu_count() or 4, 6)


def _lem(text):
    # 每 worker 进程内 get_nlp_lemma() 是 lazy singleton（spawn 模式各自加载 spaCy）
    return lemmatize_for_bm25(text) or text


def main():
    corpus = load_corpus()
    texts = corpus.texts
    n = len(texts)
    print(f"lemmatizing {n} texts with {NPROC} processes -> {OUT}", flush=True)
    t = time.time()
    results = [None] * n
    done = 0
    with Pool(NPROC) as pool:
        # imap 保序（输出顺序=输入顺序），results[i] 与 corpus 行号对齐
        for i, lem in enumerate(pool.imap(_lem, texts, chunksize=32)):
            results[i] = lem
            done += 1
            if done % 5000 == 0 or done == n:
                el = time.time() - t
                print(f"  {done}/{n} ({el:.0f}s, {el/done*1000:.1f}ms/row)", flush=True)
    with OUT.open("w") as f:
        for sid, lem in zip(corpus.session_ids, results):
            f.write(json.dumps({"session_id": sid, "text_lemmatized": lem},
                               ensure_ascii=False) + "\n")
    print(f"done: {n} texts in {time.time()-t:.0f}s, wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
