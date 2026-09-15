"""LoCoMo 统一评测 runner。

用法：
  export DASHSCOPE_API_KEY=...
  python -m harness.locomo.run_eval --adapter mem0-neug --runs 3
  python -m harness.locomo.run_eval --list

流程（每 adapter × run）：setup -> ingest_all(272 session) ->
  对 1540 题逐题 search -> answer(qwen-plus) -> judge(qwen-max) -> 记录。
产出：results/locomo/<adapter>/run_<n>.jsonl + summary.json
"""
import argparse
import json
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

from .base import SystemAdapter, timed_search
from .judge import build_answer_prompt, build_judge_prompt, parse_verdict
from .llm import DEFAULT_ANSWER_MODEL, DEFAULT_JUDGE_MODEL, LLMClient

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "locomo"

# adapter 注册表：集成就绪一个就登记一个（Qoder 侧 adapter 完成后接入）
ADAPTERS: dict[str, type[SystemAdapter]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        ADAPTERS[name] = cls
        return cls
    return deco


def _load_adapters():
    """延迟导入各 adapter，避免与注册表的循环依赖。

    各系统 adapter 独立成包：某系统实现尚未就位（文件缺失或依赖未装）时
    跳过注册，不影响其它系统运行。
    """
    def _try(fn):
        try:
            fn()
        except ImportError:
            pass

    def _naive_rag():
        from .adapters.naive_rag import NaiveRAGAdapter
        ADAPTERS.setdefault(NaiveRAGAdapter.name, NaiveRAGAdapter)

    def _graphiti():
        from .adapters.graphiti_arm import GraphitiNeo4jAdapter, GraphitiNeuGAdapter
        ADAPTERS.setdefault("graphiti-neug", GraphitiNeuGAdapter)
        ADAPTERS.setdefault("graphiti-neo4j", GraphitiNeo4jAdapter)

    def _mem0():
        from .adapters.mem0_arm import (Mem0NeuGAdapter, Mem0PgvectorAdapter,
                                        Mem0QdrantAdapter, Mem0QdrantServerAdapter)
        ADAPTERS.setdefault(Mem0NeuGAdapter.name, Mem0NeuGAdapter)
        ADAPTERS.setdefault(Mem0QdrantAdapter.name, Mem0QdrantAdapter)
        ADAPTERS.setdefault(Mem0QdrantServerAdapter.name, Mem0QdrantServerAdapter)
        ADAPTERS.setdefault(Mem0PgvectorAdapter.name, Mem0PgvectorAdapter)

    def _cognee():
        from .adapters.cognee_arm import CogneeDefaultAdapter, CogneeNeuGAdapter
        ADAPTERS.setdefault(CogneeNeuGAdapter.name, CogneeNeuGAdapter)
        ADAPTERS.setdefault(CogneeDefaultAdapter.name, CogneeDefaultAdapter)

    def _semantica():
        from .adapters.semantica import (SemanticaNativeAdapter,
                                         SemanticaNeo4jAdapter,
                                         SemanticaNeuGAdapter)
        ADAPTERS.setdefault(SemanticaNeuGAdapter.name, SemanticaNeuGAdapter)
        ADAPTERS.setdefault(SemanticaNativeAdapter.name, SemanticaNativeAdapter)
        ADAPTERS.setdefault(SemanticaNeo4jAdapter.name, SemanticaNeo4jAdapter)

    for fn in (_naive_rag, _graphiti, _mem0, _cognee, _semantica):
        _try(fn)


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_one(adapter: SystemAdapter, qa: list, sessions: list, llm: LLMClient,
            run_idx: int, top_k: int, limit: int = 0, workers: int = 1,
            limit_sessions: int = 0):
    if limit_sessions:
        sessions = sessions[:limit_sessions]
        sample_ids = {s["sample_id"] for s in sessions}
        qa = [q for q in qa if q["sample_id"] in sample_ids]
    out_dir = RESULTS / adapter.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # 查询阶段只读且摄入确定性：run>0 复用 run0 的库，不重复摄入。
    work_dir = out_dir / (f"work_run{run_idx}" if run_idx == 0 else "work_run0")
    cache_dir = RESULTS / "cache" / adapter.name.split("-")[0]  # 按系统共享抽取缓存
    marker = work_dir / ".ingest_complete"
    if marker.exists():
        # 摄入规模变化（如 --limit-sessions 调整）则整库重建，避免新旧混用。
        if json.loads(marker.read_text()).get("sessions") != len(sessions):
            shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    adapter.setup(str(work_dir), str(cache_dir))
    try:
        if marker.exists():
            print(f"[{adapter.name}] reusing ingested store in {work_dir} (run {run_idx})")
            adapter.reuse_ingested(sessions)
        else:
            print(f"[{adapter.name}] ingesting {len(sessions)} sessions ...")
            t0 = time.time()
            adapter.ingest_all(sessions)
            marker.write_text(json.dumps({"sessions": len(sessions)}))
            print(f"[{adapter.name}] ingest done in {time.time()-t0:.1f}s")

        questions = qa[:limit] if limit else qa
        search_lock = threading.Lock()  # adapter 不保证线程安全，检索加锁；LLM 调用并行

        def process_q(q):
            with search_lock:
                hits, trace = timed_search(adapter, q["question"], top_k=top_k,
                                           sample_id=q.get("sample_id"))
            context = "\n".join(h.payload for h in hits)
            answer = llm.answer(build_answer_prompt(context, q["question"]))
            verdict = parse_verdict(
                llm.judge(build_judge_prompt(q["question"], q["answer"], answer))
            )
            return {
                "sample_id": q["sample_id"], "category": q["category"],
                "question": q["question"], "reference": q["answer"],
                "model_answer": answer, "verdict": verdict,
                "search_latency_ms": round(trace.latency_ms, 3),
                "context_chars": trace.context_chars, "n_hits": len(hits),
            }

        rows = []
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            rows = [None] * len(questions)
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(process_q, q): i for i, q in enumerate(questions)}
                for fut in as_completed(futs):
                    rows[futs[fut]] = fut.result()
                    done += 1
                    if done % 100 == 0:
                        acc = sum(r["verdict"] == "CORRECT" for r in rows[:done] if r) / done
                        print(f"[{adapter.name}] {done}/{len(questions)} acc~{acc:.3f}")
        else:
            for i, q in enumerate(questions):
                rows.append(process_q(q))
                if (i + 1) % 100 == 0:
                    acc = sum(r["verdict"] == "CORRECT" for r in rows) / len(rows)
                    print(f"[{adapter.name}] {i+1}/{len(questions)} acc={acc:.3f}")

        out = out_dir / f"run_{run_idx}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return rows
    finally:
        adapter.teardown()


def summarize(adapter_name: str, all_rows: list[list[dict]]):
    summary = {"adapter": adapter_name, "runs": len(all_rows),
               "answer_model": DEFAULT_ANSWER_MODEL, "judge_model": DEFAULT_JUDGE_MODEL}
    per_run = []
    for rows in all_rows:
        if not rows:
            continue
        acc = sum(r["verdict"] == "CORRECT" for r in rows) / len(rows)
        by_cat = {}
        for c in sorted({r["category"] for r in rows}):
            sub = [r for r in rows if r["category"] == c]
            by_cat[str(c)] = round(sum(r["verdict"] == "CORRECT" for r in sub) / len(sub), 4)
        lat = [r["search_latency_ms"] for r in rows]
        per_run.append({"accuracy": round(acc, 4), "by_category": by_cat,
                        "p50_ms": round(percentile(lat, 50), 2),
                        "p95_ms": round(percentile(lat, 95), 2)})
    summary["per_run"] = per_run
    if per_run:
        summary["accuracy_mean"] = round(statistics.mean(p["accuracy"] for p in per_run), 4)
        if len(per_run) > 1:
            summary["accuracy_std"] = round(statistics.stdev(p["accuracy"] for p in per_run), 4)
    out = RESULTS / adapter_name / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    _load_adapters()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="adapter name (see --list)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="只摄入前 N 个 session，题目同步收窄到对应样本（POC 调试用）")
    ap.add_argument("--workers", type=int, default=8,
                    help="QA 阶段并发数（检索加锁、LLM 并行）；1=串行")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.adapter:
        print("registered adapters:", sorted(ADAPTERS) or "(none yet — 集成就绪后注册)")
        return 0 if args.list else 1

    cls = ADAPTERS.get(args.adapter)
    if cls is None:
        print(f"unknown adapter: {args.adapter}; available: {sorted(ADAPTERS)}")
        return 1

    qa = load_jsonl(DATA / "qa_eval.jsonl")
    sessions = load_jsonl(DATA / "sessions.jsonl")
    llm = LLMClient()

    all_rows = []
    for run_idx in range(args.runs):
        print(f"=== {args.adapter} run {run_idx} ===")
        all_rows.append(run_one(cls(), qa, sessions, llm, run_idx, args.top_k,
                                args.limit, args.workers, args.limit_sessions))
    summarize(args.adapter, all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
