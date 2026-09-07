"""抽取冻结：单臂 cognee 摄入驱动（每臂独立进程，避免 cognee 配置单例串库）。

用途：方案1（冻结抽取产物）的执行器。两臂用同一 BENCH_LLM_SEED + LLM_TEMPERATURE=0，
发出字节一致的抽取请求体；先跑 arm A 把响应灌进代理缓存，再跑 arm B 复用同一缓存 →
arm B 的抽取调用应全部命中、原样回放 arm A 的响应 → 两后端建立在同一份抽取之上。

冻结完整性的判据（backend 无关）：arm B 摄入期间代理缓存**新增条目 ≈ 0**。
用 GET /stats 的 cached_entries 在每臂前后取差即可，无需跨后端 diff 存储。

用法（每臂一个独立进程）：
  python -m harness.locomo.freeze_ingest --backend default --work-dir <dir> [--sessions N]
  python -m harness.locomo.freeze_ingest --backend neug    --work-dir <dir> [--sessions N]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["default", "neug"])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--sessions", type=int, default=0, help="只摄入前 N 个 session（0=全部）")
    ap.add_argument("--fresh", action="store_true", default=True,
                    help="摄入前清空 work_dir（默认开）：cognee incremental_loading 会对"
                         "遗留关系库跳过已摄入 data item（此前 13.9s 假摄入），冻结抽取"
                         "必须从空 work_dir 起。")
    ap.add_argument("--no-fresh", dest="fresh", action="store_false")
    args = ap.parse_args()

    from .adapters.cognee_arm import CogneeDefaultAdapter, CogneeNeuGAdapter

    sessions = [json.loads(l) for l in (DATA / "sessions.jsonl").open() if l.strip()]
    if args.sessions:
        sessions = sessions[: args.sessions]

    cls = CogneeDefaultAdapter if args.backend == "default" else CogneeNeuGAdapter
    adapter = cls()
    # cognee BaseConfig 强制绝对路径（COGNEE_LOGS_DIR 等由 work_dir 派生），
    # 传相对路径会触发 pydantic ValidationError，故此处 resolve 成绝对路径。
    wd = Path(args.work_dir).resolve()
    if args.fresh and wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True, exist_ok=True)
    cache_dir = wd / "_shared_cache"
    cache_dir.mkdir(exist_ok=True)

    adapter.setup(str(wd), str(cache_dir))
    t0 = time.time()
    try:
        adapter.ingest_all(sessions)
        # 写 run_eval 认可的 marker，令 `run_eval --runs 1` 复用本冻结库、只跑检索+答题+判分
        (wd / ".ingest_complete").write_text(json.dumps({"sessions": len(sessions)}))
    finally:
        adapter.teardown()
    print(f"[freeze] backend={args.backend} sessions={len(sessions)} "
          f"wall={time.time()-t0:.1f}s work_dir={wd}")


if __name__ == "__main__":
    main()
