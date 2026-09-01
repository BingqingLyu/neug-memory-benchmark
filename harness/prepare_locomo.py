"""LoCoMo-10 数据集准备：生成统一 harness 的评测集与摄入输入。

输入: data/locomo10.json（snap-research/locomo 官方数据）
输出:
  data/processed/qa_eval.jsonl   — cat1-4 共 1540 题（Zep 修正协议口径）
  data/processed/sessions.jsonl  — 按对话×session 展开的摄入输入
  data/processed/manifest.json   — 数据集统计与来源信息

协议对齐（benchmark-plan.md 3.1）：
  - 排除 cat5（446 题对抗性）
  - OpenViking 官方用 1528 题（差 12 题），本集以 snap-research 原始数据为准
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

EVAL_CATEGORIES = {1, 2, 3, 4}


def main():
    convs = json.loads((DATA / "locomo10.json").read_text())
    assert len(convs) == 10, f"expect 10 conversations, got {len(convs)}"

    qa_rows = []
    session_rows = []
    cats = Counter()

    for conv in convs:
        sample_id = conv["sample_id"]
        conversation = conv["conversation"]
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")

        # 展开 session_N / session_N_date_time
        idxs = sorted(
            int(m.group(1))
            for k in conversation
            if (m := re.fullmatch(r"session_(\d+)", k))
        )
        for i in idxs:
            turns = conversation.get(f"session_{i}") or []
            lines = []
            for t in turns:
                if not isinstance(t, dict):
                    continue
                speaker = t.get("speaker", "")
                text = (t.get("text") or "").strip()
                if speaker == "speaker_a" and speaker_a:
                    speaker = speaker_a
                elif speaker == "speaker_b" and speaker_b:
                    speaker = speaker_b
                if text:
                    lines.append(f"{speaker}: {text}")
            session_rows.append({
                "sample_id": sample_id,
                "session_idx": i,
                "date_time": conversation.get(f"session_{i}_date_time"),
                "n_turns": len(lines),
                "text": "\n".join(lines),
            })

        for q in conv.get("qa", []):
            cat = q.get("category")
            cats[cat] += 1
            if cat not in EVAL_CATEGORIES:
                continue
            qa_rows.append({
                "sample_id": sample_id,
                "question": q.get("question"),
                "answer": q.get("answer"),
                "category": cat,
                "evidence": q.get("evidence"),
            })

    with (OUT / "qa_eval.jsonl").open("w") as f:
        for r in qa_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "sessions.jsonl").open("w") as f:
        for r in session_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "source": "https://github.com/snap-research/locomo (data/locomo10.json)",
        "n_conversations": len(convs),
        "n_sessions": len(session_rows),
        "qa_by_category_raw": {str(k): v for k, v in sorted(cats.items(), key=lambda x: str(x[0]))},
        "qa_eval_count": len(qa_rows),
        "protocol": "cat1-4, 排除 cat5（对齐 Zep 修正口径 zep-papers#5）",
        "note": "OpenViking 官方 1528 题与本集 1540 题差 12 题，文中注明",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    total_chars = sum(len(r["text"]) for r in session_rows)
    print(f"session text total chars: {total_chars} (~{total_chars // 4} tokens)")


if __name__ == "__main__":
    main()
