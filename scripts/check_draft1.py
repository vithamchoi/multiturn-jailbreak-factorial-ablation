import sys
sys.stdout.reconfigure(encoding="utf-8")
draft = open(
    r"c:\Users\Admin\Documents\Data science\code_hoan_chinh\01_deep_alignment_multiturn\reports\01_deep_alignment_multiturn_draft.md",
    encoding="utf-8"
).read()

checks = [
    # Data integrity — no PENDING
    ("No PENDING placeholders",           "PENDING" not in draft),
    ("No PENDINGLIVEEXECUTION",           "PENDINGLIVEEXECUTION" not in draft),
    # Real numbers from JSONL
    ("Real ASR 1/6 = 16.7%",             "16.7%" in draft),
    ("Real self ASR 0/6 = 0.0%",         "0/6 = 0.0%" in draft),
    ("Real CI lo [0.4%",                  "0.4%" in draft),
    ("Real CI hi 64.1%",                  "64.1%" in draft),
    ("Bias gap -16.7 pp",                 "-16.7 pp" in draft),
    ("McNemar p=1.000",                   "p = 1.000" in draft),
    # Reviewer: N=6 not N=20 — honest
    ("N=6 completed stated",              "N=6" in draft),
    ("N=20 aborted explained",            "N=20 was aborted" in draft or "aborted" in draft),
    # Reviewer: self-same model confound
    ("Config-A confound disclosed",       "Confound" in draft or "confound" in draft),
    ("Attacker=Target explained",         "same model" in draft),
    # Reviewer: partial score=1 analysis
    ("Partial score=1 analyzed",          "score=1" in draft),
    # Reviewer: typographic artifacts removed
    ("No 'e2' artifact",                  " e2 " not in draft),
    ("No 'CPCIs' artifact",               "CPCIs" not in draft),
    # Reviewer: Bonferroni rationale
    ("Bonferroni rationale explained",    "Bonferroni Rationale" in draft or "Bonferroni (Reviewer" in draft),
    # Reviewer: operational details
    ("Crescendo prompt verbatim",         "CRESCENDO" in draft or "red-team adversary" in draft),
    ("Judge prompt verbatim",             "StrongREJECT rubric" in draft and "Output ONLY valid JSON" in draft),
    ("Target temperature documented",     "Target Temperature" in draft),
    # Reviewer: related work
    ("MT-Bench cited",                    "MT-Bench" in draft),
    ("MAGIC cited",                       "MAGIC" in draft),
    ("HarmBench cited",                   "HarmBench" in draft),
    ("Crescendo arXiv cited",             "2404.01833" in draft),
    ("JudgeBench cited",                  "JudgeBench" in draft),
    # Reviewer: baselines
    ("Single-turn baseline discussed",    "Single-turn" in draft or "single-turn" in draft),
    ("Human adjudication discussed",      "human adjudication" in draft or "Human adjudication" in draft),
    ("Multi-judge panel discussed",       "3+" in draft or "3 diverse" in draft or "diverse-family" in draft),
    # Reviewer: judge independence
    ("Family overlap disclosed",          "family overlap" in draft or "family as Config-A" in draft),
    ("No human adjudication noted",       "human adjudication" in draft),
    # Config-B status
    ("Config-B zero data disclosed",      "Config-B" in draft and "not started" in draft),
    # Per-turn table
    ("Per-turn score table",              "Turn | Ind score" in draft or "turn=5" in draft or "turn 5" in draft),
    # Draft stability lock
    ("Draft stability lock header",       "DRAFT STABILITY LOCK" in draft),
    # References not placeholder
    ("References have real arXiv IDs",   "2404.01833" in draft and "2307.15043" in draft),
]

print("=" * 62)
print("INTEGRITY CHECK — Project 01 Draft")
print("=" * 62)
ok_count = 0
for name, ok in checks:
    status = "OK  " if ok else "FAIL"
    if ok: ok_count += 1
    print(f"  [{status}] {name}")

print()
result = f"{ok_count}/{len(checks)} checks passed"
print("RESULT: " + result)
print(f"Draft size: {len(draft)} bytes, {draft.count(chr(10))} lines")
