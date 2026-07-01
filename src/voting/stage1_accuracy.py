"""
voting/stage1_accuracy.py — Stage 1 routing accuracy for the two-stage method.

Reports:
  (a) Per-true-language Stage 1 coarse routing accuracy
      (how often true-MY/ID/EN/ZH/TA is correctly routed to MSID vs mis-routed)
  (b) Error breakdown for two_stage_agree:
      Stage 1 misroute vs Stage 2 wrong ms/id call

Writes: log/log_stage1_accuracy_N.txt
"""

import sys
import numpy as np
from collections import defaultdict

from voting.core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, BUCKET_ORDER, next_path,
    load_dataset, load_lingua, run_predictions, hard_vote,
)

LOG_PATH = next_path(LOG_DIR, "log_stage1_accuracy", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 90

print(f"Log: {LOG_PATH}")
print("Loading models...")
detector = load_lingua()
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} cases. Running predictions...")
results = run_predictions(cases, detector)
print("Done.\n")

# ── Stage 1 logic ─────────────────────────────────────────────────────────────
MSID = "msid"
COARSE_TARGETS = {"en": "en", "ms": MSID, "id": MSID, "zh": "zh", "ta": "ta"}

def collapse_msid(pred):
    return MSID if pred in ("ms", "id") else pred

def stage1_hard(r: dict) -> str:
    """Majority vote on coarse {en, MSID, zh, ta}; lingua-high breaks ties."""
    from collections import defaultdict
    p1 = collapse_msid(r["lingua_pred"])
    p2 = collapse_msid(r["ld_pred"])
    p3 = collapse_msid(r["cld2_pred"])
    votes = defaultdict(int)
    for p in [p1, p2, p3]:
        if p != "unknown":
            votes[p] += 1
    if not votes:
        return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return p1 if p1 in winners else winners[0]

def stage2_agree(r: dict) -> str:
    """Stage 2: lingua-high + pycld2 agree; confidence tiebreaker."""
    lp = r["lingua_p"]
    cp = r["cld2_p"]
    l_pred = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    c_pred = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    if l_pred == c_pred:
        return l_pred
    l_conf = max(lp.get("ms", 0), lp.get("id", 0))
    c_conf = max(cp.get("ms", 0), cp.get("id", 0))
    return l_pred if l_conf >= c_conf else c_pred

# ── Run Stage 1 and two-stage on all cases ────────────────────────────────────
stage1_results = []
for r in results:
    s1_coarse   = stage1_hard(r)
    expected_coarse = COARSE_TARGETS.get(r["expected_iso"], "unknown")

    # Two-stage final prediction
    if s1_coarse != MSID:
        ts_pred = s1_coarse
        error_stage = None
    else:
        ts_pred = stage2_agree(r)
        error_stage = "stage2" if ts_pred != r["expected_iso"] else None

    if ts_pred != r["expected_iso"] and error_stage is None:
        error_stage = "stage1_misroute"

    stage1_results.append({
        **r,
        "s1_coarse":       s1_coarse,
        "expected_coarse": expected_coarse,
        "s1_correct":      int(s1_coarse == expected_coarse),
        "ts_pred":         ts_pred,
        "ts_correct":      int(ts_pred == r["expected_iso"]),
        "error_stage":     error_stage,
    })

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Stage 1 Routing Accuracy per True Language
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("TABLE 1 — STAGE 1 COARSE ROUTING ACCURACY")
print(SEP)
print("""
Stage 1 collapses ms/id -> MSID and runs hard vote on {en, MSID, zh, ta}.
A 'correct route' for true-MY or true-ID means Stage 1 returns MSID.
A 'misroute' means Stage 1 returned en, zh, or ta for a true-MY/ID case.
""")
print(f"  {'True lang':<10} {'n':>4}  {'Correct route':>14}  {'Misrouted':>10}  {'Misrouted to'}")
print("  " + "-" * 65)

for lbl in LANGUAGE_ORDER:
    rows = [r for r in stage1_results if r["expected_lbl"] == lbl]
    n = len(rows)
    correct = sum(r["s1_correct"] for r in rows)
    wrong   = n - correct
    # What did Stage 1 say for the wrong ones?
    wrong_dests = defaultdict(int)
    for r in rows:
        if not r["s1_correct"]:
            wrong_dests[r["s1_coarse"]] += 1
    dest_str = ", ".join(f"{k}({v})" for k, v in sorted(wrong_dests.items(), key=lambda x: -x[1]))
    print(f"  {lbl:<10} {n:>4}  {correct/n*100:>13.1f}%  {wrong:>10}  {dest_str or '--'}")

print()

# Stage 1 per-bucket for MY and ID (most affected)
print("Stage 1 routing by bucket — MY and ID:")
print(f"  {'Bucket':<13} {'n(MY)':>6} {'MY->MSID':>10} {'n(ID)':>6} {'ID->MSID':>10}")
print("  " + "-" * 50)
for bucket in BUCKET_ORDER:
    my_rows = [r for r in stage1_results if r["expected_lbl"] == "MY" and r["bucket"] == bucket]
    id_rows = [r for r in stage1_results if r["expected_lbl"] == "ID" and r["bucket"] == bucket]
    nmy = len(my_rows); nid = len(id_rows)
    if nmy == 0 and nid == 0:
        continue
    cmy = sum(r["s1_correct"] for r in my_rows) if nmy else 0
    cid = sum(r["s1_correct"] for r in id_rows) if nid else 0
    print(f"  {bucket:<13} {nmy:>6} {f'{cmy/nmy*100:.1f}%' if nmy else 'N/A':>10} "
          f"{nid:>6} {f'{cid/nid*100:.1f}%' if nid else 'N/A':>10}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Two-Stage Error Breakdown — Stage 1 misroute vs Stage 2 wrong call
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TABLE 2 — TWO-STAGE ERROR BREAKDOWN (two_stage_agree)")
print(SEP)
print("""
For each true language, errors are classified as:
  Stage 1 misroute: Stage 1 sent the case to wrong coarse class (not MSID for MY/ID)
  Stage 2 error:    Stage 1 correctly sent to MSID, but Stage 2 chose wrong ms/id
  No error:         two_stage_agree was correct
""")
print(f"  {'True lang':<10} {'n':>4}  {'Correct':>8}  {'S1 misroute':>12}  {'S2 error':>10}  {'Acc':>6}")
print("  " + "-" * 60)

for lbl in LANGUAGE_ORDER:
    rows = [r for r in stage1_results if r["expected_lbl"] == lbl]
    n = len(rows)
    correct    = sum(r["ts_correct"] for r in rows)
    s1_errors  = sum(1 for r in rows if r["error_stage"] == "stage1_misroute")
    s2_errors  = sum(1 for r in rows if r["error_stage"] == "stage2")
    print(f"  {lbl:<10} {n:>4}  {correct:>8}  {s1_errors:>12}  {s2_errors:>10}  {correct/n*100:>5.1f}%")

print()

# MY and ID bucket breakdown of error types
print("Error breakdown by bucket — MY and ID:")
for focus_lbl in ["MY", "ID"]:
    print(f"\n  {focus_lbl}:")
    print(f"  {'Bucket':<13} {'n':>4}  {'Correct':>8}  {'S1 misroute':>12}  {'S2 error':>10}  {'Acc':>6}")
    print("  " + "-" * 55)
    for bucket in BUCKET_ORDER:
        rows = [r for r in stage1_results if r["expected_lbl"] == focus_lbl and r["bucket"] == bucket]
        n = len(rows)
        if n == 0: continue
        correct   = sum(r["ts_correct"] for r in rows)
        s1_err    = sum(1 for r in rows if r["error_stage"] == "stage1_misroute")
        s2_err    = sum(1 for r in rows if r["error_stage"] == "stage2")
        print(f"  {bucket:<13} {n:>4}  {correct:>8}  {s1_err:>12}  {s2_err:>10}  {correct/n*100:>5.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 overall correctness for non-MY/ID languages (they go to correct class)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("STAGE 1 SUMMARY")
print(SEP)
n_total    = len(stage1_results)
s1_correct = sum(r["s1_correct"] for r in stage1_results)
ts_correct = sum(r["ts_correct"]  for r in stage1_results)

msid_cases = [r for r in stage1_results if r["expected_coarse"] == MSID]
msid_routed_correctly = sum(r["s1_correct"] for r in msid_cases)

print(f"""
Stage 1 coarse accuracy (all {n_total}):    {s1_correct}/{n_total} = {s1_correct/n_total*100:.1f}%
Stage 1 MSID routing (MY+ID, n={len(msid_cases)}):  {msid_routed_correctly}/{len(msid_cases)} = {msid_routed_correctly/len(msid_cases)*100:.1f}%
two_stage_agree final accuracy:        {ts_correct}/{n_total} = {ts_correct/n_total*100:.1f}%
""")

print(f"Log saved: {LOG_PATH}")
