"""
voting/voting_ablation.py — Leave-one-model-out ablation study.

Objective 4 (per the task spec):
  Re-run hard/soft/weighted voting with each of the 3 models removed in turn.
  9 ablation configurations + 3-model baseline = 12 total.

Hypothesis: removing langdetect should recover MY accuracy without harming ID.

Writes: log/log_voting_ablation_N.txt
"""

import sys
import numpy as np
from collections import defaultdict

from voting.core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, BUCKET_ORDER, DEFAULT_WEIGHTS,
    TARGET_LANGS, next_path, load_dataset, load_lingua, run_predictions,
    hard_vote, soft_vote, weighted_vote, pct,
    lingua_probs, langdetect_probs, pycld2_probs,
)

LOG_PATH = next_path(LOG_DIR, "log_voting_ablation", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 100

print(f"Log: {LOG_PATH}")
print("Loading lingua-high detector...")
detector = load_lingua()
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} test cases. Running predictions...")
results = run_predictions(cases, detector)
print("Predictions complete.\n")

# ── 2-model voting variants ────────────────────────────────────────────────────

def hard_vote_2(pred_a, conf_a, pred_b, conf_b):
    """Hard vote between 2 models. On disagreement, higher confidence wins."""
    if pred_a == pred_b:
        return pred_a
    return pred_a if conf_a >= conf_b else pred_b

def soft_vote_2(probs_a, probs_b):
    """Average 2-model probability vectors; pick max."""
    combined = {l: (probs_a.get(l, 0.0) + probs_b.get(l, 0.0)) / 2
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)

def weighted_vote_2(probs_a, probs_b, w_a, w_b):
    """Weighted 2-model probability sum; pick max."""
    combined = {l: probs_a.get(l, 0.0) * w_a + probs_b.get(l, 0.0) * w_b
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)

# ── Compute ablation predictions ───────────────────────────────────────────────
ablation_preds = []
for r in results:
    lp = r["lingua_p"]
    dp = r["ld_p"]
    cp = r["cld2_p"]
    l_pred, l_conf = r["lingua_pred"], r["lingua_conf"]
    d_pred, d_conf = r["ld_pred"],     r["ld_conf"]
    c_pred, c_conf = r["cld2_pred"],   r["cld2_conf"]
    wl = DEFAULT_WEIGHTS["lingua"]
    wd = DEFAULT_WEIGHTS["langdetect"]
    wp = DEFAULT_WEIGHTS["pycld2"]

    ablation_preds.append({
        # 3-model baseline
        "base_hard":     r["hard"],
        "base_soft":     r["soft"],
        "base_weighted": r["weighted"],
        # No langdetect (lingua + pycld2)
        "no_ld_hard":     hard_vote_2(l_pred, l_conf, c_pred, c_conf),
        "no_ld_soft":     soft_vote_2(lp, cp),
        "no_ld_weighted": weighted_vote_2(lp, cp, wl, wp),
        # No lingua-high (langdetect + pycld2)
        "no_li_hard":     hard_vote_2(d_pred, d_conf, c_pred, c_conf),
        "no_li_soft":     soft_vote_2(dp, cp),
        "no_li_weighted": weighted_vote_2(dp, cp, wd, wp),
        # No pycld2 (lingua + langdetect)
        "no_py_hard":     hard_vote_2(l_pred, l_conf, d_pred, d_conf),
        "no_py_soft":     soft_vote_2(lp, dp),
        "no_py_weighted": weighted_vote_2(lp, dp, wl, wd),
        # Expected
        "expected_iso": r["expected_iso"],
        "expected_lbl": r["expected_lbl"],
        "bucket":       r["bucket"],
    })

# ── Print per-language accuracy table ─────────────────────────────────────────
ABLATION_CONFIGS = [
    # (key,               display_label)
    ("base_hard",         "3-model hard"),
    ("base_soft",         "3-model soft"),
    ("base_weighted",     "3-model weighted"),
    ("no_ld_hard",        "no-LD hard"),
    ("no_ld_soft",        "no-LD soft"),
    ("no_ld_weighted",    "no-LD weighted"),
    ("no_li_hard",        "no-LI hard"),
    ("no_li_soft",        "no-LI soft"),
    ("no_li_weighted",    "no-LI weighted"),
    ("no_py_hard",        "no-PY hard"),
    ("no_py_soft",        "no-PY soft"),
    ("no_py_weighted",    "no-PY weighted"),
]

# Also include lingua-high individual as reference
INDIVIDUAL_REF = [
    ("lingua_pred", "lingua-high"),
    ("ld_pred",     "langdetect"),
    ("cld2_pred",   "pycld2"),
]

print(SEP)
print("ABLATION — INDIVIDUAL MODEL REFERENCE")
print(SEP)
col_w = 14
ref_names = [n for _, n in INDIVIDUAL_REF]
hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in ref_names)
print(hdr)
print("-" * len(hdr))
for lbl in LANGUAGE_ORDER:
    rows = [r for r in results if r["expected_lbl"] == lbl]
    n = len(rows)
    parts = [f"{pct(sum(1 for r in rows if r[k] == r['expected_iso']), n):>{col_w}}"
             for k, _ in INDIVIDUAL_REF]
    print(f"  {lbl:<5} | {n:>3} | " + " | ".join(parts))
print("-" * len(hdr))
n_all = len(results)
parts = [f"{pct(sum(1 for r in results if r[k] == r['expected_iso']), n_all):>{col_w}}"
         for k, _ in INDIVIDUAL_REF]
print(f"  {'ALL':<5} | {n_all:>3} | " + " | ".join(parts))

print(f"\n{SEP}")
print("ABLATION — ALL 12 CONFIGURATIONS (3-model baseline + 9 ablations)")
print(SEP)
print("Columns: 3-model baseline (3 strategies) | no-langdetect | no-lingua-high | no-pycld2")
print("LD = langdetect  LI = lingua-high  PY = pycld2\n")

col_w = 16
config_names = [n for _, n in ABLATION_CONFIGS]
hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in config_names)
print(hdr)
print("-" * len(hdr))

for lbl in LANGUAGE_ORDER:
    rows = [r for r in ablation_preds if r["expected_lbl"] == lbl]
    n = len(rows)
    parts = []
    for k, _ in ABLATION_CONFIGS:
        c = sum(1 for r in rows if r[k] == r["expected_iso"])
        parts.append(f"{pct(c, n):>{col_w}}")
    print(f"  {lbl:<5} | {n:>3} | " + " | ".join(parts))

print("-" * len(hdr))
n_all = len(ablation_preds)
parts = []
for k, _ in ABLATION_CONFIGS:
    c = sum(1 for r in ablation_preds if r[k] == r["expected_iso"])
    parts.append(f"{pct(c, n_all):>{col_w}}")
print(f"  {'ALL':<5} | {n_all:>3} | " + " | ".join(parts))

# ── Per-bucket accuracy for the critical MY row ───────────────────────────────
print(f"\n{SEP}")
print("ABLATION — MY ACCURACY BY BUCKET (key focus: ms/id confusion)")
print(SEP)

FOCUS_CONFIGS = [
    ("lingua_pred",   "lingua-high (ref)"),
    ("base_hard",     "3-model hard"),
    ("base_soft",     "3-model soft"),
    ("no_ld_hard",    "no-LD hard"),
    ("no_ld_soft",    "no-LD soft"),
    ("no_ld_weighted","no-LD wgtd"),
]
col_w = 16
fname = [n for _, n in FOCUS_CONFIGS]
hdr2 = f"  {'BUCKET':<12} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in fname)
print(hdr2)
print("-" * len(hdr2))

def get_val(r, k):
    if k in ("lingua_pred", "ld_pred", "cld2_pred"):
        return r.get(k, "")
    return r.get(k, "")

for bucket in BUCKET_ORDER:
    rows_a = [r for r in ablation_preds if r["expected_lbl"] == "MY" and r["bucket"] == bucket]
    rows_r = [r for r in results        if r["expected_lbl"] == "MY" and r["bucket"] == bucket]
    n = len(rows_a)
    if n == 0:
        continue
    parts = []
    for k, _ in FOCUS_CONFIGS:
        if k in ("lingua_pred", "ld_pred", "cld2_pred"):
            c = sum(1 for r in rows_r if r[k] == r["expected_iso"])
            parts.append(f"{pct(c, n):>{col_w}}")
        else:
            c = sum(1 for r in rows_a if r[k] == r["expected_iso"])
            parts.append(f"{pct(c, n):>{col_w}}")
    print(f"  {bucket:<12} | {n:>3} | " + " | ".join(parts))

print("-" * len(hdr2))
rows_a = [r for r in ablation_preds if r["expected_lbl"] == "MY"]
rows_r = [r for r in results        if r["expected_lbl"] == "MY"]
n = len(rows_a)
parts = []
for k, _ in FOCUS_CONFIGS:
    if k in ("lingua_pred", "ld_pred", "cld2_pred"):
        c = sum(1 for r in rows_r if r[k] == r["expected_iso"])
    else:
        c = sum(1 for r in rows_a if r[k] == r["expected_iso"])
    parts.append(f"{pct(c, n):>{col_w}}")
print(f"  {'ALL':<12} | {n:>3} | " + " | ".join(parts))

# ── Hypothesis verdict ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("HYPOTHESIS VERDICT")
print(SEP)

# Compute key numbers for the verdict
my_lingua  = sum(1 for r in results if r["expected_lbl"] == "MY" and r["lingua_pred"] == r["expected_iso"])
my_hard    = sum(1 for r in ablation_preds if r["expected_lbl"] == "MY" and r["base_hard"] == r["expected_iso"])
my_no_ld_h = sum(1 for r in ablation_preds if r["expected_lbl"] == "MY" and r["no_ld_hard"] == r["expected_iso"])
my_no_ld_s = sum(1 for r in ablation_preds if r["expected_lbl"] == "MY" and r["no_ld_soft"] == r["expected_iso"])
my_no_ld_w = sum(1 for r in ablation_preds if r["expected_lbl"] == "MY" and r["no_ld_weighted"] == r["expected_iso"])

id_lingua  = sum(1 for r in results if r["expected_lbl"] == "ID" and r["lingua_pred"] == r["expected_iso"])
id_hard    = sum(1 for r in ablation_preds if r["expected_lbl"] == "ID" and r["base_hard"] == r["expected_iso"])
id_no_ld_h = sum(1 for r in ablation_preds if r["expected_lbl"] == "ID" and r["no_ld_hard"] == r["expected_iso"])
id_no_ld_s = sum(1 for r in ablation_preds if r["expected_lbl"] == "ID" and r["no_ld_soft"] == r["expected_iso"])
n_my = sum(1 for r in results if r["expected_lbl"] == "MY")
n_id = sum(1 for r in results if r["expected_lbl"] == "ID")

print(f"""
Hypothesis: Removing langdetect should recover MY accuracy without harming ID.

MY ACCURACY
  lingua-high individual:    {my_lingua}/{n_my} = {my_lingua/n_my*100:.1f}%
  3-model hard vote:         {my_hard}/{n_my}  = {my_hard/n_my*100:.1f}%
  no-langdetect hard:        {my_no_ld_h}/{n_my} = {my_no_ld_h/n_my*100:.1f}%
  no-langdetect soft:        {my_no_ld_s}/{n_my} = {my_no_ld_s/n_my*100:.1f}%
  no-langdetect weighted:    {my_no_ld_w}/{n_my} = {my_no_ld_w/n_my*100:.1f}%

ID ACCURACY
  lingua-high individual:    {id_lingua}/{n_id} = {id_lingua/n_id*100:.1f}%
  3-model hard vote:         {id_hard}/{n_id}  = {id_hard/n_id*100:.1f}%
  no-langdetect hard:        {id_no_ld_h}/{n_id} = {id_no_ld_h/n_id*100:.1f}%
  no-langdetect soft:        {id_no_ld_s}/{n_id} = {id_no_ld_s/n_id*100:.1f}%

VERDICT:
""")

my_recovered = my_no_ld_h >= my_lingua
id_preserved = id_no_ld_h >= id_lingua - 2   # allow ≤2 cases noise

if my_recovered and id_preserved:
    print("  SUPPORTED — Removing langdetect recovers MY accuracy to at least lingua-high")
    print("  individual level, while ID accuracy is preserved or improved.")
elif my_recovered and not id_preserved:
    print("  PARTIALLY SUPPORTED — MY accuracy recovers but ID accuracy drops.")
    print("  The two-stage approach (Obj 6) is needed to maintain both.")
else:
    print("  NOT SUPPORTED in the simple ablation. See two-stage voting (Obj 6)")
    print("  for a more targeted fix that exploits per-class model strengths.")

print(f"\nLog saved: {LOG_PATH}")
