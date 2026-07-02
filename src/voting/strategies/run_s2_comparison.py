"""
voting/strategies/run_s2_comparison.py — formerly voting/scenario2/voting_s2.py.
Scenario 2: replace langdetect with openlid-v3.
Ensemble: lingua-high + openlid-v3 + pycld2

Compares against Scenario 1 (lingua-high + langdetect + pycld2) inline.
Writes all output to results/logs_scenario2/test_case_7_enmyid/final_report_s2_N.txt

hard_vote_3 / soft_vote_3 / weighted_vote_3 and the S1_WEIGHTS/S2_WEIGHTS live in
voting.strategies.hard / .soft / .weighted (see voting/strategies/__init__.py for the
old-name -> new-name mapping).
"""

import sys
import numpy as np
from collections import defaultdict

import fasttext
from lingua import LanguageDetectorBuilder
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

from voting.core import (
    ROOT, TARGET_LANGS, LANGUAGE_ORDER, BUCKET_ORDER, LINGUA_LANGS,
    load_dataset, lingua_probs, langdetect_probs, pycld2_probs, pick_top,
)
from voting.strategies import openlid_probs
from voting.strategies.hard import hard_vote_3
from voting.strategies.soft import soft_vote_3
from voting.strategies.weighted import weighted_vote_3, S1_WEIGHTS, S2_WEIGHTS

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR    = ROOT / "results" / "logs_scenario2" / "test_case_7_enmyid"
SRC_DIR    = ROOT / "models"
DS_DIR     = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "final_report_s2.txt"

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 105

print(f"Log: {LOG_PATH}")
print("Scenario 2: lingua-high + openlid-v3 + pycld2")
print("Scenario 1: lingua-high + langdetect  + pycld2  (re-run for side-by-side comparison)\n")

# ── Load dataset ───────────────────────────────────────────────────────────────
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} test cases.\n")

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()
print("  lingua-high  : ready")

openlid_model = fasttext.load_model(str(SRC_DIR / "openlid-v3.bin"))
print("  openlid-v3   : ready (fasttext-based, FLORES-200 tags)")
print("  langdetect   : ready (lazy, seed=0)  [Scenario 1 comparison]")
print("  pycld2       : ready (compiled C++)")
print("All models loaded.\n")

# ── Run predictions for both scenarios ────────────────────────────────────────
print("Running predictions for both scenarios (this takes ~5-10 min due to langdetect)...")
s1_results = []   # lingua + langdetect + pycld2
s2_results = []   # lingua + openlid   + pycld2

for i, case in enumerate(cases):
    text = case["text"]
    lp   = lingua_probs(detector, text)
    dp   = langdetect_probs(text)
    op   = openlid_probs(openlid_model, text)
    cp   = pycld2_probs(text)

    l_pred = pick_top(lp)
    d_pred = pick_top(dp)
    o_pred = pick_top(op)
    c_pred = pick_top(cp)

    base = {
        "expected_iso": case["expected_iso"],
        "expected_lbl": case["expected_lbl"],
        "bucket":       case["bucket"],
        "lingua_pred":  l_pred,
        "lingua_p":     lp,
        "cld2_pred":    c_pred,
        "cld2_p":       cp,
    }

    s1_results.append({**base,
        "m2_pred":   d_pred,
        "m2_p":      dp,
        "hard":      hard_vote_3(l_pred, d_pred, c_pred, l_pred),
        "soft":      soft_vote_3(lp, dp, cp),
        "weighted":  weighted_vote_3(lp, dp, cp, S1_WEIGHTS),
    })
    s2_results.append({**base,
        "m2_pred":   o_pred,
        "m2_p":      op,
        "hard":      hard_vote_3(l_pred, o_pred, c_pred, l_pred),
        "soft":      soft_vote_3(lp, op, cp),
        "weighted":  weighted_vote_3(lp, op, cp, S2_WEIGHTS),
    })

print("Done.\n")

# ── Accuracy helpers ───────────────────────────────────────────────────────────
def pct(c, n):
    return f"{c/n*100:5.1f}%" if n else "  N/A "

def lang_acc(results, key, lbl=None):
    rows = [r for r in results if r["expected_lbl"] == lbl] if lbl else results
    if not rows: return 0, 0
    c = sum(1 for r in rows if r[key] == r["expected_iso"])
    return c, len(rows)

def print_table(s1, s2, s1_label, s2_label, title, lbl_filter=None):
    strats = [("lingua_pred", "lingua-high"),
              ("m2_pred",     s2_label),
              ("cld2_pred",   "pycld2"),
              ("hard",        "hard"),
              ("soft",        "soft"),
              ("weighted",    "weighted")]
    col_w = 14
    print(f"\n{title}")
    hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(
        f"{'S1 '+n if k not in ('lingua_pred','cld2_pred','hard','soft','weighted') else 'S1 '+n:>{col_w}}"
        for k, n in strats
    )
    # Simpler header
    names = ["lingua-high", f"S1:{s1_label[:8]}", "pycld2", "hard", "soft", "weighted"]
    hdr  = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{'S1 '+n:>{col_w}}" for n in names)
    hdr2 = f"  {'':5}   {'':3}   " + "   ".join(f"{'S2 '+n:>{col_w}}" for n in names)
    print(hdr)
    print(hdr2)
    print("-" * max(len(hdr), len(hdr2)))

    rows_scope = LANGUAGE_ORDER if not lbl_filter else [lbl_filter]
    for scope_lbl in rows_scope + (["ALL"] if not lbl_filter else []):
        lbl = None if scope_lbl == "ALL" else scope_lbl
        r1 = [r for r in s1 if r["expected_lbl"] == lbl] if lbl else s1
        r2 = [r for r in s2 if r["expected_lbl"] == lbl] if lbl else s2
        n = len(r1)
        if n == 0: continue
        s1_parts = [pct(sum(1 for r in r1 if r[k] == r["expected_iso"]), n) for k, _ in strats]
        s2_parts = [pct(sum(1 for r in r2 if r[k] == r["expected_iso"]), n) for k, _ in strats]
        print(f"  {scope_lbl:<5} | {n:>3} | " + " | ".join(f"{p:>{col_w}}" for p in s1_parts))
        print(f"  {'':<5}   {'':<3}   " + "   ".join(f"{p:>{col_w}}" for p in s2_parts))
        print()

# ── Print results ──────────────────────────────────────────────────────────────
print(SEP)
print("SCENARIO COMPARISON")
print(f"  S1: lingua-high + langdetect  + pycld2  (weights: {S1_WEIGHTS})")
print(f"  S2: lingua-high + openlid-v3  + pycld2  (weights: {S2_WEIGHTS})")
print(SEP)

STRATS = [("lingua_pred", "lingua-high"),
          ("m2_pred",     "model2"),
          ("cld2_pred",   "pycld2"),
          ("hard",        "hard"),
          ("soft",        "soft"),
          ("weighted",    "weighted")]
col_w = 10

# Header
h_s1 = "S1 " + " | S1 ".join(f"{'lang-d' if k=='m2_pred' else n:<{col_w}}" for k, n in STRATS)
h_s2 = "S2 " + " | S2 ".join(f"{'openlid' if k=='m2_pred' else n:<{col_w}}" for k, n in STRATS)

print(f"\n  {'LANG':<5} | {'n':>3} | {'lingua':>{col_w}} | {'S1 langdet':>{col_w}} | {'S2 openlid':>{col_w}} | {'pycld2':>{col_w}} | {'S1 hard':>{col_w}} | {'S2 hard':>{col_w}} | {'S1 soft':>{col_w}} | {'S2 soft':>{col_w}} | {'S1 wgt':>{col_w}} | {'S2 wgt':>{col_w}}")
print("  " + "-" * 120)

for scope in LANGUAGE_ORDER + ["ALL"]:
    lbl = None if scope == "ALL" else scope
    r1 = [r for r in s1_results if r["expected_lbl"] == lbl] if lbl else s1_results
    r2 = [r for r in s2_results if r["expected_lbl"] == lbl] if lbl else s2_results
    n = len(r1)
    if n == 0: continue
    lingua_acc  = sum(1 for r in r1 if r["lingua_pred"] == r["expected_iso"])
    s1_m2_acc   = sum(1 for r in r1 if r["m2_pred"]    == r["expected_iso"])
    s2_m2_acc   = sum(1 for r in r2 if r["m2_pred"]    == r["expected_iso"])
    pycld2_acc  = sum(1 for r in r1 if r["cld2_pred"]  == r["expected_iso"])
    s1_hard     = sum(1 for r in r1 if r["hard"]       == r["expected_iso"])
    s2_hard     = sum(1 for r in r2 if r["hard"]       == r["expected_iso"])
    s1_soft     = sum(1 for r in r1 if r["soft"]       == r["expected_iso"])
    s2_soft     = sum(1 for r in r2 if r["soft"]       == r["expected_iso"])
    s1_wgt      = sum(1 for r in r1 if r["weighted"]   == r["expected_iso"])
    s2_wgt      = sum(1 for r in r2 if r["weighted"]   == r["expected_iso"])

    def p(c): return f"{c/n*100:5.1f}%"
    print(f"  {scope:<5} | {n:>3} | {p(lingua_acc):>{col_w}} | {p(s1_m2_acc):>{col_w}} | {p(s2_m2_acc):>{col_w}} | {p(pycld2_acc):>{col_w}} | {p(s1_hard):>{col_w}} | {p(s2_hard):>{col_w}} | {p(s1_soft):>{col_w}} | {p(s2_soft):>{col_w}} | {p(s1_wgt):>{col_w}} | {p(s2_wgt):>{col_w}}")

print(f"""
Columns:
  lingua     = lingua-high (same for both scenarios — shared model)
  S1 langdet = langdetect individual     S2 openlid = openlid-v3 individual
  pycld2     = pycld2 (same for both — shared model)
  S1/S2 hard = hard vote (majority)
  S1/S2 soft = soft vote (avg probability)
  S1/S2 wgt  = weighted vote (AUC weights)
""")

# ── Per-bucket accuracy ────────────────────────────────────────────────────────
print(SEP)
print("PER-BUCKET ACCURACY — INDIVIDUAL MODEL COMPARISON")
print(SEP)

for bucket in BUCKET_ORDER:
    r1 = [r for r in s1_results if r["bucket"] == bucket]
    r2 = [r for r in s2_results if r["bucket"] == bucket]
    if not r1: continue
    print(f"\nBucket: {bucket}  (n={len(r1)})")
    print(f"  {'LANG':<5} | {'n':>3} | {'lingua':>8} | {'S1 langdet':>10} | {'S2 openlid':>10} | {'pycld2':>8} | {'S1 hard':>8} | {'S2 hard':>8} | {'S1 soft':>8} | {'S2 soft':>8}")
    print("  " + "-" * 90)
    for lbl in LANGUAGE_ORDER:
        b1 = [r for r in r1 if r["expected_lbl"] == lbl]
        b2 = [r for r in r2 if r["expected_lbl"] == lbl]
        n = len(b1)
        if n == 0: continue
        iso = b1[0]["expected_iso"]
        li  = sum(1 for r in b1 if r["lingua_pred"] == iso)
        d1  = sum(1 for r in b1 if r["m2_pred"]     == iso)
        d2  = sum(1 for r in b2 if r["m2_pred"]     == iso)
        cy  = sum(1 for r in b1 if r["cld2_pred"]   == iso)
        h1  = sum(1 for r in b1 if r["hard"]        == iso)
        h2  = sum(1 for r in b2 if r["hard"]        == iso)
        s1  = sum(1 for r in b1 if r["soft"]        == iso)
        s2  = sum(1 for r in b2 if r["soft"]        == iso)
        p = lambda c: f"{c/n*100:5.1f}%"
        print(f"  {lbl:<5} | {n:>3} | {p(li):>8} | {p(d1):>10} | {p(d2):>10} | {p(cy):>8} | {p(h1):>8} | {p(h2):>8} | {p(s1):>8} | {p(s2):>8}")

# ── McNemar tests: S1 vs S2 on each voting strategy ──────────────────────────
print(f"\n{SEP}")
print("McNEMAR'S TESTS: SCENARIO 1 vs SCENARIO 2 (same test cases, paired)")
print(SEP)
print("""
Testing whether replacing langdetect with openlid-v3 significantly changes
voting accuracy on the languages most likely to be affected: MY, ID.
Library: statsmodels.stats.contingency_tables.mcnemar
""")

def run_mcnemar(arr_a, arr_b):
    b = int(np.sum((arr_a == 1) & (arr_b == 0)))
    c = int(np.sum((arr_a == 0) & (arr_b == 1)))
    n11 = int(np.sum((arr_a == 1) & (arr_b == 1)))
    n00 = int(np.sum((arr_a == 0) & (arr_b == 0)))
    table = np.array([[n11, b], [c, n00]])
    exact = (b + c) < 25
    res = sm_mcnemar(table, exact=exact, correction=not exact)
    return res.pvalue, b, c, "exact" if exact else "chi2"

def sig(p):
    if p < 0.0014: return "***"
    if p < 0.01:   return " **"
    if p < 0.05:   return "  *"
    return "   "

COMPARE_STRATS = [("hard", "hard"), ("soft", "soft"), ("weighted", "weighted")]

for focus_lbl in LANGUAGE_ORDER + ["ALL"]:
    lbl = None if focus_lbl == "ALL" else focus_lbl
    r1 = [r for r in s1_results if r["expected_lbl"] == lbl] if lbl else s1_results
    r2 = [r for r in s2_results if r["expected_lbl"] == lbl] if lbl else s2_results
    n = len(r1)
    print(f"-- {focus_lbl}  (n={n}) " + "-" * 55)
    print(f"  {'Strategy':<12} {'S1 acc':>8} {'S2 acc':>8}  {'p-value':>10}  sig   b     c   method")
    for k, name in COMPARE_STRATS:
        s1_arr = np.array([int(r[k] == r["expected_iso"]) for r in r1])
        s2_arr = np.array([int(r[k] == r["expected_iso"]) for r in r2])
        s1_a = s1_arr.mean() * 100
        s2_a = s2_arr.mean() * 100
        p, b, c, method = run_mcnemar(s1_arr, s2_arr)
        diff = s2_a - s1_a
        arrow = "+" if diff >= 0 else ""
        print(f"  {name:<12} {s1_a:>7.1f}%  {s2_a:>7.1f}%  ({arrow}{diff:.1f}pp)  {p:>8.4f}  {sig(p)}  {b:>3}  {c:>3}  {method}")
    print()

# ── Summary verdict ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SCENARIO 2 vs SCENARIO 1 — SUMMARY VERDICT")
print(SEP)

for lbl_scope in LANGUAGE_ORDER + ["ALL"]:
    lbl = None if lbl_scope == "ALL" else lbl_scope
    r1 = [r for r in s1_results if r["expected_lbl"] == lbl] if lbl else s1_results
    r2 = [r for r in s2_results if r["expected_lbl"] == lbl] if lbl else s2_results
    n = len(r1)
    s1_best = max(sum(1 for r in r1 if r[k] == r["expected_iso"]) for k, _ in STRATS)
    s2_best = max(sum(1 for r in r2 if r[k] == r["expected_iso"]) for k, _ in STRATS)
    delta = (s2_best - s1_best) / n * 100
    arrow = "+" if delta >= 0 else ""
    verdict = "IMPROVED" if delta > 0.5 else ("DEGRADED" if delta < -0.5 else "SAME")
    print(f"  {lbl_scope:<5}  S1 best: {s1_best/n*100:5.1f}%  S2 best: {s2_best/n*100:5.1f}%  delta: {arrow}{delta:.1f}pp  -> {verdict}")

print(f"\nWeights used:")
print(f"  S1: lingua={S1_WEIGHTS['lingua']} | langdetect={S1_WEIGHTS['model2']} | pycld2={S1_WEIGHTS['pycld2']}")
print(f"  S2: lingua={S2_WEIGHTS['lingua']} | openlid-v3={S2_WEIGHTS['model2']} | pycld2={S2_WEIGHTS['pycld2']}")
print(f"\nLog saved: {LOG_PATH}")
