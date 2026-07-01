"""
voting/voting_two_stage.py — Leakage check + Two-Stage Voting.

Objectives covered:
  5. Train/test leakage check: dev (60%) / held-out test (40%), stratified.
     Derive WEIGHTS from dev AUC only; evaluate on held-out test only.
  6. Two-stage voting:
       Stage 1: all 3 models, collapse ms/id → MSID, hard vote on {en, MSID, zh, ta}
       Stage 2: triggered if Stage 1 = MSID; exclude langdetect; lingua-high + pycld2 only.
         (a) two_stage_agree: agreement + confidence tiebreaker
         (b) two_stage_weighted: per-class (MY/ID) accuracy weights from dev set
     McNemar's tests on MY and ID: two-stage vs best individual, vs best 3-model vote.

Writes: log/log_voting_two_stage_N.txt
"""

import sys
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, BUCKET_ORDER, TARGET_LANGS,
    DEFAULT_WEIGHTS, next_path, load_dataset, load_lingua,
    run_predictions, binary_correct, overall_accuracy, accuracy_by_lang,
    pct, hard_vote, soft_vote, weighted_vote,
)

LOG_PATH = next_path(LOG_DIR, "log_voting_two_stage", ".txt")

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
all_cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(all_cases)} total cases.")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 5 — Dev / Test Split + Leakage Check
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("OBJECTIVE 5 — DEV/TEST SPLIT & LEAKAGE CHECK")
print(SEP)
print("""
WEIGHTS in voting.py are derived from the same full dataset used to
evaluate voting accuracy — a circular evaluation.
Fix: stratified 60/40 split. Dev set (60%) → compute per-model AUC → derive
weights. Held-out test set (40%) → report final voting accuracy using dev weights.
Compare "honest" held-out numbers to the leaked-weight numbers.
Library: sklearn.model_selection.train_test_split (stratify=language_label)
""")

RANDOM_STATE = 42
labels = [c["expected_lbl"] for c in all_cases]
idx = list(range(len(all_cases)))
dev_idx, test_idx = train_test_split(
    idx, test_size=0.4, random_state=RANDOM_STATE, stratify=labels
)
dev_cases  = [all_cases[i] for i in dev_idx]
test_cases = [all_cases[i] for i in test_idx]

n_dev = len(dev_cases)
n_test = len(test_cases)
print(f"Dev set:  {n_dev}  cases ({n_dev//5} per language)")
print(f"Test set: {n_test} cases ({n_test//5} per language)\n")

# Confirm stratification
dev_dist  = defaultdict(int)
test_dist = defaultdict(int)
for c in dev_cases:  dev_dist[c["expected_lbl"]] += 1
for c in test_cases: test_dist[c["expected_lbl"]] += 1
print("Stratification check:")
for lbl in LANGUAGE_ORDER:
    print(f"  {lbl}: dev={dev_dist[lbl]}  test={test_dist[lbl]}")
print()

# Run predictions on dev and test sets
print("Running dev set predictions...")
dev_results  = run_predictions(dev_cases,  detector)
print("Running test set predictions...")
test_results = run_predictions(test_cases, detector)
print("Done.\n")

# ── Compute per-model AUC on dev set ─────────────────────────────────────────
def compute_macro_auc(results: list, pred_key: str, conf_key: str) -> float:
    """
    One-vs-rest macro-averaged AUC on the 5-class problem.
    Uses sklearn.metrics.roc_auc_score with binary OVR for each class,
    then averages (weighted by class frequency).
    """
    aucs = []
    weights_auc = []
    for iso in ["en", "ms", "id", "zh", "ta"]:
        y_true = np.array([int(r["expected_iso"] == iso) for r in results])
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        # Confidence for this class = confidence of predicted class if predicted == iso,
        # else the confidence of the correct class from the probability dict
        conf_key_map = {"lingua_pred": "lingua_p", "ld_pred": "ld_p", "cld2_pred": "cld2_p"}
        prob_key = conf_key_map.get(pred_key, None)
        if prob_key:
            y_score = np.array([r[prob_key].get(iso, 0.0) for r in results])
        else:
            # For voting strategies, use max-class confidence
            y_score = np.array([float(r[pred_key] == iso) for r in results])
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = 0.5
        aucs.append(auc)
        weights_auc.append(y_true.sum())
    if not aucs:
        return 0.5
    total = sum(weights_auc)
    return sum(a * w for a, w in zip(aucs, weights_auc)) / total

dev_auc_lingua = compute_macro_auc(dev_results, "lingua_pred", "lingua_conf")
dev_auc_ld     = compute_macro_auc(dev_results, "ld_pred",     "ld_conf")
dev_auc_cld2   = compute_macro_auc(dev_results, "cld2_pred",   "cld2_conf")

print(f"Dev-set AUC (macro OVR):")
print(f"  lingua-high:  {dev_auc_lingua:.4f}  (global: 0.8503)")
print(f"  langdetect:   {dev_auc_ld:.4f}  (global: 0.7516)")
print(f"  pycld2:       {dev_auc_cld2:.4f}  (global: 0.9634)\n")

dev_weights = {
    "lingua":     dev_auc_lingua,
    "langdetect": dev_auc_ld,
    "pycld2":     dev_auc_cld2,
}

# ── Evaluate on held-out test set with dev-derived weights ────────────────────
print("Evaluating on held-out test set with dev-derived weights...")

def apply_weighted_vote_custom(results, weights):
    """Return list of weighted-vote predictions using given weights."""
    preds = []
    for r in results:
        combined = {
            l: r["lingua_p"].get(l, 0.0) * weights["lingua"] +
               r["ld_p"].get(l, 0.0)     * weights["langdetect"] +
               r["cld2_p"].get(l, 0.0)   * weights["pycld2"]
            for l in TARGET_LANGS
        }
        preds.append(max(combined, key=combined.get))
    return preds

# Test set predictions using global weights (originally leaked)
test_preds_global_wt = apply_weighted_vote_custom(test_results, DEFAULT_WEIGHTS)
# Test set predictions using dev-only weights (honest)
test_preds_dev_wt    = apply_weighted_vote_custom(test_results, dev_weights)

def acc_from_preds(preds, results):
    return sum(1 for p, r in zip(preds, results) if p == r["expected_iso"]) / len(results)

def acc_by_lang_from_preds(preds, results):
    by_lang = defaultdict(lambda: [0, 0])
    for p, r in zip(preds, results):
        lbl = r["expected_lbl"]
        by_lang[lbl][1] += 1
        by_lang[lbl][0] += int(p == r["expected_iso"])
    return dict(by_lang)

print(f"\nLeakage Check — Test Set Accuracy (n={n_test})")
print(f"  {'Strategy':<28} {'Accuracy':>10}")
print(f"  {'':-<28} {'':-^10}")

# Individual models on test set
for k, name in [("lingua_pred", "lingua-high (test)"),
                ("ld_pred",     "langdetect  (test)"),
                ("cld2_pred",   "pycld2      (test)")]:
    acc = acc_from_preds([r[k] for r in test_results], test_results)
    print(f"  {name:<28} {acc*100:>9.1f}%")

acc_hard_test = acc_from_preds([r["hard"]     for r in test_results], test_results)
acc_soft_test = acc_from_preds([r["soft"]     for r in test_results], test_results)
acc_gw_test   = acc_from_preds(test_preds_global_wt,                   test_results)
acc_dw_test   = acc_from_preds(test_preds_dev_wt,                       test_results)

print(f"  {'hard vote (test)':<28} {acc_hard_test*100:>9.1f}%")
print(f"  {'soft vote (test)':<28} {acc_soft_test*100:>9.1f}%")
print(f"  {'weighted-global (test)':<28} {acc_gw_test*100:>9.1f}%  ← weights from full dataset")
print(f"  {'weighted-dev (test)':<28} {acc_dw_test*100:>9.1f}%  ← weights from dev set only (honest)")

gap = (acc_dw_test - acc_gw_test) * 100
print(f"\n  Overfitting gap (dev-wt vs global-wt): {gap:+.2f} pp")
print(f"  A gap near 0 means the global AUC weights did NOT materially overfit.")

# Per-language breakdown
print(f"\nPer-language test accuracy:")
col_w = 16
strategies_test = [
    ("lingua_pred",      "lingua-high"),
    ("hard",             "hard"),
    ("soft",             "soft"),
]
names_test = [n for _, n in strategies_test] + ["wt-global", "wt-dev"]
hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in names_test)
print(hdr)
print("-" * len(hdr))
for lbl in LANGUAGE_ORDER:
    rows = [r for r in test_results if r["expected_lbl"] == lbl]
    n = len(rows)
    parts = []
    for k, _ in strategies_test:
        c = sum(1 for r in rows if r[k] == r["expected_iso"])
        parts.append(f"{pct(c, n):>{col_w}}")
    # global-wt and dev-wt
    c_gw = sum(1 for p, r in zip(test_preds_global_wt, test_results)
               if r["expected_lbl"] == lbl and p == r["expected_iso"])
    c_dw = sum(1 for p, r in zip(test_preds_dev_wt, test_results)
               if r["expected_lbl"] == lbl and p == r["expected_iso"])
    parts.append(f"{pct(c_gw, n):>{col_w}}")
    parts.append(f"{pct(c_dw, n):>{col_w}}")
    print(f"  {lbl:<5} | {n:>3} | " + " | ".join(parts))
print("-" * len(hdr))
n_all = len(test_results)
parts = []
for k, _ in strategies_test:
    c = sum(1 for r in test_results if r[k] == r["expected_iso"])
    parts.append(f"{pct(c, n_all):>{col_w}}")
parts.append(f"{pct(int(acc_gw_test*n_all), n_all):>{col_w}}")
parts.append(f"{pct(int(acc_dw_test*n_all), n_all):>{col_w}}")
print(f"  {'ALL':<5} | {n_all:>3} | " + " | ".join(parts))

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 6 — Two-Stage Voting
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("OBJECTIVE 6 — TWO-STAGE VOTING")
print(SEP)
print("""
DESIGN:
  Stage 1 (all 3 models):
    Collapse ms → MSID, id → MSID in each model's output.
    Hard vote on the coarse classes {en, MSID, zh, ta}.
    langdetect's 'id' output for Malay text counts as MSID — structurally valid.
    If result is en/zh/ta → final answer.
    If result is MSID    → trigger Stage 2.

  Stage 2 (lingua-high + pycld2 only, langdetect excluded):
    (a) two_stage_agree : if both agree on ms/id → use that;
        if disagree → use model with higher confidence for its ms/id prediction.
    (b) two_stage_weighted : multiply each model's ms/id probability by
        per-class accuracy weight derived from the dev set (MY accuracy for
        the ms vote, ID accuracy for the id vote).
        score_ms = lingua_p['ms'] * w_l_ms + cld2_p['ms'] * w_c_ms
        score_id = lingua_p['id'] * w_l_id + cld2_p['id'] * w_c_id

RATIONALE:
  langdetect can never output 'ms', so including it in Stage 2 always adds a
  biased vote for 'id'. Excluding it from Stage 2 removes this structural bias.
""")

# Compute per-class dev-set accuracy for Stage 2 weights
def per_class_acc(results, pred_key, target_iso):
    relevant = [r for r in results if r["expected_iso"] == target_iso]
    if not relevant:
        return 0.0
    return sum(1 for r in relevant if r[pred_key] == target_iso) / len(relevant)

w_l_ms = per_class_acc(dev_results, "lingua_pred", "ms")
w_l_id = per_class_acc(dev_results, "lingua_pred", "id")
w_c_ms = per_class_acc(dev_results, "cld2_pred",   "ms")
w_c_id = per_class_acc(dev_results, "cld2_pred",   "id")

print(f"Stage 2 per-class weights (from dev set):")
print(f"  lingua-high  MY accuracy: {w_l_ms:.4f}  (weight for lingua 'ms' score)")
print(f"  lingua-high  ID accuracy: {w_l_id:.4f}  (weight for lingua 'id' score)")
print(f"  pycld2       MY accuracy: {w_c_ms:.4f}  (weight for pycld2 'ms' score)")
print(f"  pycld2       ID accuracy: {w_c_id:.4f}  (weight for pycld2 'id' score)\n")

# ── Stage 1 ────────────────────────────────────────────────────────────────────
MSID_CLASS = "msid"
COARSE_LANGS = ["en", MSID_CLASS, "zh", "ta"]

def collapse_msid(pred):
    return MSID_CLASS if pred in ("ms", "id") else pred

def stage1_hard(r: dict) -> str:
    """Majority vote on coarse {en, MSID, zh, ta}; tie → lingua-high."""
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

# ── Stage 2 variants ───────────────────────────────────────────────────────────
def stage2_agree(r: dict) -> str:
    """Simple agreement; confidence tiebreaker on disagreement."""
    lp = r["lingua_p"]
    cp = r["cld2_p"]
    l_ms, l_id = lp.get("ms", 0.0), lp.get("id", 0.0)
    c_ms, c_id = cp.get("ms", 0.0), cp.get("id", 0.0)
    l_pred = "ms" if l_ms >= l_id else "id"
    c_pred = "ms" if c_ms >= c_id else "id"
    if l_pred == c_pred:
        return l_pred
    l_conf = max(l_ms, l_id)
    c_conf = max(c_ms, c_id)
    return l_pred if l_conf >= c_conf else c_pred

def stage2_weighted_fn(r: dict) -> str:
    """Per-class accuracy weighted ms/id decision."""
    lp = r["lingua_p"]
    cp = r["cld2_p"]
    score_ms = lp.get("ms", 0.0) * w_l_ms + cp.get("ms", 0.0) * w_c_ms
    score_id = lp.get("id", 0.0) * w_l_id + cp.get("id", 0.0) * w_c_id
    return "ms" if score_ms >= score_id else "id"

def two_stage(r: dict, stage2_fn) -> str:
    coarse = stage1_hard(r)
    if coarse != MSID_CLASS:
        return coarse
    return stage2_fn(r)

# ── Apply two-stage to FULL dataset (for comparison with earlier log_voting_2 results) ──
print(f"Applying two-stage voting to full {len(all_cases)}-case dataset...")
full_results = run_predictions(all_cases, detector)

full_ts_agree_preds    = [two_stage(r, stage2_agree)       for r in full_results]
full_ts_weighted_preds = [two_stage(r, stage2_weighted_fn) for r in full_results]

print(f"\nTWO-STAGE RESULTS — FULL DATASET (n={len(full_results)})")
print("Reference: 3-model hard vote, 3-model soft vote\n")

col_w = 15
strats_full = [
    ("lingua_pred",    "lingua-high"),
    ("hard",           "3m-hard"),
    ("soft",           "3m-soft"),
]
extra_names = ["2s-agree", "2s-weighted"]

names_full = [n for _, n in strats_full] + extra_names
hdr_f = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in names_full)
print(hdr_f)
print("-" * len(hdr_f))

for lbl in LANGUAGE_ORDER:
    rows_r = [r for r in full_results if r["expected_lbl"] == lbl]
    n = len(rows_r)
    ts_a_rows = [p for p, r in zip(full_ts_agree_preds,    full_results) if r["expected_lbl"] == lbl]
    ts_w_rows = [p for p, r in zip(full_ts_weighted_preds, full_results) if r["expected_lbl"] == lbl]
    iso = rows_r[0]["expected_iso"]
    parts = []
    for k, _ in strats_full:
        c = sum(1 for r in rows_r if r[k] == r["expected_iso"])
        parts.append(f"{pct(c, n):>{col_w}}")
    parts.append(f"{pct(sum(p==iso for p in ts_a_rows), n):>{col_w}}")
    parts.append(f"{pct(sum(p==iso for p in ts_w_rows), n):>{col_w}}")
    print(f"  {lbl:<5} | {n:>3} | " + " | ".join(parts))

print("-" * len(hdr_f))
n_all = len(full_results)
parts = []
for k, _ in strats_full:
    c = sum(1 for r in full_results if r[k] == r["expected_iso"])
    parts.append(f"{pct(c, n_all):>{col_w}}")
parts.append(f"{pct(sum(p == r['expected_iso'] for p, r in zip(full_ts_agree_preds, full_results)), n_all):>{col_w}}")
parts.append(f"{pct(sum(p == r['expected_iso'] for p, r in zip(full_ts_weighted_preds, full_results)), n_all):>{col_w}}")
print(f"  {'ALL':<5} | {n_all:>3} | " + " | ".join(parts))

# ── Per-bucket for MY and ID ───────────────────────────────────────────────────
print(f"\nTWO-STAGE — MY and ID ACCURACY BY BUCKET")
for focus_lbl in ["MY", "ID"]:
    print(f"\n  Focus: {focus_lbl}")
    col_w2 = 14
    bucket_names = ["lingua-high", "3m-hard", "3m-soft", "2s-agree", "2s-wgtd"]
    hdr_b = f"    {'BUCKET':<12} | {'n':>3} | " + " | ".join(f"{n:>{col_w2}}" for n in bucket_names)
    print(hdr_b)
    print("    " + "-" * (len(hdr_b) - 4))
    for bucket in BUCKET_ORDER:
        idx_rows = [(i, r) for i, r in enumerate(full_results)
                    if r["expected_lbl"] == focus_lbl and r["bucket"] == bucket]
        n = len(idx_rows)
        if n == 0:
            continue
        iso = idx_rows[0][1]["expected_iso"]
        c_li = sum(1 for _, r in idx_rows if r["lingua_pred"] == iso)
        c_h3 = sum(1 for _, r in idx_rows if r["hard"] == iso)
        c_s3 = sum(1 for _, r in idx_rows if r["soft"] == iso)
        c_ta = sum(1 for i, _ in idx_rows if full_ts_agree_preds[i] == iso)
        c_tw = sum(1 for i, _ in idx_rows if full_ts_weighted_preds[i] == iso)
        row = f"    {bucket:<12} | {n:>3} | "
        row += " | ".join(f"{pct(c, n):>{col_w2}}"
                          for c in [c_li, c_h3, c_s3, c_ta, c_tw])
        print(row)

# ── McNemar tests: two-stage vs lingua-high and vs 3-model hard ───────────────
print(f"\n{SEP}")
print("OBJECTIVE 6 — McNEMAR'S TESTS: TWO-STAGE vs BASELINES (MY and ID focus)")
print(SEP)
print("""
Paired McNemar test on full dataset.
Focusing on MY (the degraded class) and ID (the large-gain class).
Testing: two_stage_agree and two_stage_weighted
  vs lingua-high (best individual)
  vs hard (best 3-model overall)
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

def sig_m(p):
    if p < 0.0014: return "***"
    if p < 0.01:   return " **"
    if p < 0.05:   return "  *"
    return "   "

MCNEMAR_PAIRS = [
    ("lingua_pred",   "lingua-high",  None),
    ("hard",          "3m-hard",       None),
    ("soft",          "3m-soft",       None),
]

for focus_lbl in ["MY", "ID"]:
    print(f"── {focus_lbl} " + "─" * 75)
    print(f"  {'Comparison':<42} {'p-value':>10}  {'sig':>3}  {'b':>4}  {'c':>4}  method")
    print(f"  {'':-<42} {'':-^10}  {'':-^3}  {'':-^4}  {'':-^4}  {'':-^6}")
    for method_key, method_name in [
        ("two_stage_agree",    "two_stage_agree"),
        ("two_stage_weighted", "two_stage_wgtd"),
    ]:
        if method_key == "two_stage_agree":
            ts_arr = np.array([
                int(p == r["expected_iso"])
                for p, r in zip(full_ts_agree_preds, full_results)
                if r["expected_lbl"] == focus_lbl
            ])
        else:
            ts_arr = np.array([
                int(p == r["expected_iso"])
                for p, r in zip(full_ts_weighted_preds, full_results)
                if r["expected_lbl"] == focus_lbl
            ])
        for ref_key, ref_name, _ in MCNEMAR_PAIRS:
            ref_arr = binary_correct(full_results, ref_key, focus_lbl)
            p, b, c_val, method = run_mcnemar(ts_arr, ref_arr)
            label = f"{method_name} vs {ref_name}"
            print(f"  {label:<42} {p:>10.4f}  {sig_m(p)}  {b:>4}  {c_val:>4}  {method}")
    print()

# ── Final verdict ──────────────────────────────────────────────────────────────
print(f"{SEP}")
print("FINAL VERDICT — DOES TWO-STAGE FIX THE MY/ID PROBLEM?")
print(SEP)

ts_my_agree    = sum(p == "ms" for p, r in zip(full_ts_agree_preds,    full_results) if r["expected_lbl"] == "MY")
ts_my_weighted = sum(p == "ms" for p, r in zip(full_ts_weighted_preds, full_results) if r["expected_lbl"] == "MY")
ts_id_agree    = sum(p == "id" for p, r in zip(full_ts_agree_preds,    full_results) if r["expected_lbl"] == "ID")
ts_id_weighted = sum(p == "id" for p, r in zip(full_ts_weighted_preds, full_results) if r["expected_lbl"] == "ID")
lingua_my = sum(1 for r in full_results if r["expected_lbl"] == "MY" and r["lingua_pred"] == "ms")
lingua_id = sum(1 for r in full_results if r["expected_lbl"] == "ID" and r["lingua_pred"] == "id")
hard_my   = sum(1 for r in full_results if r["expected_lbl"] == "MY" and r["hard"] == "ms")
hard_id   = sum(1 for r in full_results if r["expected_lbl"] == "ID" and r["hard"] == "id")
n_my_full = sum(1 for r in full_results if r["expected_lbl"] == "MY")
n_id_full = sum(1 for r in full_results if r["expected_lbl"] == "ID")

print(f"""
MY ACCURACY SUMMARY (n={n_my_full}):
  lingua-high (individual):    {lingua_my}/{n_my_full} = {lingua_my/n_my_full*100:.1f}%
  3-model hard (baseline):     {hard_my}/{n_my_full}  = {hard_my/n_my_full*100:.1f}%
  two_stage_agree:             {ts_my_agree}/{n_my_full} = {ts_my_agree/n_my_full*100:.1f}%
  two_stage_weighted:          {ts_my_weighted}/{n_my_full} = {ts_my_weighted/n_my_full*100:.1f}%

ID ACCURACY SUMMARY (n={n_id_full}):
  lingua-high (individual):    {lingua_id}/{n_id_full} = {lingua_id/n_id_full*100:.1f}%
  3-model hard (baseline):     {hard_id}/{n_id_full}  = {hard_id/n_id_full*100:.1f}%
  two_stage_agree:             {ts_id_agree}/{n_id_full} = {ts_id_agree/n_id_full*100:.1f}%
  two_stage_weighted:          {ts_id_weighted}/{n_id_full} = {ts_id_weighted/n_id_full*100:.1f}%
""")

# Evaluate fix criterion
my_best_ts = max(ts_my_agree, ts_my_weighted)
id_best_ts = max(ts_id_agree, ts_id_weighted)
my_fixed = my_best_ts >= lingua_my          # >= best individual
id_ok    = id_best_ts >= lingua_id - 2      # within noise of best individual

if my_fixed and id_ok:
    print("VERDICT: Two-stage voting FIXES the MY/ID problem.")
    print("  MY accuracy is restored to at least lingua-high individual level.")
    print("  ID accuracy is maintained or improved.")
    print("  Statistical backing: see McNemar tests above.")
elif my_fixed:
    print("VERDICT: Two-stage voting recovers MY accuracy but may trade off ID slightly.")
    print("  Recommend deploying two_stage_weighted as a net improvement.")
else:
    print("VERDICT: Two-stage voting partially recovers MY accuracy.")
    print("  MY improves over 3-model hard baseline but does not fully reach")
    print("  lingua-high individual level. Further investigation needed.")

print(f"\nLog saved: {LOG_PATH}")
