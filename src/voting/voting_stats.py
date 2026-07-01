"""
voting/voting_stats.py — Rigorous statistical evaluation of the voting ensemble.

Objectives:
  1. McNemar's test (paired significance) for each comparison pair, per language
  2. Binomial confidence intervals (95 %) on all reported accuracies
  3. Cohen's kappa + pairwise agreement between the 3 individual models
  4. Calibration analysis: ECE + reliability diagrams + confidence distributions

Writes:
  log/log_voting_stats_N.txt
  calibration/reliability_N.png
  calibration/conf_dist_N.png
"""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Statistical libraries — see function-level comments for exact APIs used
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar   # McNemar
from statsmodels.stats.proportion import proportion_confint               # binomial CI
from sklearn.metrics import cohen_kappa_score                             # kappa
from sklearn.calibration import calibration_curve                         # reliability

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ROOT, LOG_DIR, CALIB_DIR, DS_DIR,
    LANGUAGE_ORDER, BUCKET_ORDER, IND_KEYS, VOTE_KEYS, ALL_KEYS, KEY_NAMES,
    next_path, load_dataset, load_lingua, run_predictions,
    binary_correct, pred_labels, overall_accuracy, accuracy_by_lang, pct,
)

LOG_PATH   = next_path(LOG_DIR,   "log_voting_stats", ".txt")
RELI_PATH  = next_path(CALIB_DIR, "reliability",      ".png")
CDIST_PATH = next_path(CALIB_DIR, "conf_dist",        ".png")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 100

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"Log: {LOG_PATH}")
print("Loading lingua-high detector...")
detector = load_lingua()
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} test cases. Running predictions...")
results = run_predictions(cases, detector)
print("Predictions complete.\n")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 1 — McNemar's Test
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("OBJECTIVE 1 — McNEMAR'S TEST (paired significance)")
print(SEP)
print("""
Two classifiers scored on the same 475 cases. The 2×2 table:
  [[both correct, A correct B wrong],
   [A wrong B correct, both wrong]]
Key cells: b = (A right, B wrong), c = (A wrong, B right).
If b+c < 25 → exact binomial; else → chi-squared (correction=True).
Bonferroni threshold: α = 0.05 / 36 comparisons = 0.0014  → flagged ***
Standard thresholds: ** p < 0.01,  * p < 0.05
Library: statsmodels.stats.contingency_tables.mcnemar
""")

COMPARISON_PAIRS = [
    ("lingua_pred", "hard"),
    ("lingua_pred", "soft"),
    ("lingua_pred", "weighted"),
    ("hard",        "soft"),
    ("hard",        "weighted"),
    ("soft",        "weighted"),
]

def run_mcnemar(arr_a: np.ndarray, arr_b: np.ndarray):
    """
    Run McNemar's test on binary correct/incorrect arrays.
    Returns (stat, p_value, b, c, method).
    """
    b = int(np.sum((arr_a == 1) & (arr_b == 0)))
    c = int(np.sum((arr_a == 0) & (arr_b == 1)))
    n00 = int(np.sum((arr_a == 0) & (arr_b == 0)))
    n11 = int(np.sum((arr_a == 1) & (arr_b == 1)))
    table = np.array([[n11, b], [c, n00]])
    use_exact = (b + c) < 25
    res = sm_mcnemar(table, exact=use_exact, correction=not use_exact)
    method = "exact" if use_exact else "chi2"
    return res.statistic, res.pvalue, b, c, method

def sig_marker(p):
    if p < 0.0014: return "***"
    if p < 0.01:   return " **"
    if p < 0.05:   return "  *"
    return "   "

for scope in LANGUAGE_ORDER + ["ALL"]:
    lang_filter = None if scope == "ALL" else scope
    n = len([r for r in results if r["expected_lbl"] == lang_filter]) if lang_filter else len(results)
    print(f"── {scope}  (n={n}) " + "─" * 60)
    print(f"  {'Pair':<35} {'p-value':>10}  {'sig':>3}  {'b':>4}  {'c':>4}  {'method'}")
    print(f"  {'':-<35} {'':-^10}  {'':-^3}  {'':-^4}  {'':-^4}  {'':-^6}")
    for ka, kb in COMPARISON_PAIRS:
        arr_a = binary_correct(results, ka, lang_filter)
        arr_b = binary_correct(results, kb, lang_filter)
        stat, p, b, c, method = run_mcnemar(arr_a, arr_b)
        name_a = KEY_NAMES[ka]
        name_b = KEY_NAMES[kb]
        pair_label = f"{name_a} vs {name_b}"
        print(f"  {pair_label:<35} {p:>10.4f}  {sig_marker(p)}  {b:>4}  {c:>4}  {method}")
    print()

print("""Interpretation note:
  - The -6.3% MY drop (lingua-high 65.3% -> hard 58.9%) and +16.8% ID gain
    (lingua-high 69.5% -> soft 86.3%) are the two headline findings.
    See significance markers above to assess statistical reliability.
  - Small n (95 per language) means b+c is often < 25; exact test used.
""")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 2 — Binomial Confidence Intervals
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("OBJECTIVE 2 — BINOMIAL CONFIDENCE INTERVALS (95%, Wilson method)")
print(SEP)
print("""
Each reported accuracy is a Bernoulli proportion. Wilson interval (statsmodels
proportion_confint method='wilson') is used — it handles p near 0/1 better
than the normal approximation.
Library: statsmodels.stats.proportion.proportion_confint
""")

col_names = [KEY_NAMES[k] for k in ALL_KEYS]
col_w = 26
hdr = f"  {'Lang':<5} | {'n':>3} | " + " | ".join(f"{n:^{col_w}}" for n in col_names)
print(hdr)
print("-" * len(hdr))

for scope in LANGUAGE_ORDER + ["ALL"]:
    lang_filter = None if scope == "ALL" else scope
    rows = [r for r in results if r["expected_lbl"] == lang_filter] if lang_filter else results
    n = len(rows)
    parts = []
    for k in ALL_KEYS:
        c = sum(1 for r in rows if r[k] == r["expected_iso"])
        lo, hi = proportion_confint(c, n, alpha=0.05, method="wilson")
        parts.append(f"{c/n*100:5.1f}% [{lo*100:4.1f}%–{hi*100:4.1f}%]")
    print(f"  {scope:<5} | {n:>3} | " + " | ".join(f"{p:^{col_w}}" for p in parts))

print("""
CI width at n=95 is ≈ ±10 pp for proportions near 50% and ≈ ±5 pp near 95%.
Overlapping CIs do not imply no difference — use McNemar (Obj. 1) for that.
""")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 3 — Cohen's Kappa + Pairwise Agreement
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("OBJECTIVE 3 — INTER-MODEL AGREEMENT (Cohen's kappa + pairwise agreement rate)")
print(SEP)
print("""
Cohen's kappa measures agreement correcting for chance.
  κ > 0.8 → near-perfect; 0.6–0.8 → substantial; 0.4–0.6 → moderate
Low kappa = diverse voters (ensemble should help); high kappa = redundant voters.
Library: sklearn.metrics.cohen_kappa_score
""")

MODEL_KEYS   = ["lingua_pred", "ld_pred", "cld2_pred"]
MODEL_NAMES  = ["lingua-high", "langdetect", "pycld2"]
KAPPA_PAIRS  = [
    ("lingua_pred", "ld_pred",   "lingua-high vs langdetect"),
    ("lingua_pred", "cld2_pred", "lingua-high vs pycld2"),
    ("ld_pred",     "cld2_pred", "langdetect  vs pycld2"),
]

for scope in LANGUAGE_ORDER + ["ALL"]:
    lang_filter = None if scope == "ALL" else scope
    rows = [r for r in results if r["expected_lbl"] == lang_filter] if lang_filter else results
    n = len(rows)
    print(f"── {scope}  (n={n}) " + "─" * 55)
    print(f"  {'Pair':<32} {'kappa':>7}  {'agree %':>8}  note")
    for ka, kb, label in KAPPA_PAIRS:
        preds_a = [r[ka] for r in rows]
        preds_b = [r[kb] for r in rows]
        kappa   = cohen_kappa_score(preds_a, preds_b)
        agree   = sum(a == b for a, b in zip(preds_a, preds_b)) / n * 100
        note = ""
        # Flag structural artifact: langdetect outputs 'id' for Malay, so
        # agreement with pycld2 on MY cases is structurally inflated — not
        # independent agreement
        if scope == "MY" and "ld_pred" in (ka, kb):
            note = "← structural (ld→id biased vs true ms)"
        print(f"  {label:<32} {kappa:>7.4f}  {agree:>7.1f}%  {note}")
    print()

print("""Notes:
  - High kappa between langdetect and pycld2 on MY cases signals a structural
    artifact, not genuine agreement: langdetect can only output 'id' for Malay,
    so any pycld2 output of 'id' on a true-MY case creates spurious 'agreement'.
    This violates the voter-independence assumption for the ms/id axis.
  - The overall kappas guide which model pairs are genuinely diverse.
""")

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 4 — Calibration (ECE + Reliability Diagrams)
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("OBJECTIVE 4 — CALIBRATION ANALYSIS (ECE + reliability diagrams)")
print(SEP)
print("""
A well-calibrated model: when it says 0.7 confidence → it's correct 70% of the time.
ECE = Σ (n_bin/n_total) × |avg_confidence − fraction_correct|  (lower is better)
Library: sklearn.calibration.calibration_curve for reliability; ECE computed manually.
""")

MODEL_CONF_KEYS = [
    ("lingua_pred", "lingua_conf", "lingua_p",  "lingua-high"),
    ("ld_pred",     "ld_conf",     "ld_p",      "langdetect"),
    ("cld2_pred",   "cld2_conf",   "cld2_p",    "pycld2"),
]

N_BINS = 10

def compute_ece(conf_arr, correct_arr, n_bins=N_BINS):
    """
    Expected Calibration Error — manual computation.
    conf_arr: confidence for the predicted class.
    correct_arr: binary 1/0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf_arr)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf_arr >= lo) & (conf_arr < hi) if i < n_bins - 1 else (conf_arr >= lo) & (conf_arr <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = conf_arr[mask].mean()
        avg_acc  = correct_arr[mask].mean()
        ece += (n_bin / n) * abs(avg_conf - avg_acc)
    return ece

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Reliability Diagrams — Individual Models", fontsize=13)

print(f"  {'Model':<14} {'ECE':>7}  {'Avg conf (correct)':>20}  {'Avg conf (wrong)':>18}")
print(f"  {'':-<14} {'':-^7}  {'':-^20}  {'':-^18}")

for ax, (pred_key, conf_key, prob_key, model_name) in zip(axes, MODEL_CONF_KEYS):
    confs   = np.array([r[conf_key] for r in results])
    correct = np.array([int(r[pred_key] == r["expected_iso"]) for r in results])

    ece = compute_ece(confs, correct)

    mean_conf_correct = confs[correct == 1].mean() if (correct == 1).any() else float("nan")
    mean_conf_wrong   = confs[correct == 0].mean() if (correct == 0).any() else float("nan")

    print(f"  {model_name:<14} {ece:>7.4f}  {mean_conf_correct:>20.4f}  {mean_conf_wrong:>18.4f}")

    # Reliability diagram using sklearn.calibration.calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(correct, confs, n_bins=N_BINS, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label=model_name, color="steelblue")
    except Exception:
        # If too few unique confidence values, fall back to quantile strategy
        frac_pos, mean_pred = calibration_curve(correct, confs, n_bins=N_BINS, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "s-", label=model_name, color="steelblue")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"{model_name}\nECE = {ece:.4f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RELI_PATH, dpi=150)
plt.close()
print(f"\nReliability diagram saved: {RELI_PATH}")

# ── Confidence distributions: correct vs wrong ─────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Confidence Distribution — Correct vs Incorrect Predictions", fontsize=13)

print(f"\n  {'Model':<14} {'Median conf (correct)':>22}  {'Median conf (wrong)':>21}")
print(f"  {'':-<14} {'':-^22}  {'':-^21}")

for ax2, (pred_key, conf_key, prob_key, model_name) in zip(axes2, MODEL_CONF_KEYS):
    confs   = np.array([r[conf_key] for r in results])
    correct = np.array([int(r[pred_key] == r["expected_iso"]) for r in results])

    c_right = confs[correct == 1]
    c_wrong = confs[correct == 0]
    med_right = np.median(c_right) if len(c_right) else float("nan")
    med_wrong = np.median(c_wrong) if len(c_wrong) else float("nan")
    print(f"  {model_name:<14} {med_right:>22.4f}  {med_wrong:>21.4f}")

    bins = np.linspace(0, 1, 21)
    ax2.hist(c_right, bins=bins, alpha=0.6, color="green",  label=f"correct (n={len(c_right)})",  density=True)
    ax2.hist(c_wrong, bins=bins, alpha=0.6, color="red",    label=f"wrong   (n={len(c_wrong)})",  density=True)
    ax2.set_xlabel("Confidence (max class probability)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"{model_name}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CDIST_PATH, dpi=150)
plt.close()
print(f"Confidence distribution plot saved: {CDIST_PATH}")

print("""
Calibration interpretation:
  - If one model's confidence scale is systematically higher than another's,
    soft_vote and weighted_vote will be biased toward that model regardless of
    AUC -- because raw probability averaging amplifies the high-confidence model.
  - The mean/median confidence gap between correct and wrong predictions
    measures discrimination (not calibration) -- a well-discriminating model
    is confident when right and uncertain when wrong.
""")

print(f"\n{SEP}")
print(f"Log saved: {LOG_PATH}")
print(f"Plots:     {RELI_PATH}")
print(f"           {CDIST_PATH}")
