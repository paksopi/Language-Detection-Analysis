"""
voting/auc_unified.py — Unified AUC computation matching benchmarkV5.py's method.

benchmarkV5 computes AUC as: binary = (top-1 correct?), score = top-1 confidence.
The old voting_two_stage.py used per-class OVR probability — a different metric that
yields inconsistent results (pycld2: 0.9634 -> 0.7756).
This module provides compute_binary_auc() for use by all voting_*.py scripts.

Writes: log/log_auc_unified_N.txt
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, next_path,
    load_dataset, load_lingua, run_predictions,
)

LOG_PATH = next_path(LOG_DIR, "log_auc_unified", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 80

# ── Unified AUC function (public — import this in other scripts) ──────────────
def compute_binary_auc(results: list, pred_key: str, conf_key: str) -> float:
    """
    AUC matching benchmarkV5.py exactly:
      binary[i] = 1 if results[i][pred_key] == results[i]['expected_iso'] else 0
      score[i]  = results[i][conf_key]   (top-1 predicted-class confidence)
    Returns sklearn.metrics.roc_auc_score(binary, scores).
    Returns nan if only one class present (AUC undefined).
    Library: sklearn.metrics.roc_auc_score
    """
    binary = [int(r[pred_key] == r["expected_iso"]) for r in results]
    scores = [r[conf_key] for r in results]
    if len(set(binary)) < 2:
        return float("nan")
    return float(roc_auc_score(binary, scores))

# ── Demo: run on full 475-case dataset and compare against benchmarkV5 values ─
print(f"Log: {LOG_PATH}")
print("Loading lingua-high detector...")
detector = load_lingua()
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} cases. Running predictions...")
results = run_predictions(cases, detector)
print("Done.\n")

MODEL_CONF_PAIRS = [
    ("lingua_pred", "lingua_conf", "lingua-high",  0.8503),
    ("ld_pred",     "ld_conf",     "langdetect",   0.7516),
    ("cld2_pred",   "cld2_conf",   "pycld2",       0.9634),
]

print(SEP)
print("UNIFIED AUC (binary correct + top-1 confidence, matching benchmarkV5)")
print(SEP)
print(f"\n  {'Model':<14} {'Unified AUC':>12}  {'benchmarkV5':>12}  {'delta':>8}  match?")
print(f"  {'':-<14} {'':-^12}  {'':-^12}  {'':-^8}  {'':-^6}")

for pred_key, conf_key, name, bv5_auc in MODEL_CONF_PAIRS:
    auc = compute_binary_auc(results, pred_key, conf_key)
    delta = auc - bv5_auc
    match = "YES" if abs(delta) < 0.005 else "NO"
    print(f"  {name:<14} {auc:>12.4f}  {bv5_auc:>12.4f}  {delta:>+8.4f}  {match}")

print(f"""
Notes:
  - benchmarkV5 computes AUC from top-1 prediction binary + top-1 confidence score.
  - The old compute_macro_auc in voting_two_stage.py used per-class OVR probability
    vectors, yielding different (inconsistent) results.
  - pycld2's 0.9634 AUC comes from its sharp discrimination: very high confidence
    when correct, near-zero when it declines to predict (unknown cases).
  - This function is importable: from auc_unified import compute_binary_auc
""")

print(f"Log saved: {LOG_PATH}")
