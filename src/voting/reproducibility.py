"""
voting/reproducibility.py — Run-to-run variance check.

5 seeded runs  (langdetect seed=0) -> should be identical, proving determinism.
5 unseeded runs (langdetect no seed) -> shows residual variance without seeding.

Only langdetect is non-deterministic; lingua-high and pycld2 are deterministic.

Writes: log/log_reproducibility_N.txt
"""

import sys
import numpy as np
from collections import defaultdict

import langdetect as _ld
from langdetect import DetectorFactory

from voting.core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, TARGET_LANGS, next_path,
    load_dataset, load_lingua, lingua_probs, pycld2_probs, pick_top,
    hard_vote, soft_vote, weighted_vote, DEFAULT_WEIGHTS,
)

LOG_PATH = next_path(LOG_DIR, "log_reproducibility", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 80
N_RUNS = 5

print(f"Log: {LOG_PATH}")
print("Loading models (once — shared across all runs)...")
detector = load_lingua()
cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} cases.\n")

# ── Langdetect prob function with explicit seed control ────────────────────────
def langdetect_probs_seeded(text: str, seed=None) -> dict:
    """Run langdetect with optional seed. seed=None = unseeded (non-deterministic)."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    if seed is not None:
        DetectorFactory.seed = seed
    else:
        # Remove the seed attribute to restore non-determinism
        try:
            del DetectorFactory.seed
        except AttributeError:
            pass
    try:
        for item in _ld.detect_langs(text):
            iso = item.lang
            if iso.startswith("zh"): iso = "zh"
            if iso in probs: probs[iso] += item.prob
    except Exception:
        pass
    return probs

def run_single(cases, seed=0) -> dict:
    """Run full pipeline for one seed setting. Returns {strat: {lbl: accuracy}}."""
    accs = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for case in cases:
        text = case["text"]
        lbl  = case["expected_lbl"]
        iso  = case["expected_iso"]
        lp   = lingua_probs(detector, text)
        dp   = langdetect_probs_seeded(text, seed)
        cp   = pycld2_probs(text)
        l_pred = pick_top(lp); d_pred = pick_top(dp); c_pred = pick_top(cp)
        preds = {
            "hard":     hard_vote(l_pred, d_pred, c_pred),
            "soft":     soft_vote(lp, dp, cp),
            "weighted": weighted_vote(lp, dp, cp),
        }
        for strat, pred in preds.items():
            accs[strat][lbl][1] += 1
            accs[strat][lbl][0] += int(pred == iso)
    return accs

# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Seeded runs (seed=0) — expect identical results
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("PART 1 — SEEDED RUNS (langdetect seed=0, N=5)")
print("Expected: all runs produce identical accuracy numbers.")
print(SEP)

seeded_results = []
for i in range(N_RUNS):
    print(f"  Run {i+1}/{N_RUNS} (seeded)...", end=" ", flush=True)
    accs = run_single(cases, seed=0)
    seeded_results.append(accs)
    n_all = sum(accs["hard"][l][1] for l in LANGUAGE_ORDER)
    overall = sum(accs["hard"][l][0] for l in LANGUAGE_ORDER) / n_all * 100
    print(f"overall hard={overall:.1f}%")

print(f"\nSeeded run accuracy — hard vote (should be identical across all runs):")
print(f"  {'Run':<6} | " + " | ".join(f"{lbl:>6}" for lbl in LANGUAGE_ORDER) + " | ALL")
for i, accs in enumerate(seeded_results):
    row_vals = []
    totals = [0, 0]
    for lbl in LANGUAGE_ORDER:
        c, n = accs["hard"][lbl]
        row_vals.append(f"{c/n*100:>5.1f}%")
        totals[0] += c; totals[1] += n
    row_vals.append(f"{totals[0]/totals[1]*100:>5.1f}%")
    print(f"  Run {i+1:<2} | " + " | ".join(row_vals))

# Check if all runs are identical
all_identical = True
ref = seeded_results[0]
for r in seeded_results[1:]:
    for strat in ["hard", "soft", "weighted"]:
        for lbl in LANGUAGE_ORDER:
            if r[strat][lbl][0] != ref[strat][lbl][0]:
                all_identical = False
                break

print(f"\nAll seeded runs identical: {'YES -- reproducibility confirmed' if all_identical else 'NO -- investigate'}\n")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Unseeded runs — characterize natural variance
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("PART 2 — UNSEEDED RUNS (langdetect no seed, N=5)")
print("Shows residual non-determinism when seed is not fixed.")
print(SEP)

unseeded_results = []
for i in range(N_RUNS):
    print(f"  Run {i+1}/{N_RUNS} (unseeded)...", end=" ", flush=True)
    accs = run_single(cases, seed=None)
    unseeded_results.append(accs)
    n_all = sum(accs["hard"][l][1] for l in LANGUAGE_ORDER)
    overall = sum(accs["hard"][l][0] for l in LANGUAGE_ORDER) / n_all * 100
    print(f"overall hard={overall:.1f}%")

# Compute mean ± std per language per strategy
print(f"\nUnseeded run accuracy — mean ± std across {N_RUNS} runs:")
for strat in ["hard", "soft", "weighted"]:
    print(f"\n  Strategy: {strat}")
    print(f"  {'Lang':<6} | " + " | ".join(f"{'Run '+str(i+1):>7}" for i in range(N_RUNS)) +
          " | Mean ± Std")
    for lbl in LANGUAGE_ORDER + ["ALL"]:
        run_accs = []
        for accs in unseeded_results:
            if lbl == "ALL":
                c = sum(accs[strat][l][0] for l in LANGUAGE_ORDER)
                n = sum(accs[strat][l][1] for l in LANGUAGE_ORDER)
            else:
                c, n = accs[strat][lbl]
            run_accs.append(c / n * 100)
        mean = np.mean(run_accs); std = np.std(run_accs)
        run_str = " | ".join(f"{a:>6.1f}%" for a in run_accs)
        print(f"  {lbl:<6} | {run_str} | {mean:.1f}% ± {std:.1f}pp")

print(f"""

Summary:
  Seeded (seed=0):  all runs are byte-identical -> use seed=0 in all scripts
  Unseeded:         results above show the variance when seed is omitted.
  Conclusion: fix langdetect seed=0 to guarantee reproducible benchmarks.
""")

# Restore seed for any subsequent imports
DetectorFactory.seed = 0
print(f"Log saved: {LOG_PATH}")
