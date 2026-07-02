"""
src/benchmark/reweight_by_real_distribution.py

The 70.8%/56.5%/etc. accuracy figures in the report (reports/language_detection_ensemble_evaluation.md)
are all computed on `test_case_7_enmyid.txt`'s bucket composition: 677/399/94/73/30 cases across the
five word-count buckets (1 word / 2 words / 3-7 / 8-16 / 17-50). That split is a synthetic, hand-built
test design (§1.1 of the report) — 53% of it is single-word cases, the bucket the report itself (§12)
flags as having the weakest, least-representative signal (peak accuracy ~49% even for the best model).

Real production query-length distributions almost certainly do NOT look like this test set. This
script recomputes ALL/EN/MY/ID accuracy as a weighted average over each strategy's *per-bucket*
accuracy, using a caller-supplied word-count histogram as the weights instead of the test set's raw
bucket counts. It does not re-run any model — it takes the per-bucket accuracy that has already been
measured (embedded below, sourced from the report and its underlying logs) and asks "what would the
blended accuracy be if buckets were mixed in these proportions instead?"

Usage:
    python src/benchmark/reweight_by_real_distribution.py                  # built-in preset distributions
    python src/benchmark/reweight_by_real_distribution.py --histogram my_histogram.json
    python src/benchmark/reweight_by_real_distribution.py --strategy s2_weighted --histogram my_histogram.json

A histogram JSON file maps bucket name -> relative weight, e.g.:
    {"1 word": 0.05, "2 words": 0.10, "3-7 words": 0.45, "8-16 words": 0.30, "17-50 words": 0.10}
Weights need not sum to 1 — they are normalized before use.

LIMITATION: this only reweights the word-count-bucket mix, not the EN/MY/ID language mix. The "ALL"
figure still assumes production traffic splits across languages roughly the way this test set does
(432 EN / 421 MY / 420 ID, i.e. close to even thirds). No real-traffic language-mix data exists yet
to calibrate that separately; if EN/MY/ID traffic proportions are known to differ substantially from
even thirds, extend `combine_all()` below to accept a language-mix histogram the same way.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voting.core import BUCKET_ORDER, LANGUAGE_ORDER

# ── Embedded per-bucket accuracy, (correct, total) per bucket per language ──────────────
#
# Sources (all on test_case_7_enmyid.txt, n=1,273: EN 432 / MY 421 / ID 420):
#   - "langdetect"              : report §1.3 (current production baseline, single model)
#   - "lingua_high"             : report §1.3 (best individual model)
#   - "s1_two_stage_weighted"   : EN from S1 3-model hard vote (results/logs/test_case_7_enmyid/
#                                 log_voting_2.txt, "hard" column — Stage 1's coarse result is final
#                                 for EN, so it equals the plain hard vote there); MY/ID from
#                                 results/logs/test_case_7_enmyid/log_voting_two_stage_1.txt
#                                 ("2s-wgtd" column, "TWO-STAGE — MY and ID ACCURACY BY BUCKET").
#   - "s2_weighted"             : recomputed live (lingua-high + openlid-v3 + pycld2, weighted vote,
#                                 weights from src/voting/strategies/weighted.py's S2_WEIGHTS) since
#                                 the original run (src/voting/strategies/, formerly scenario2/) only
#                                 logged bucket breakdowns for hard/soft, not weighted. Overall
#                                 totals reproduce the report's 92.1% / 43.7% / 76.0% exactly
#                                 (398/432, 184/421, 319/420).
#
# Every strategy's totals below sum to the report's headline EN/MY/ID/ALL figures — this is a
# reweighting of real measured numbers, not a simulation.
BUCKET_ACCURACY = {
    "langdetect": {
        "1 word":      {"EN": (58, 226),  "MY": (0, 226),  "ID": (58, 225)},
        "2 words":     {"EN": (64, 141),  "MY": (0, 129),  "ID": (65, 129)},
        "3-7 words":   {"EN": (28, 30),   "MY": (0, 32),   "ID": (29, 32)},
        "8-16 words":  {"EN": (25, 25),   "MY": (0, 24),   "ID": (24, 24)},
        "17-50 words": {"EN": (10, 10),   "MY": (0, 10),   "ID": (10, 10)},
    },
    "lingua_high": {
        "1 word":      {"EN": (201, 226), "MY": (107, 226), "ID": (111, 225)},
        "2 words":     {"EN": (133, 141), "MY": (79, 129),  "ID": (80, 129)},
        "3-7 words":   {"EN": (29, 30),   "MY": (25, 32),   "ID": (21, 32)},
        "8-16 words":  {"EN": (25, 25),   "MY": (17, 24),   "ID": (23, 24)},
        "17-50 words": {"EN": (10, 10),   "MY": (7, 10),    "ID": (8, 10)},
    },
    "s1_two_stage_weighted": {
        "1 word":      {"EN": (201, 226), "MY": (104, 226), "ID": (114, 225)},
        "2 words":     {"EN": (133, 141), "MY": (79, 129),  "ID": (83, 129)},
        "3-7 words":   {"EN": (30, 30),   "MY": (27, 32),   "ID": (29, 32)},
        "8-16 words":  {"EN": (25, 25),   "MY": (20, 24),   "ID": (24, 24)},
        "17-50 words": {"EN": (10, 10),   "MY": (8, 10),    "ID": (8, 10)},
    },
    "s2_weighted": {
        "1 word":      {"EN": (199, 226), "MY": (74, 226),  "ID": (146, 225)},
        "2 words":     {"EN": (134, 141), "MY": (53, 129),  "ID": (107, 129)},
        "3-7 words":   {"EN": (30, 30),   "MY": (27, 32),   "ID": (32, 32)},
        "8-16 words":  {"EN": (25, 25),   "MY": (21, 24),   "ID": (24, 24)},
        "17-50 words": {"EN": (10, 10),   "MY": (9, 10),    "ID": (10, 10)},
    },
}

# Original per-language totals in the test set — used as the language-mix weights for ALL (see
# LIMITATION in the module docstring).
LANG_TOTALS = {"EN": 432, "MY": 421, "ID": 420}

# The test set's own bucket composition, expressed as a histogram — reproduces the report's
# published figures exactly when passed through reweight(), which is a useful correctness check.
TEST_SET_HISTOGRAM = {"1 word": 677, "2 words": 399, "3-7 words": 94, "8-16 words": 73, "17-50 words": 30}

# 2-3 plausible real-traffic distributions, expressed as relative weights (need not sum to 1).
# These are illustrative, not measured — swap in a real query-length histogram via --histogram
# once one is available from production logs.
PRESET_DISTRIBUTIONS = {
    "test_set (current, for comparison)": TEST_SET_HISTOGRAM,
    "chat_short (mostly 1-2 word lookups)": {
        "1 word": 0.45, "2 words": 0.35, "3-7 words": 0.15, "8-16 words": 0.04, "17-50 words": 0.01,
    },
    "query_typical (mostly 3-16 word queries)": {
        "1 word": 0.05, "2 words": 0.10, "3-7 words": 0.45, "8-16 words": 0.30, "17-50 words": 0.10,
    },
    "long_form (mostly 8-50 word content)": {
        "1 word": 0.02, "2 words": 0.03, "3-7 words": 0.15, "8-16 words": 0.35, "17-50 words": 0.45,
    },
}


def reweight(bucket_acc: dict, histogram: dict) -> dict:
    """
    Recompute {EN, MY, ID, ALL} accuracy for one strategy's bucket_acc table, using `histogram`
    (bucket -> relative weight) instead of the raw bucket counts baked into bucket_acc.

    bucket_acc: {bucket: {lang: (correct, total)}}
    histogram:  {bucket: weight}  (any positive scale; normalized internally)

    Returns {lang: accuracy_fraction} for lang in LANGUAGE_ORDER + ["ALL"].
    """
    total_weight = sum(histogram.get(b, 0.0) for b in BUCKET_ORDER)
    if total_weight <= 0:
        raise ValueError("histogram has no positive weight on any known bucket")

    per_lang = {}
    for lang in LANGUAGE_ORDER:
        acc_sum = 0.0
        for bucket in BUCKET_ORDER:
            w = histogram.get(bucket, 0.0)
            if w <= 0:
                continue
            correct, n = bucket_acc[bucket][lang]
            rate = correct / n if n else 0.0
            acc_sum += w * rate
        per_lang[lang] = acc_sum / total_weight

    per_lang["ALL"] = combine_all(per_lang)
    return per_lang


def combine_all(per_lang: dict) -> float:
    """Blend per-language reweighted accuracy into one ALL figure, weighted by this test set's
    original EN/MY/ID mix (see LIMITATION in the module docstring)."""
    total_n = sum(LANG_TOTALS.values())
    return sum(per_lang[lang] * LANG_TOTALS[lang] for lang in LANGUAGE_ORDER) / total_n


def load_histogram(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_table(strategy: str, distributions: dict):
    bucket_acc = BUCKET_ACCURACY[strategy]
    print(f"\nStrategy: {strategy}")
    col_w = 8
    hdr = f"  {'Distribution':<42} | " + " | ".join(f"{l:>{col_w}}" for l in LANGUAGE_ORDER + ["ALL"])
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, hist in distributions.items():
        acc = reweight(bucket_acc, hist)
        row = f"  {name:<42} | " + " | ".join(f"{acc[l]*100:>{col_w-1}.1f}%" for l in LANGUAGE_ORDER + ["ALL"])
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histogram", type=str, default=None,
                         help="Path to a JSON file mapping bucket name -> weight. If omitted, "
                              "prints all built-in preset distributions instead.")
    parser.add_argument("--strategy", type=str, default=None, choices=list(BUCKET_ACCURACY),
                         help="Only print this strategy. Default: print all strategies.")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else list(BUCKET_ACCURACY)

    if args.histogram:
        hist = load_histogram(args.histogram)
        distributions = {Path(args.histogram).name: hist}
    else:
        distributions = PRESET_DISTRIBUTIONS

    print("Reweighting bucket-level accuracy by word-count distribution.")
    print("(See TEST_SET_HISTOGRAM row for the report's own published figures, as a sanity check.)")
    for strategy in strategies:
        print_table(strategy, distributions)


if __name__ == "__main__":
    main()
