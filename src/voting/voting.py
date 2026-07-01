import sys
from pathlib import Path
from collections import defaultdict

from voting.core import (
    ROOT, LOG_DIR, DS_DIR, EXPECTED_TO_ISO, TARGET_LANGS, LANGUAGE_ORDER, BUCKET_ORDER,
    next_path, bucket_for, load_lingua, lingua_probs, langdetect_probs, pycld2_probs,
    hard_vote, soft_vote, weighted_vote, pick_top, DEFAULT_WEIGHTS,
)

# ==============================================================================
# PATHS
# ==============================================================================
LOG_PATH = next_path(LOG_DIR, "log_voting", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_PATH)


# ==============================================================================
# DATASET
# ==============================================================================
TEST_CASE_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else DS_DIR / "test_case_7_enmyid.txt"

if not TEST_CASE_FILE.exists():
    print(f"Error: Could not find {TEST_CASE_FILE!r}")
    sys.exit(1)

test_cases = []
with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, text = line.split("|", 1)
            test_cases.append({"expected": label.strip().upper(), "text": text.strip()})

print(f"Loaded {len(test_cases)} test cases from {TEST_CASE_FILE.name!r}")
print(f"Log: {LOG_PATH}\n")


# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("Loading models...")
detector = load_lingua()

print("  lingua-high   : ready")
print("  langdetect    : ready (lazy load)")
print("  pycld2        : ready (compiled C++)")
print("All models loaded.\n")


# ==============================================================================
# VOTING WEIGHTS  (ROC AUC from benchmarkV5 — log_combined_3.txt, strict scoring)
# ==============================================================================
#
# METHODOLOGICAL NOTE — langdetect AUC correction
#
# The initial benchmark (log_combined_1.txt / log_combined_2.txt) applied an
# id-as-ms proxy rule for langdetect: when the expected language was Malay (ms)
# and langdetect output Indonesian (id), the prediction was credited as correct.
# This was introduced to give langdetect a "fair" score given it ships no Malay
# profile, but it was methodologically inconsistent — no other model received
# equivalent credit for wrong predictions — and it masked a finding that proved
# central to the voting design: langdetect has zero recall on Malay (0/95 = 0.0%
# across all word-count buckets under strict exact-match scoring).
#
# Removing the proxy rule dropped langdetect's overall accuracy from 66.7% to
# 53.7% and its ROC AUC from 0.7593 to 0.7516. The corrected AUC is used here.
# The zero Malay recall is not a flaw in the ensemble design — it is the reason
# langdetect works in the ensemble: its systematic 'id' output for Malay text is
# a known, predictable signal that lingua-high and pycld2 reliably outvote 2-1,
# effectively making langdetect a strong ID/EN/TA voter that never competes on
# the MY axis.
#
# Source: log_combined_3.txt (strict scoring, no fallbacks). Weights themselves
# live in voting.core.DEFAULT_WEIGHTS — shared with every other script here.


# ==============================================================================
# RUN EVALUATION
# ==============================================================================
STRATEGIES = ["lingua-high", "langdetect", "pycld2", "hard", "soft", "weighted"]

# overall stats: {strategy: {lang_lbl: {total, correct}}}
lang_stats   = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))
bucket_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0})))

for case in test_cases:
    text         = case["text"]
    expected_lbl = case["expected"]
    if expected_lbl not in EXPECTED_TO_ISO:
        continue
    expected_iso = EXPECTED_TO_ISO[expected_lbl]
    bucket       = bucket_for(text)

    # --- individual model predictions ---
    lingua_p  = lingua_probs(detector, text)
    ld_p      = langdetect_probs(text)
    cld2_p    = pycld2_probs(text)

    lingua_pred = pick_top(lingua_p)
    ld_pred     = pick_top(ld_p)
    cld2_pred   = pick_top(cld2_p)

    # langdetect has no ms profile so it outputs 'id' for Malay text.
    # No score adjustment is made — the vote from lingua-high and pycld2 corrects this.

    preds = {
        "lingua-high": lingua_pred,
        "langdetect":  ld_pred,
        "pycld2":      cld2_pred,
        "hard":        hard_vote(lingua_pred, ld_pred, cld2_pred),
        "soft":        soft_vote(lingua_p, ld_p, cld2_p),
        "weighted":    weighted_vote(lingua_p, ld_p, cld2_p),
    }

    for strat, pred in preds.items():
        correct = int(pred == expected_iso)
        lang_stats[strat][expected_lbl]["total"]   += 1
        lang_stats[strat][expected_lbl]["correct"] += correct
        bucket_stats[strat][bucket][expected_lbl]["total"]   += 1
        bucket_stats[strat][bucket][expected_lbl]["correct"] += correct


# ==============================================================================
# PRINT RESULTS
# ==============================================================================
SEP = "=" * 115

def pct(s): return f"{s['correct']/s['total']*100:5.1f}%" if s["total"] else "  N/A "

# --- Overall accuracy ---
print(SEP)
print("VOTING RESULTS — OVERALL ACCURACY BY LANGUAGE")
print(SEP)
col_w = 12
hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{s:>{col_w}}" for s in STRATEGIES)
print(hdr)
print("-" * len(hdr))

for lbl in LANGUAGE_ORDER:
    n = lang_stats["lingua-high"][lbl]["total"]
    if n == 0:
        continue
    row = f"  {lbl:<5} | {n:>3} | " + " | ".join(
        f"{pct(lang_stats[s][lbl]):>{col_w}}" for s in STRATEGIES
    )
    print(row)

total = sum(lang_stats["lingua-high"][l]["total"] for l in LANGUAGE_ORDER)
print("-" * len(hdr))
overall_row = f"  {'ALL':<5} | {total:>3} | " + " | ".join(
    f"{sum(lang_stats[s][l]['correct'] for l in LANGUAGE_ORDER) / total * 100:>{col_w-1}.1f}%"
    for s in STRATEGIES
)
print(overall_row)

# --- Per-bucket accuracy ---
print(f"\n{SEP}")
print("VOTING RESULTS — ACCURACY BY BUCKET & LANGUAGE")
print(SEP)

for bucket in BUCKET_ORDER:
    if bucket not in bucket_stats["lingua-high"]:
        continue
    print(f"\nBucket: {bucket}")
    print(hdr)
    for lbl in LANGUAGE_ORDER:
        s0 = bucket_stats["lingua-high"][bucket].get(lbl, {"total": 0, "correct": 0})
        if s0["total"] == 0:
            continue
        n = s0["total"]
        row = f"  {lbl:<5} | {n:>3} | " + " | ".join(
            f"{pct(bucket_stats[s][bucket].get(lbl, {'total':0,'correct':0})):>{col_w}}"
            for s in STRATEGIES
        )
        print(row)

# --- Vote gain summary ---
print(f"\n{SEP}")
print("VOTE GAIN OVER BEST INDIVIDUAL MODEL")
print(SEP)

individual = ["lingua-high", "langdetect", "pycld2"]
voting     = ["hard", "soft", "weighted"]

for lbl in LANGUAGE_ORDER + ["ALL"]:
    if lbl == "ALL":
        ind_accs = {
            s: sum(lang_stats[s][l]["correct"] for l in LANGUAGE_ORDER) / total * 100
            for s in individual
        }
        vote_accs = {
            s: sum(lang_stats[s][l]["correct"] for l in LANGUAGE_ORDER) / total * 100
            for s in voting
        }
    else:
        n = lang_stats["lingua-high"][lbl]["total"]
        if n == 0:
            continue
        ind_accs  = {s: lang_stats[s][lbl]["correct"] / n * 100 for s in individual}
        vote_accs = {s: lang_stats[s][lbl]["correct"] / n * 100 for s in voting}

    best_ind  = max(ind_accs,  key=ind_accs.get)
    best_vote = max(vote_accs, key=vote_accs.get)
    gain = vote_accs[best_vote] - ind_accs[best_ind]
    sign = "+" if gain >= 0 else ""
    print(f"  {lbl:<5}  best individual: {best_ind:<12} {ind_accs[best_ind]:5.1f}%  |  "
          f"best vote: {best_vote:<8} {vote_accs[best_vote]:5.1f}%  |  gain: {sign}{gain:.1f}%")

print(f"\n{'='*115}")
print(f"Voting weights used  —  lingua-high: {DEFAULT_WEIGHTS['lingua']}  |  "
      f"langdetect: {DEFAULT_WEIGHTS['langdetect']}  |  pycld2: {DEFAULT_WEIGHTS['pycld2']}")
print(f"(Weights derived from ROC AUC scores in benchmarkV5)")
print(f"\nLog saved to: {LOG_PATH}")
