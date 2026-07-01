import sys
import math
import re
import regex
from pathlib import Path
from collections import defaultdict

from lingua import Language, LanguageDetectorBuilder
import langdetect
import pycld2

# ==============================================================================
# PATHS
# ==============================================================================
ROOT    = Path(__file__).parent.parent.parent
LOG_DIR = ROOT / "results" / "logs" / "test_case_7_enmyid"
DS_DIR  = ROOT / "data"

def next_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while (directory / f"{stem}_{n}{suffix}").exists():
        n += 1
    return directory / f"{stem}_{n}{suffix}"

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

EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id"}
TARGET_LANGS    = ["en", "ms", "id"]
LANGUAGE_ORDER  = ["EN", "MY", "ID"]

print(f"Loaded {len(test_cases)} test cases from {TEST_CASE_FILE.name!r}")
print(f"Log: {LOG_PATH}\n")


# ==============================================================================
# WORD-COUNT BUCKETING
# ==============================================================================
CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3)]

def is_cjk(text):
    cjk = sum(1 for ch in text if any(s <= ord(ch) <= e for s, e in CJK_RANGES))
    return cjk > 0 and cjk >= len(text.replace(" ", "")) * 0.3

def bucket_for(text):
    if is_cjk(text):
        n = len(re.sub(r'[^\w]', '', text))
        if n <= 2:   return "1 word"
        elif n <= 6:  return "2 words"
        elif n <= 15: return "3-7 words"
        elif n <= 48: return "8-16 words"
        else:         return "17-50 words"
    else:
        clean = re.sub(r'[^\w\s஀-௿]', '', text)
        n = len(clean.split())
        if n <= 1:   return "1 word"
        elif n <= 2:  return "2 words"
        elif n <= 7:  return "3-7 words"
        elif n <= 16: return "8-16 words"
        else:         return "17-50 words"

BUCKET_ORDER = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]


# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("Loading models...")

LANGS = (Language.ENGLISH, Language.MALAY, Language.INDONESIAN)
detector = LanguageDetectorBuilder.from_languages(*LANGS).build()

print("  lingua-high   : ready")
print("  langdetect    : ready (lazy load)")
print("  pycld2        : ready (compiled C++)")
print("All models loaded.\n")


# ==============================================================================
# LINGUA-HIGH — ISO mapper
# ==============================================================================
_LINGUA_ISO = {
    Language.ENGLISH:    "en",
    Language.MALAY:      "ms",
    Language.INDONESIAN: "id",
}

def lingua_probs(text):
    """Return {iso: confidence} for all 5 target languages."""
    confs = detector.compute_language_confidence_values(text)
    return {_LINGUA_ISO[c.language]: c.value for c in confs if c.language in _LINGUA_ISO}


# ==============================================================================
# LANGDETECT — ISO mapper
# ==============================================================================
def langdetect_probs(text):
    """Return {iso: probability} for detected languages, normalised to target set."""
    try:
        raw = langdetect.detect_langs(text)
    except Exception:
        return {l: 0.0 for l in TARGET_LANGS}

    probs = {l: 0.0 for l in TARGET_LANGS}
    for item in raw:
        iso = item.lang
        if iso.startswith("zh"):
            iso = "zh"
        if iso in probs:
            probs[iso] += item.prob
    return probs


# ==============================================================================
# PYCLD2 — ISO mapper
# ==============================================================================
def pycld2_probs(text):
    """Return {iso: confidence} from CLD2 percent scores, normalised to target set."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    try:
        _, _, details = pycld2.detect(text)
        total = sum(d[2] for d in details if d[1].lower() in TARGET_LANGS
                    or d[1].lower().startswith("zh"))
        if total == 0:
            return probs
        for d in details:
            iso = d[1].lower()
            if iso.startswith("zh"):
                iso = "zh"
            if iso in probs:
                probs[iso] += d[2] / 100.0
    except Exception:
        pass
    return probs


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
# Source: log_combined_3.txt (strict scoring, no fallbacks)
WEIGHTS = {
    "lingua":      0.8503,   # lingua-high AUC (log_combined_3.txt)
    "langdetect":  0.7516,   # corrected AUC — proxy rule removed (was 0.7593)
    "pycld2":      0.9634,   # pycld2 AUC (unchanged)
}


# ==============================================================================
# VOTING STRATEGIES
# ==============================================================================
def hard_vote(lingua_pred, ld_pred, cld2_pred):
    """Majority vote on raw predicted labels. Ties go to lingua-high."""
    votes = defaultdict(int)
    for pred in [lingua_pred, ld_pred, cld2_pred]:
        if pred and pred != "unknown":
            votes[pred] += 1
    if not votes:
        return "unknown"
    max_votes = max(votes.values())
    winners = [lang for lang, v in votes.items() if v == max_votes]
    # tie → trust lingua-high
    return lingua_pred if lingua_pred in winners else winners[0]


def soft_vote(lingua_p, ld_p, cld2_p):
    """Average probability across all 3 models, pick highest."""
    combined = {l: (lingua_p.get(l, 0.0) + ld_p.get(l, 0.0) + cld2_p.get(l, 0.0)) / 3
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)


def weighted_vote(lingua_p, ld_p, cld2_p):
    """Weighted sum of probabilities by AUC weight, pick highest."""
    combined = {}
    for l in TARGET_LANGS:
        combined[l] = (
            lingua_p.get(l, 0.0)  * WEIGHTS["lingua"]     +
            ld_p.get(l, 0.0)      * WEIGHTS["langdetect"] +
            cld2_p.get(l, 0.0)    * WEIGHTS["pycld2"]
        )
    return max(combined, key=combined.get)


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
    lingua_p  = lingua_probs(text)
    ld_p      = langdetect_probs(text)
    cld2_p    = pycld2_probs(text)

    lingua_pred = max(lingua_p, key=lingua_p.get) if any(lingua_p.values()) else "unknown"
    ld_pred     = max(ld_p,     key=ld_p.get)     if any(ld_p.values())     else "unknown"
    cld2_pred   = max(cld2_p,   key=cld2_p.get)   if any(cld2_p.values())   else "unknown"

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
print(f"Voting weights used  —  lingua-high: {WEIGHTS['lingua']}  |  "
      f"langdetect: {WEIGHTS['langdetect']}  |  pycld2: {WEIGHTS['pycld2']}")
print(f"(Weights derived from ROC AUC scores in benchmarkV5)")
print(f"\nLog saved to: {LOG_PATH}")
