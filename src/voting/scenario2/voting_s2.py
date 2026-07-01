"""
voting_scenario2/voting_s2.py
Scenario 2: replace langdetect with openlid-v3.
Ensemble: lingua-high + openlid-v3 + pycld2

Compares against Scenario 1 (lingua-high + langdetect + pycld2) inline.
Writes all output to voting_scenario2/log_s2_N.txt
"""

import sys
import re
import regex
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

import fasttext
import pycld2
from lingua import Language, LanguageDetectorBuilder
import langdetect as _ld
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent.parent
OUT_DIR    = ROOT / "results" / "logs_scenario2" / "test_case_7_enmyid"
SRC_DIR    = ROOT / "models"
DS_DIR     = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def next_path(directory, stem, suffix):
    n = 1
    while (directory / f"{stem}_{n}{suffix}").exists():
        n += 1
    return directory / f"{stem}_{n}{suffix}"

LOG_PATH = OUT_DIR / "final_report_s2.txt"

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_LANGS    = ["en", "ms", "id"]
LANGUAGE_ORDER  = ["EN", "MY", "ID"]
EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id"}
BUCKET_ORDER    = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]
LINGUA_LANGS    = (Language.ENGLISH, Language.MALAY, Language.INDONESIAN)
_LINGUA_ISO     = {
    Language.ENGLISH:    "en",
    Language.MALAY:      "ms",
    Language.INDONESIAN: "id",
}

# OpenLID FLORES-200 -> ISO mapping (strict — only confirmed mappings)
OPENLID_TO_ISO = {
    "eng_Latn": "en",
    "zsm_Latn": "ms",
    "msa_Latn": "ms",
    "ind_Latn": "id",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "cmn_Hans": "zh",   # Mandarin Chinese (simplified) — FLORES-200 variant
    "cmn_Hant": "zh",
    "tam_Taml": "ta",
}

# AUC weights (from log_combined_3.txt strict scoring)
# Scenario 1 (for comparison)
S1_WEIGHTS = {"lingua": 0.8503, "model2": 0.7516, "pycld2": 0.9634}
# Scenario 2
S2_WEIGHTS = {"lingua": 0.8503, "model2": 0.7097, "pycld2": 0.9634}

SEP = "=" * 105

print(f"Log: {LOG_PATH}")
print("Scenario 2: lingua-high + openlid-v3 + pycld2")
print("Scenario 1: lingua-high + langdetect  + pycld2  (re-run for side-by-side comparison)\n")

# ── Bucketing ──────────────────────────────────────────────────────────────────
CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3)]

def is_cjk(text):
    n = sum(1 for ch in text if any(s <= ord(ch) <= e for s, e in CJK_RANGES))
    return n > 0 and n >= len(text.replace(" ", "")) * 0.3

def bucket_for(text):
    if is_cjk(text):
        n = len(re.sub(r"[^\w]", "", text))
        if n <= 2:    return "1 word"
        elif n <= 6:  return "2 words"
        elif n <= 15: return "3-7 words"
        elif n <= 48: return "8-16 words"
        else:         return "17-50 words"
    else:
        clean = re.sub(r"[^\w\s஀-௿]", "", text)
        n = len(clean.split())
        if n <= 1:    return "1 word"
        elif n <= 2:  return "2 words"
        elif n <= 7:  return "3-7 words"
        elif n <= 16: return "8-16 words"
        else:         return "17-50 words"

# ── Load dataset ───────────────────────────────────────────────────────────────
cases = []
with open(DS_DIR / "test_case_7_enmyid.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, text = line.split("|", 1)
            label = label.strip().upper()
            if label in EXPECTED_TO_ISO:
                cases.append({
                    "expected_lbl": label,
                    "expected_iso": EXPECTED_TO_ISO[label],
                    "text":         text.strip(),
                    "bucket":       bucket_for(text.strip()),
                })
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

# ── OpenLID preprocessor ───────────────────────────────────────────────────────
_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACES  = regex.compile(r"\s\s+")

def preprocess_openlid(text):
    text = text.strip().replace("\n", " ").lower()
    text = _SPACES.sub(" ", text)
    text = _NONWORD.sub("", text)
    return text

def openlid_label_to_iso(label):
    code = label.replace("__label__", "")
    return OPENLID_TO_ISO.get(code, "unknown")

# ── Probability functions ──────────────────────────────────────────────────────
def lingua_probs(text):
    confs = detector.compute_language_confidence_values(text)
    return {_LINGUA_ISO[c.language]: c.value for c in confs if c.language in _LINGUA_ISO}

def openlid_probs(text):
    """Return {iso: prob} from openlid-v3. Uses top-20 FLORES-200 predictions."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    processed = preprocess_openlid(text)
    if not processed.strip():
        return probs
    try:
        labels, scores = openlid_model.predict(processed, k=20)
        for label, score in zip(labels, scores):
            iso = openlid_label_to_iso(label)
            if iso in probs:
                probs[iso] += float(score)
    except Exception:
        pass
    return probs

def langdetect_probs(text):
    probs = {l: 0.0 for l in TARGET_LANGS}
    try:
        for item in _ld.detect_langs(text):
            iso = item.lang
            if iso.startswith("zh"): iso = "zh"
            if iso in probs: probs[iso] += item.prob
    except Exception:
        pass
    return probs

def pycld2_probs(text):
    probs = {l: 0.0 for l in TARGET_LANGS}
    try:
        _, _, details = pycld2.detect(text)
        for d in details:
            iso = d[1].lower()
            if iso.startswith("zh"): iso = "zh"
            if iso in probs: probs[iso] += d[2] / 100.0
    except Exception:
        pass
    return probs

def _top(probs):
    return max(probs, key=probs.get) if any(probs.values()) else "unknown"

# ── Voting strategies (generic — works for both scenarios) ─────────────────────
def hard_vote_3(p1, p2, p3, tiebreak_pred):
    votes = defaultdict(int)
    for p in [p1, p2, p3]:
        if p and p != "unknown": votes[p] += 1
    if not votes: return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return tiebreak_pred if tiebreak_pred in winners else winners[0]

def soft_vote_3(pa, pb, pc):
    combined = {l: (pa.get(l, 0.0) + pb.get(l, 0.0) + pc.get(l, 0.0)) / 3
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)

def weighted_vote_3(pa, pb, pc, weights):
    combined = {l: pa.get(l, 0.0) * weights["lingua"] +
                   pb.get(l, 0.0) * weights["model2"] +
                   pc.get(l, 0.0) * weights["pycld2"]
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)

# ── Run predictions for both scenarios ────────────────────────────────────────
print("Running predictions for both scenarios (this takes ~5-10 min due to langdetect)...")
s1_results = []   # lingua + langdetect + pycld2
s2_results = []   # lingua + openlid   + pycld2

for i, case in enumerate(cases):
    text = case["text"]
    lp   = lingua_probs(text)
    dp   = langdetect_probs(text)
    op   = openlid_probs(text)
    cp   = pycld2_probs(text)

    l_pred = _top(lp)
    d_pred = _top(dp)
    o_pred = _top(op)
    c_pred = _top(cp)

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
