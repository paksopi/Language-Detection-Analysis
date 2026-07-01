"""
voting/core.py — shared utilities for the voting evaluation suite.
Importable with no side effects (no stdout redirect, no arg parsing).
All model loading is explicit via load_lingua().
"""

from pathlib import Path
from collections import defaultdict
import re
import numpy as np

# Reproducibility: set seed BEFORE any langdetect import or call
from langdetect import DetectorFactory
DetectorFactory.seed = 0
import langdetect as _ld

import pycld2
from lingua import Language, LanguageDetectorBuilder

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
LOG_DIR   = ROOT / "results" / "logs" / "test_case_7_enmyid"
DS_DIR    = ROOT / "data"
CALIB_DIR = ROOT / "results" / "calibration"
for _d in (LOG_DIR, CALIB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_LANGS    = ["en", "ms", "id"]
LANGUAGE_ORDER  = ["EN", "MY", "ID"]
EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id"}
BUCKET_ORDER    = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]

LINGUA_LANGS = (
    Language.ENGLISH, Language.MALAY, Language.INDONESIAN,
)
_LINGUA_ISO = {
    Language.ENGLISH:    "en",
    Language.MALAY:      "ms",
    Language.INDONESIAN: "id",
}

# AUC-derived voting weights (log_combined_3.txt, strict scoring)
DEFAULT_WEIGHTS = {
    "lingua":     0.8503,
    "langdetect": 0.7516,   # corrected — id-as-ms proxy rule removed
    "pycld2":     0.9634,
}

IND_KEYS  = ["lingua_pred", "ld_pred", "cld2_pred"]
VOTE_KEYS = ["hard", "soft", "weighted"]
ALL_KEYS  = IND_KEYS + VOTE_KEYS
KEY_NAMES = {
    "lingua_pred": "lingua-high",
    "ld_pred":     "langdetect",
    "cld2_pred":   "pycld2",
    "hard":        "hard",
    "soft":        "soft",
    "weighted":    "weighted",
}

# ── Utilities ──────────────────────────────────────────────────────────────────
def next_path(directory: Path, stem: str, suffix: str) -> Path:
    n = 1
    while (directory / f"{stem}_{n}{suffix}").exists():
        n += 1
    return directory / f"{stem}_{n}{suffix}"

# ── Bucketing ──────────────────────────────────────────────────────────────────
CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3)]

def is_cjk(text: str) -> bool:
    cjk = sum(1 for ch in text if any(s <= ord(ch) <= e for s, e in CJK_RANGES))
    return cjk > 0 and cjk >= len(text.replace(" ", "")) * 0.3

def bucket_for(text: str) -> str:
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

# ── Dataset ────────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
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
    return cases

# ── Model loading ──────────────────────────────────────────────────────────────
def load_lingua():
    return LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()

# ── Per-model probability functions ───────────────────────────────────────────
def lingua_probs(detector, text: str) -> dict:
    """Return {iso: confidence} for all 5 target langs from lingua-high."""
    confs = detector.compute_language_confidence_values(text)
    return {_LINGUA_ISO[c.language]: c.value for c in confs if c.language in _LINGUA_ISO}

def langdetect_probs(text: str) -> dict:
    """Return {iso: probability} from langdetect. No 'ms' profile → Malay maps to 'id'."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    try:
        for item in _ld.detect_langs(text):
            iso = item.lang
            if iso.startswith("zh"):
                iso = "zh"
            if iso in probs:
                probs[iso] += item.prob
    except Exception:
        pass
    return probs

def pycld2_probs(text: str) -> dict:
    """Return {iso: confidence} from CLD2 percent scores."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    try:
        _, _, details = pycld2.detect(text)
        for d in details:
            iso = d[1].lower()
            if iso.startswith("zh"):
                iso = "zh"
            if iso in probs:
                probs[iso] += d[2] / 100.0
    except Exception:
        pass
    return probs

def pick_top(probs: dict) -> str:
    """Return the highest-probability language, or 'unknown' if all scores are zero."""
    return max(probs, key=probs.get) if any(probs.values()) else "unknown"

# ── Voting strategies ──────────────────────────────────────────────────────────
def hard_vote(lingua_pred, ld_pred, cld2_pred):
    """Majority vote; ties broken by lingua-high."""
    votes = defaultdict(int)
    for p in [lingua_pred, ld_pred, cld2_pred]:
        if p and p != "unknown":
            votes[p] += 1
    if not votes:
        return "unknown"
    top = max(votes.values())
    winners = [l for l, v in votes.items() if v == top]
    return lingua_pred if lingua_pred in winners else winners[0]

def soft_vote(lingua_p, ld_p, cld2_p):
    """Average probability across 3 models; pick max."""
    combined = {l: (lingua_p.get(l, 0.0) + ld_p.get(l, 0.0) + cld2_p.get(l, 0.0)) / 3
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)

def weighted_vote(lingua_p, ld_p, cld2_p, weights=None):
    """AUC-weighted probability sum; pick max."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    combined = {
        l: lingua_p.get(l, 0.0) * weights["lingua"] +
           ld_p.get(l, 0.0)     * weights["langdetect"] +
           cld2_p.get(l, 0.0)   * weights["pycld2"]
        for l in TARGET_LANGS
    }
    return max(combined, key=combined.get)

def soft_vote_combined(lingua_p, ld_p, cld2_p) -> dict:
    """Return the combined probability dict (not just argmax) for soft voting."""
    return {l: (lingua_p.get(l, 0.0) + ld_p.get(l, 0.0) + cld2_p.get(l, 0.0)) / 3
            for l in TARGET_LANGS}

def weighted_vote_combined(lingua_p, ld_p, cld2_p, weights=None) -> dict:
    """Return the weighted probability dict (not just argmax)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    return {
        l: lingua_p.get(l, 0.0) * weights["lingua"] +
           ld_p.get(l, 0.0)     * weights["langdetect"] +
           cld2_p.get(l, 0.0)   * weights["pycld2"]
        for l in TARGET_LANGS
    }

# ── Run all predictions ────────────────────────────────────────────────────────
def run_predictions(cases: list, detector) -> list:
    """
    Run all 3 models + 3 voting strategies on every case.
    Returns list of result dicts. Each dict has:
      expected_iso, expected_lbl, bucket,
      lingua_pred, lingua_conf, lingua_p,
      ld_pred,     ld_conf,     ld_p,
      cld2_pred,   cld2_conf,   cld2_p,
      hard, soft, weighted
    """
    results = []
    for case in cases:
        text = case["text"]
        lp   = lingua_probs(detector, text)
        dp   = langdetect_probs(text)
        cp   = pycld2_probs(text)
        l_pred = pick_top(lp)
        d_pred = pick_top(dp)
        c_pred = pick_top(cp)
        results.append({
            "expected_iso": case["expected_iso"],
            "expected_lbl": case["expected_lbl"],
            "bucket":       case["bucket"],
            "lingua_pred":  l_pred,
            "lingua_conf":  lp.get(l_pred, 0.0) if l_pred != "unknown" else 0.0,
            "lingua_p":     lp,
            "ld_pred":      d_pred,
            "ld_conf":      dp.get(d_pred, 0.0) if d_pred != "unknown" else 0.0,
            "ld_p":         dp,
            "cld2_pred":    c_pred,
            "cld2_conf":    cp.get(c_pred, 0.0) if c_pred != "unknown" else 0.0,
            "cld2_p":       cp,
            "hard":         hard_vote(l_pred, d_pred, c_pred),
            "soft":         soft_vote(lp, dp, cp),
            "weighted":     weighted_vote(lp, dp, cp),
        })
    return results

# ── Accuracy helpers ───────────────────────────────────────────────────────────
def binary_correct(results: list, key: str, lang_filter=None) -> np.ndarray:
    """Binary array (1=correct, 0=wrong), optionally filtered to one language label."""
    rows = [r for r in results if r["expected_lbl"] == lang_filter] if lang_filter else results
    return np.array([int(r[key] == r["expected_iso"]) for r in rows])

def pred_labels(results: list, key: str, lang_filter=None) -> list:
    rows = [r for r in results if r["expected_lbl"] == lang_filter] if lang_filter else results
    return [r[key] for r in rows]

def overall_accuracy(results: list, key: str) -> float:
    return sum(1 for r in results if r[key] == r["expected_iso"]) / len(results) if results else 0.0

def accuracy_by_lang(results: list, key: str) -> dict:
    """Return {lbl: (correct, total)}."""
    by_lang = defaultdict(lambda: [0, 0])
    for r in results:
        lbl = r["expected_lbl"]
        by_lang[lbl][1] += 1
        by_lang[lbl][0] += int(r[key] == r["expected_iso"])
    return dict(by_lang)

def pct(correct, total):
    return f"{correct/total*100:5.1f}%" if total else "  N/A "

def print_accuracy_table(results: list, keys: list, names: list, title: str):
    print(f"\n{title}")
    col_w = 13
    hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{n:>{col_w}}" for n in names)
    print(hdr)
    print("-" * len(hdr))
    for lbl in LANGUAGE_ORDER:
        rows = [r for r in results if r["expected_lbl"] == lbl]
        n = len(rows)
        if n == 0:
            continue
        parts = []
        for k in keys:
            c = sum(1 for r in rows if r[k] == r["expected_iso"])
            parts.append(f"{pct(c, n):>{col_w}}")
        print(f"  {lbl:<5} | {n:>3} | " + " | ".join(parts))
    print("-" * len(hdr))
    n_all = len(results)
    parts = []
    for k in keys:
        c = sum(1 for r in results if r[k] == r["expected_iso"])
        parts.append(f"{pct(c, n_all):>{col_w}}")
    print(f"  {'ALL':<5} | {n_all:>3} | " + " | ".join(parts))
