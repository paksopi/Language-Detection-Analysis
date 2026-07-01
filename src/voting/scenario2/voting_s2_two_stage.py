"""
voting_scenario2/voting_s2_two_stage.py — Two-stage voting for Scenario 2.

Scenario 2 trio: lingua-high + openlid-v3 + pycld2 (no langdetect).

Stage 1: collapse ms/id -> MSID, hard vote on {en, MSID} with all 3 S2 models.
Stage 2 variants (triggered if Stage 1 = MSID):
  s2_ts_agree:    lingua + pycld2 agree on ms/id; confidence tiebreaker (mirrors S1).
  s2_ts_weighted: per-class dev-accuracy weighted vote (lingua + pycld2).
  s2_ts_3agree:   all 3 S2 models vote on ms/id.

Writes: voting_scenario2/log_s2_two_stage_N.txt
"""

import sys
import regex
import numpy as np
from collections import defaultdict, Counter

import fasttext
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from voting.core import (
    ROOT, TARGET_LANGS, LANGUAGE_ORDER, BUCKET_ORDER, EXPECTED_TO_ISO, next_path,
    load_dataset, load_lingua, lingua_probs, pycld2_probs, langdetect_probs,
    run_predictions as run_s1_predictions,
)

OUT_DIR  = ROOT / "results" / "logs_scenario2" / "test_case_7_enmyid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "final_report_s2_two_stage.txt"
SRC_DIR  = ROOT / "models"
DS_DIR   = ROOT / "data"

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)
SEP = "=" * 90

OPENLID_TO_ISO = {
    "eng_Latn": "en", "zsm_Latn": "ms", "msa_Latn": "ms",
    "ind_Latn": "id", "zho_Hans": "zh", "zho_Hant": "zh",
    "cmn_Hans": "zh", "cmn_Hant": "zh", "tam_Taml": "ta",
}
_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACES  = regex.compile(r"\s\s+")

def preprocess_openlid(text):
    text = text.strip().replace("\n", " ").lower()
    text = _SPACES.sub(" ", text)
    text = _NONWORD.sub("", text)
    return text

def openlid_probs(model, text):
    probs = {l: 0.0 for l in TARGET_LANGS}
    processed = preprocess_openlid(text)
    if not processed.strip():
        return probs
    try:
        labels, scores = model.predict(processed, k=20)
        for label, score in zip(labels, scores):
            code = label.replace("__label__", "")
            iso  = OPENLID_TO_ISO.get(code, "unknown")
            if iso in probs:
                probs[iso] += float(score)
    except Exception:
        pass
    return probs

_CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3), (0x20000, 0x2A6DF)]
def _is_cjk_char(ch):
    cp = ord(ch)
    return any(s <= cp <= e for s, e in _CJK_RANGES)

def bucket_for(text):
    cjk_chars = sum(1 for ch in text if _is_cjk_char(ch))
    words = len(text.split()) + cjk_chars
    if words <= 1:  return "1 word"
    if words == 2:  return "2 words"
    if words <= 7:  return "3-7 words"
    if words <= 16: return "8-16 words"
    return "17-50 words"

print(f"Log: {LOG_PATH}")
print("Loading models...")
detector      = load_lingua()
openlid_model = fasttext.load_model(str(SRC_DIR / "openlid-v3.bin"))
print("  lingua-high, openlid-v3, pycld2 ready.\n")

cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} cases.")

# ── Run S2 predictions (lingua + openlid + pycld2) ────────────────────────────
print("Running S2 predictions...")
s2_all = []
for case in cases:
    text = case["text"]
    lp   = lingua_probs(detector, text)
    op   = openlid_probs(openlid_model, text)
    cp   = pycld2_probs(text)
    l_pred = max(lp, key=lp.get) if any(lp.values()) else "unknown"
    o_pred = max(op, key=op.get) if any(op.values()) else "unknown"
    c_pred = max(cp, key=cp.get) if any(cp.values()) else "unknown"
    s2_all.append({
        "expected_iso": case["expected_iso"],
        "expected_lbl": case["expected_lbl"],
        "bucket":       bucket_for(text),
        "lingua_pred":  l_pred, "lingua_p":  lp,
        "openlid_pred": o_pred, "openlid_p": op,
        "cld2_pred":    c_pred, "cld2_p":    cp,
    })

# ── Run S1 predictions (lingua + langdetect + pycld2) for comparison ──────────
print("Running S1 predictions for McNemar comparison...")
s1_all = run_s1_predictions(cases, detector)
print("Done.\n")

# ── Dev/test split (same seed=42 as voting_two_stage.py) ─────────────────────
labels_for_split = [r["expected_lbl"] for r in s2_all]
dev_idx, test_idx = train_test_split(
    range(len(s2_all)), test_size=0.4,
    stratify=labels_for_split, random_state=42,
)
s2_dev   = [s2_all[i] for i in dev_idx]
s2_test  = [s2_all[i] for i in test_idx]
s1_test  = [s1_all[i] for i in test_idx]
print(f"Split: dev={len(s2_dev)}, test={len(s2_test)}\n")

# ── Stage 1 ───────────────────────────────────────────────────────────────────
MSID = "msid"

def collapse_msid(pred):
    return MSID if pred in ("ms", "id") else pred

def stage1_s2(r):
    """Majority vote (lingua+openlid+pycld2) on collapsed {en,MSID,zh,ta}; lingua ties."""
    p1 = collapse_msid(r["lingua_pred"])
    p2 = collapse_msid(r["openlid_pred"])
    p3 = collapse_msid(r["cld2_pred"])
    votes = Counter(p for p in [p1, p2, p3] if p != "unknown")
    if not votes: return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return p1 if p1 in winners else winners[0]

# ── Stage 2 variants ──────────────────────────────────────────────────────────
def s2_agree_lc(r):
    """lingua + pycld2 only — mirrors S1 Stage 2."""
    lp, cp = r["lingua_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    if l == c: return l
    lc = max(lp.get("ms", 0), lp.get("id", 0))
    cc = max(cp.get("ms", 0), cp.get("id", 0))
    return l if lc >= cc else c

def s2_3agree(r):
    """All 3 S2 models vote on ms/id."""
    lp, op, cp = r["lingua_p"], r["openlid_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    o = "ms" if op.get("ms", 0) >= op.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    votes = Counter([l, o, c])
    if votes["ms"] != votes["id"]:
        return max(votes, key=votes.get)
    confs = [(max(lp.get("ms", 0), lp.get("id", 0)), l),
             (max(op.get("ms", 0), op.get("id", 0)), o),
             (max(cp.get("ms", 0), cp.get("id", 0)), c)]
    return max(confs, key=lambda x: x[0])[1]

def fit_weighted(dev_results):
    """Per-class dev-accuracy weights for lingua + pycld2."""
    acc = {m: {iso: [0, 0] for iso in ("ms", "id")} for m in ("l", "c")}
    for r in dev_results:
        exp = r["expected_iso"]
        if exp not in ("ms", "id"): continue
        lp, cp = r["lingua_p"], r["cld2_p"]
        for m, p in [("l", lp), ("c", cp)]:
            pred = "ms" if p.get("ms", 0) >= p.get("id", 0) else "id"
            acc[m][exp][1] += 1
            acc[m][exp][0] += int(pred == exp)
    w = {m: {iso: acc[m][iso][0] / max(acc[m][iso][1], 1) for iso in ("ms", "id")}
         for m in ("l", "c")}
    print(f"  [Dev-fitted weights] w_lingua_ms={w['l']['ms']:.4f}  w_lingua_id={w['l']['id']:.4f}")
    print(f"                       w_cld2_ms={w['c']['ms']:.4f}    w_cld2_id={w['c']['id']:.4f}")
    def _fn(r):
        lp, cp = r["lingua_p"], r["cld2_p"]
        ms_score = w["l"]["ms"] * lp.get("ms", 0) + w["c"]["ms"] * cp.get("ms", 0)
        id_score = w["l"]["id"] * lp.get("id", 0) + w["c"]["id"] * cp.get("id", 0)
        return "ms" if ms_score >= id_score else "id"
    return _fn

print("Fitting Stage 2 weighted from dev set...")
stage2_weighted = fit_weighted(s2_dev)
print()

# ── S1 Stage 1+2 for comparison ───────────────────────────────────────────────
def s1_ts_agree(r):
    p1 = collapse_msid(r["lingua_pred"]); p2 = collapse_msid(r["ld_pred"]); p3 = collapse_msid(r["cld2_pred"])
    votes = Counter(p for p in [p1, p2, p3] if p != "unknown")
    if not votes: return "unknown"
    top_v = max(votes.values()); winners = [l for l, v in votes.items() if v == top_v]
    s1 = p1 if p1 in winners else winners[0]
    if s1 != MSID: return s1
    lp, cp = r["lingua_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    if l == c: return l
    lc = max(lp.get("ms", 0), lp.get("id", 0)); cc = max(cp.get("ms", 0), cp.get("id", 0))
    return l if lc >= cc else c

# ── Generate all predictions on test set ──────────────────────────────────────
METHODS = [
    ("s2_ts_agree",    "S2 ts_agree   (L+C Stage2)"),
    ("s2_ts_3agree",   "S2 ts_3agree  (L+O+C Stage2)"),
    ("s2_ts_weighted", "S2 ts_weighted (weighted)"),
    ("s2_hard",        "S2 hard (baseline)"),
]

def s2_hard_pred(r):
    votes = Counter(p for p in [r["lingua_pred"], r["openlid_pred"], r["cld2_pred"]] if p != "unknown")
    if not votes: return "unknown"
    top_v = max(votes.values()); winners = [l for l, v in votes.items() if v == top_v]
    return r["lingua_pred"] if r["lingua_pred"] in winners else winners[0]

for r in s2_test:
    s1 = stage1_s2(r)
    r["s2_ts_agree"]    = s1 if s1 != MSID else s2_agree_lc(r)
    r["s2_ts_3agree"]   = s1 if s1 != MSID else s2_3agree(r)
    r["s2_ts_weighted"] = s1 if s1 != MSID else stage2_weighted(r)
    r["s2_hard"]        = s2_hard_pred(r)

for r in s1_test:
    r["s1_ts_agree"] = s1_ts_agree(r)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Accuracy
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print(f"S2 TWO-STAGE — TEST SET ACCURACY (n={len(s2_test)})")
print(SEP)
print(f"  {'Method':<30} | " + " | ".join(f"{l:>6}" for l in LANGUAGE_ORDER) + " | ALL")
print("  " + "-" * (35 + 11 * len(LANGUAGE_ORDER)))
for mk, mname in METHODS:
    accs = []; totals = [0, 0]
    for lbl in LANGUAGE_ORDER:
        iso  = EXPECTED_TO_ISO[lbl]
        rows = [r for r in s2_test if r["expected_lbl"] == lbl]
        n = len(rows); c = sum(1 for r in rows if r[mk] == iso)
        accs.append(f"{c/n*100:>5.1f}%"); totals[0] += c; totals[1] += n
    accs.append(f"{totals[0]/totals[1]*100:>5.1f}%")
    print(f"  {mname:<30} | " + " | ".join(accs))

# S1 two_stage_agree for reference
s1_ref_accs = []; s1_totals = [0, 0]
for lbl in LANGUAGE_ORDER:
    iso  = EXPECTED_TO_ISO[lbl]
    rows = [r for r in s1_test if r["expected_lbl"] == lbl]
    n = len(rows); c = sum(1 for r in rows if r["s1_ts_agree"] == iso)
    s1_ref_accs.append(f"{c/n*100:>5.1f}%"); s1_totals[0] += c; s1_totals[1] += n
s1_ref_accs.append(f"{s1_totals[0]/s1_totals[1]*100:>5.1f}%")
print(f"  {'S1 two_stage_agree (ref)':<30} | " + " | ".join(s1_ref_accs))

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Bucket breakdown for MY and ID
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ACCURACY BY BUCKET — MY and ID")
print(SEP)
for focus_lbl in ["MY", "ID"]:
    iso = EXPECTED_TO_ISO[focus_lbl]
    print(f"\n  {focus_lbl}:")
    print(f"  {'Bucket':<14} | {'n':>4} | " + " | ".join(f"{mk:>18}" for mk, _ in METHODS))
    for bucket in BUCKET_ORDER:
        rows = [r for r in s2_test if r["expected_lbl"] == focus_lbl and r["bucket"] == bucket]
        n = len(rows)
        if n == 0: continue
        vals = [f"{sum(1 for r in rows if r[mk]==iso)/n*100:>17.1f}%" for mk, _ in METHODS]
        print(f"  {bucket:<14} | {n:>4} | " + " | ".join(vals))

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — McNemar tests: each S2 method vs S1 two_stage_agree
# Bonferroni: 4 methods x 3 scopes = 12, but per-scope family = 4, alpha=0.05/4=0.0125
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("McNEMAR TESTS — S2 METHODS vs S1 two_stage_agree")
print("Scope-corrected Bonferroni: alpha_corr = 0.05/4 = 0.0125 per scope")
print(SEP)
ALPHA_CORR = 0.0125
print(f"  {'Scope':<5} | {'Method':<30} | {'b':>4} | {'c':>4} | {'p-value':>9} | {'sig':>4}")
print("  " + "-" * 65)

for scope in ["ALL", "MY", "ID"]:
    rows_s2 = s2_test if scope == "ALL" else [r for r in s2_test if r["expected_lbl"] == scope]
    rows_s1 = s1_test if scope == "ALL" else [r for r in s1_test if r["expected_lbl"] == scope]
    n = len(rows_s2)
    for mk, mname in METHODS:
        b = sum(1 for r2, r1 in zip(rows_s2, rows_s1)
                if r2[mk] == r2["expected_iso"] and r1["s1_ts_agree"] != r1["expected_iso"])
        c = sum(1 for r2, r1 in zip(rows_s2, rows_s1)
                if r2[mk] != r2["expected_iso"] and r1["s1_ts_agree"] == r1["expected_iso"])
        try:
            res = mcnemar([[0, b],[c, 0]], exact=(b+c)<25, correction=(b+c)>=25)
            p = res.pvalue
        except Exception:
            p = float("nan")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA_CORR else "ns"
        print(f"  {scope:<5} | {mname:<30} | {b:>4} | {c:>4} | {p:>9.4f} | {sig:>4}")

print(f"\nLog saved: {LOG_PATH}")
