"""
voting/kappa_s2.py — Inter-model diversity for Scenario 2 trio.

Computes Cohen's kappa + pairwise agreement for:
  lingua-high vs openlid-v3, openlid-v3 vs pycld2, lingua-high vs pycld2
Focuses on MY/ID to check if openlid-v3 is genuinely diverse from the others
on the ms/id axis (i.e. not just a softer repeat of langdetect's structural bias).

Writes: log/log_kappa_s2_N.txt
"""

import sys
import regex
from pathlib import Path
from collections import defaultdict

import fasttext
from sklearn.metrics import cohen_kappa_score
from langdetect import DetectorFactory
DetectorFactory.seed = 0

from voting.core import (
    ROOT, LOG_DIR, DS_DIR, LANGUAGE_ORDER, TARGET_LANGS, next_path,
    load_dataset, load_lingua, lingua_probs, pycld2_probs,
)

LOG_PATH = next_path(LOG_DIR, "log_kappa_s2", ".txt")

class Logger:
    def __init__(self, path):
        self.terminal = sys.__stdout__
        self.log = open(path, "w", encoding="utf-8")
    def write(self, m):  self.terminal.write(m); self.log.write(m)
    def flush(self):     self.terminal.flush();  self.log.flush()

sys.stdout = Logger(LOG_PATH)

SEP = "=" * 85
ROOT_PATH = Path(__file__).parent.parent.parent
SRC_DIR   = ROOT_PATH / "models"

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

print(f"Log: {LOG_PATH}")
print("Loading models...")
detector      = load_lingua()
openlid_model = fasttext.load_model(str(SRC_DIR / "openlid-v3.bin"))
print("  lingua-high, openlid-v3, pycld2 ready.\n")

cases = load_dataset(DS_DIR / "test_case_7_enmyid.txt")
print(f"Loaded {len(cases)} cases. Running predictions...")

results = []
for case in cases:
    text = case["text"]
    lp = lingua_probs(detector, text)
    op = openlid_probs(openlid_model, text)
    cp = pycld2_probs(text)
    l_pred = max(lp, key=lp.get) if any(lp.values()) else "unknown"
    o_pred = max(op, key=op.get) if any(op.values()) else "unknown"
    c_pred = max(cp, key=cp.get) if any(cp.values()) else "unknown"
    results.append({
        "expected_iso": case["expected_iso"],
        "expected_lbl": case["expected_lbl"],
        "lingua_pred":  l_pred,
        "openlid_pred": o_pred,
        "cld2_pred":    c_pred,
    })
print("Done.\n")

# ── Kappa + agreement ──────────────────────────────────────────────────────────
PAIRS = [
    ("lingua_pred",  "openlid_pred", "lingua-high vs openlid-v3"),
    ("openlid_pred", "cld2_pred",    "openlid-v3  vs pycld2"),
    ("lingua_pred",  "cld2_pred",    "lingua-high vs pycld2 (ref)"),
]

print(SEP)
print("SCENARIO 2 TRIO — COHEN'S KAPPA + PAIRWISE AGREEMENT")
print("(lingua-high + openlid-v3 + pycld2)")
print(SEP)
print("""
Library: sklearn.metrics.cohen_kappa_score
Reference for MY: κ < 0 means the pair agrees less than chance (structural artifact).
""")

for scope in LANGUAGE_ORDER + ["ALL"]:
    lbl  = None if scope == "ALL" else scope
    rows = [r for r in results if r["expected_lbl"] == lbl] if lbl else results
    n    = len(rows)
    print(f"-- {scope}  (n={n}) " + "-" * 55)
    print(f"  {'Pair':<32} {'kappa':>7}  {'agree %':>8}  note")
    for ka, kb, label in PAIRS:
        preds_a = [r[ka] for r in rows]
        preds_b = [r[kb] for r in rows]
        kappa   = cohen_kappa_score(preds_a, preds_b)
        agree   = sum(a == b for a, b in zip(preds_a, preds_b)) / n * 100
        note    = ""
        if scope == "MY" and ("openlid" in ka or "openlid" in kb):
            # Check if openlid outputs id vs ms on MY cases
            ol_preds = [r["openlid_pred"] for r in rows]
            n_id_for_my = sum(1 for p in ol_preds if p == "id")
            n_ms_for_my = sum(1 for p in ol_preds if p == "ms")
            note = f"openlid: {n_ms_for_my} ms, {n_id_for_my} id for true MY"
        print(f"  {label:<32} {kappa:>7.4f}  {agree:>7.1f}%  {note}")
    print()

# Per-language prediction distribution for openlid-v3
print(SEP)
print("OPENLID-V3 PREDICTION DISTRIBUTION PER TRUE LANGUAGE")
print("(shows whether errors are structurally biased like langdetect's id-for-ms)")
print(SEP)
print(f"\n  {'True lang':<10} | " + " | ".join(f"pred={l:>4}" for l in TARGET_LANGS) + " | pred=unk")
print("  " + "-" * 75)
for lbl in LANGUAGE_ORDER:
    rows = [r for r in results if r["expected_lbl"] == lbl]
    n = len(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[r["openlid_pred"]] += 1
    row_str = " | ".join(f"{counts.get(l, 0):>8}" for l in TARGET_LANGS)
    unk = counts.get("unknown", 0)
    print(f"  {lbl:<10} | {row_str} | {unk:>7}")

# S1 kappa comparison
print(f"""
COMPARISON WITH SCENARIO 1 (lingua-high + langdetect + pycld2):
  lingua vs langdetect (MY): kappa = -0.0191, agreement 18.1%  (see voting_stats.py Objective 3)
    -> langdetect has 0 ms predictions for true MY (fully structural)
  lingua vs openlid   (MY): see table above
    -> if kappa > -0.05 and openlid outputs some ms, it is a better voter

KEY QUESTION: Does openlid-v3 bring genuine diversity on the ms/id axis?
  - If it outputs ms for at least some true-MY cases AND its errors
    differ from lingua-high's errors -> YES, genuine diversity.
  - If it always outputs id for true-MY -> same structural problem as langdetect.
""")
print(f"Log saved: {LOG_PATH}")
