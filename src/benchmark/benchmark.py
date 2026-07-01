
import os
import sys
import time
import re
import math
import regex
import tracemalloc
from pathlib import Path
from collections import defaultdict
from lingua import Language, LanguageDetectorBuilder
import langdetect
import langid as langid_lib
import fasttext
import pycld2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
ROOT    = Path(__file__).parent.parent.parent
LOG_DIR = ROOT / "results" / "logs" / "test_case_7_enmyid"
SRC_DIR = ROOT / "models"
CM_DIR  = ROOT / "results" / "confusion_matrix"
ROC_DIR = ROOT / "results" / "roc_curve"
DS_DIR  = ROOT / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def next_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return directory/stem_N.suffix where N is the next unused integer."""
    n = 1
    while (directory / f"{stem}_{n}{suffix}").exists():
        n += 1
    return directory / f"{stem}_{n}{suffix}"

LOG_PATH = LOG_DIR / "final_report_benchmark.txt"
CM_PATH  = next_path(CM_DIR,  "confusion_matrix_all", ".png")
ROC_PATH = next_path(ROC_DIR, "roc_curve_all", ".png")

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

TEST_CASE_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else DS_DIR / "test_case_7_enmyid.txt"

if not os.path.exists(TEST_CASE_FILE):
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
        else:
            test_cases.append({"expected": "UNKNOWN", "text": line})

if not test_cases:
    print(f"Error: {TEST_CASE_FILE!r} had no usable lines.")
    sys.exit(1)

EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id"}
total_cases = len(test_cases)
print(f"Loaded {total_cases} test cases from {TEST_CASE_FILE!r}\n")


# ==============================================================================
# WORD-COUNT BUCKETING (CJK & Punctuation Aware)
# ==============================================================================
CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7A3)]

def is_cjk_text(text):
    cjk_count = sum(1 for ch in text if any(s <= ord(ch) <= e for s, e in CJK_RANGES))
    return cjk_count > 0 and cjk_count >= len(text.replace(" ", "")) * 0.3

def bucket_for(text):
    if is_cjk_text(text):
        n = len(re.sub(r'[^\w]', '', text))
        if n <= 2:  return "1 word"
        elif n <= 6:  return "2 words"
        elif n <= 15: return "3-7 words"
        elif n <= 48: return "8-16 words"
        else:         return "17-50 words"
    else:
        clean = re.sub(r'[^\w\s஀-௿]', '', text)
        n = len(clean.split())
        if n <= 1:  return "1 word"
        elif n <= 2:  return "2 words"
        elif n <= 7:  return "3-7 words"
        elif n <= 16: return "8-16 words"
        else:         return "17-50 words"

BUCKET_ORDER   = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]
LANGUAGE_ORDER = ["EN", "MY", "ID"]


# ==============================================================================
# MODEL REGISTRY
# ==============================================================================
# Each entry: key (short id), display name, column header (11 chars max)
MODELS = [
    {"key": "ld",   "name": "langdetect", "hdr": " langdetect"},
    {"key": "low",  "name": "lingua-low",  "hdr": "  lingua-low"},
    {"key": "high", "name": "lingua-high", "hdr": " lingua-high"},
    {"key": "lgid", "name": "langid",      "hdr": "     langid"},
    {"key": "ft",   "name": "fasttext",    "hdr": "    fasttext"},
    {"key": "olid", "name": "openlid-v3",  "hdr": " openlid-v3"},
    {"key": "cld2", "name": "pycld2",      "hdr": "      pycld2"},
]
MODEL_KEYS   = [m["key"]  for m in MODELS]
MODEL_COLORS = ["steelblue", "forestgreen", "darkorange", "purple", "crimson", "saddlebrown", "deeppink"]



# ==============================================================================
# LANGUAGE CODE NORMALISERS
# ==============================================================================
# OpenLID uses FLORES-200 style tags (e.g. eng_Latn, zsm_Latn)
OPENLID_TO_ISO = {
    "eng_Latn": "en",
    "zsm_Latn": "ms",
    "msa_Latn": "ms",
    "ind_Latn": "id",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "tam_Taml": "ta",
}

def openlid_label_to_iso(label):
    code = label.replace("__label__", "")
    return OPENLID_TO_ISO.get(code, "unknown")

# fasttext lid.176.ftz uses __label__<iso>
def ft_label_to_iso(label):
    code = label.replace("__label__", "")
    if code.startswith("zh"): return "zh"
    return code

# pycld2 BCP-47 codes
def cld2_code_to_iso(code):
    code = code.lower()
    if code.startswith("zh"): return "zh"
    return code

# OpenLID preprocessor (required by the model)
_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACES  = regex.compile(r"\s\s+")

def preprocess_openlid(text):
    text = text.strip().replace('\n', ' ').lower()
    text = _SPACES.sub(" ", text)
    text = _NONWORD.sub("", text)
    return text


# ==============================================================================
# 2. BENCHMARK 1: RAM & MEMORY PROFILING
# ==============================================================================
print("=" * 110)
print("BENCHMARK 1: RAM & MEMORY PROFILING")
print("=" * 110)
print("Starting tracemalloc... building detector models in memory.\n")

LANGS = (Language.ENGLISH, Language.MALAY, Language.INDONESIAN)

tracemalloc.start()
detector_high = LanguageDetectorBuilder.from_languages(*LANGS).build()
_, peak_high = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
detector_low = LanguageDetectorBuilder.from_languages(*LANGS).with_low_accuracy_mode().build()
_, peak_low = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  -> Lingua (HIGH accuracy):  {peak_high / (1024*1024):.2f} MB peak RAM to build")
print(f"  -> Lingua (LOW accuracy):   {peak_low  / (1024*1024):.2f} MB peak RAM to build")
print(f"  -> langdetect:   loads profiles lazily — not directly measurable with tracemalloc")
print(f"  -> langid:       pure-Python model loaded at import")
print(f"  -> fasttext/OpenLID: C++ memory not visible to tracemalloc")
print(f"  -> pycld2:       compiled C++ — memory not visible to tracemalloc\n")


# ==============================================================================
# 3. LOAD ALL NON-LINGUA MODELS
# ==============================================================================
print("Loading auxiliary models...")

# fasttext LID (lid.176.ftz — Facebook, 176 languages)
ft_model = fasttext.load_model(str(SRC_DIR / "lid.176.ftz"))

# OpenLID v3 (HPLT, 194 languages, fasttext-based)
openlid_model = fasttext.load_model(str(SRC_DIR / "openlid-v3.bin"))

# langid — restrict to our target languages for a fair comparison
langid_lib.set_languages(['en', 'ms', 'id'])

print("All models loaded.\n")


# ==============================================================================
# 4. BENCHMARK 2: RAW PROCESSING SPEED
# ==============================================================================
print("=" * 110)
print("BENCHMARK 2: RAW PROCESSING SPEED (bucketed by word count, ms per call)")
print("=" * 110)

REPEATS = 100
texts_by_bucket = defaultdict(list)
for case in test_cases:
    texts_by_bucket[bucket_for(case["text"])].append(case["text"])

# Warm up all models to exclude first-call init overhead
_warmup = "hello world"
langdetect.detect(_warmup)
detector_low.detect_language_of(_warmup)
detector_high.detect_language_of(_warmup)
langid_lib.classify(_warmup)
ft_model.predict(_warmup, k=1)
openlid_model.predict(preprocess_openlid(_warmup), k=1)
pycld2.detect(_warmup)

hdr = f"{'Bucket':<14} | {'n':>4} | " + " | ".join(f"{m['hdr']}" for m in MODELS) + " | fastest"
print(hdr)
print("-" * len(hdr))

for bucket in BUCKET_ORDER:
    texts = texts_by_bucket.get(bucket, [])
    if not texts:
        print(f"{bucket:<14} | (no test cases)")
        continue
    total_calls = REPEATS * len(texts)

    def _time(fn):
        t = time.perf_counter()
        for _ in range(REPEATS):
            for tx in texts:
                fn(tx)
        return (time.perf_counter() - t) / total_calls * 1000

    def _ld_safe(tx):
        try: langdetect.detect(tx)
        except Exception: pass

    times = {
        "ld":   _time(_ld_safe),
        "low":  _time(lambda tx: detector_low.detect_language_of(tx)),
        "high": _time(lambda tx: detector_high.detect_language_of(tx)),
        "lgid": _time(lambda tx: langid_lib.classify(tx)),
        "ft":   _time(lambda tx: ft_model.predict(tx.replace('\n', ' '), k=1)),
        "olid": _time(lambda tx: openlid_model.predict(preprocess_openlid(tx), k=1)),
        "cld2": _time(lambda tx: pycld2.detect(tx)),
    }

    # wrap langdetect in try/except during timing
    t = time.perf_counter()
    for _ in range(REPEATS):
        for tx in texts:
            try: langdetect.detect(tx)
            except Exception: pass
    times["ld"] = (time.perf_counter() - t) / total_calls * 1000

    fastest = min(times, key=times.get)
    row = f"{bucket:<14} | {len(texts):>4} | " + " | ".join(f"{times[m['key']]:>11.4f}ms" for m in MODELS) + f" | {MODEL_KEYS[MODEL_KEYS.index(fastest)] if fastest in MODEL_KEYS else fastest}"
    # use model name for fastest
    fastest_name = next(m["name"] for m in MODELS if m["key"] == fastest)
    row = f"{bucket:<14} | {len(texts):>4} | " + " | ".join(f"{times[m['key']]:>11.4f}ms" for m in MODELS) + f" | {fastest_name}"
    print(row)

print()


# ==============================================================================
# 5. BENCHMARK 3: ACCURACY
# ==============================================================================
print("=" * 110)
print("BENCHMARK 3: ACCURACY BY BUCKET (strict exact-match scoring — no fallbacks)")
print("=" * 110)

bucket_lang_stats = defaultdict(lambda: defaultdict(lambda: {k: 0 for k in ["total"] + MODEL_KEYS}))
lang_stats        = defaultdict(lambda: {k: 0 for k in ["total"] + MODEL_KEYS})

# Per-model accumulator for visualizations
y_true = []
viz = {k: {"preds": [], "binary": [], "scores": []} for k in MODEL_KEYS}

for case in test_cases:
    text          = case["text"]
    expected_lbl  = case["expected"]
    if expected_lbl not in EXPECTED_TO_ISO: continue
    expected_iso  = EXPECTED_TO_ISO[expected_lbl]
    bucket        = bucket_for(text)

    # --- langdetect ---
    try:
        ld_langs = langdetect.detect_langs(text)
        ld_iso   = ld_langs[0].lang
        ld_score = ld_langs[0].prob
        if ld_iso.startswith("zh"): ld_iso = "zh"
    except Exception:
        ld_iso, ld_score = "unknown", 0.0

    # --- lingua low ---
    low_lang  = detector_low.detect_language_of(text)
    low_iso   = low_lang.iso_code_639_1.name.lower() if low_lang else "unknown"
    low_confs = detector_low.compute_language_confidence_values(text)
    low_score = low_confs[0].value if low_confs else 0.0

    # --- lingua high ---
    high_lang  = detector_high.detect_language_of(text)
    high_iso   = high_lang.iso_code_639_1.name.lower() if high_lang else "unknown"
    high_confs = detector_high.compute_language_confidence_values(text)
    high_score = high_confs[0].value if high_confs else 0.0

    # --- langid (restricted to 5 langs) ---
    try:
        lgid_lbl, lgid_logprob = langid_lib.classify(text)
        lgid_iso   = lgid_lbl
        lgid_score = math.exp(max(lgid_logprob, -500))  # logprob → [0,1]
    except Exception:
        lgid_iso, lgid_score = "unknown", 0.0

    # --- fasttext lid.176 ---
    try:
        ft_labels, ft_probs = ft_model.predict(text.replace('\n', ' '), k=1)
        ft_iso   = ft_label_to_iso(ft_labels[0])
        ft_score = float(ft_probs[0])
    except Exception:
        ft_iso, ft_score = "unknown", 0.0

    # --- OpenLID v3 ---
    try:
        olid_labels, olid_probs = openlid_model.predict(preprocess_openlid(text), k=1)
        olid_iso   = openlid_label_to_iso(olid_labels[0])
        olid_score = float(olid_probs[0])
    except Exception:
        olid_iso, olid_score = "unknown", 0.0

    # --- pycld2 ---
    try:
        _, _, cld2_details = pycld2.detect(text)
        cld2_iso   = cld2_code_to_iso(cld2_details[0][1])
        cld2_score = cld2_details[0][2] / 100.0
    except Exception:
        cld2_iso, cld2_score = "unknown", 0.0

    results = {
        "ld":   (ld_iso   == expected_iso, ld_iso,   ld_score),
        "low":  (low_iso  == expected_iso, low_iso,  low_score),
        "high": (high_iso == expected_iso, high_iso, high_score),
        "lgid": (lgid_iso == expected_iso, lgid_iso, lgid_score),
        "ft":   (ft_iso   == expected_iso, ft_iso,   ft_score),
        "olid": (olid_iso == expected_iso, olid_iso, olid_score),
        "cld2": (cld2_iso == expected_iso, cld2_iso, cld2_score),
    }

    y_true.append(expected_iso)
    for k, (ok, pred_iso, score) in results.items():
        viz[k]["preds"].append(pred_iso)
        viz[k]["binary"].append(int(ok))
        viz[k]["scores"].append(score)

    bucket_s = bucket_lang_stats[bucket][expected_lbl]
    global_s  = lang_stats[expected_lbl]
    for s in (bucket_s, global_s):
        s["total"] += 1
        for k, (ok, _, _) in results.items():
            s[k] += int(ok)

# --- Print per-bucket accuracy ---
col_hdr = f"  {'LANG':<5} | {'n':>3} | " + " | ".join(f"{m['hdr'].strip():>10}" for m in MODELS)
for bucket in BUCKET_ORDER:
    if bucket not in bucket_lang_stats: continue
    print(f"\nBucket: {bucket}")
    print(col_hdr)
    for lang in LANGUAGE_ORDER:
        if lang not in bucket_lang_stats[bucket]: continue
        s = bucket_lang_stats[bucket][lang]
        n = s["total"]
        row = f"  {lang:<5} | {n:>3} | " + " | ".join(f"{s[m['key']]/n*100:>9.1f}%" for m in MODELS)
        print(row)

# --- Print overall accuracy ---
print("\n" + "=" * 110)
print("BENCHMARK 3b: ACCURACY BY LANGUAGE (all buckets combined)")
print("=" * 110)
print(col_hdr)
print("-" * len(col_hdr))
for lang in LANGUAGE_ORDER:
    if lang not in lang_stats: continue
    s = lang_stats[lang]
    n = s["total"]
    print(f"  {lang:<5} | {n:>3} | " + " | ".join(f"{s[m['key']]/n*100:>9.1f}%" for m in MODELS))

overall_total = sum(s["total"] for s in lang_stats.values())
print("-" * len(col_hdr))
overall_row = f"  {'ALL':<5} | {overall_total:>3} | " + " | ".join(
    f"{sum(s[m['key']] for s in lang_stats.values()) / overall_total * 100:>9.1f}%"
    for m in MODELS
)
print(overall_row)


# ==============================================================================
# 6. BENCHMARK 4: CONFUSION MATRICES & ROC CURVES — ALL MODELS
# ==============================================================================
print("\n" + "=" * 110)
print("BENCHMARK 4: CONFUSION MATRICES & ROC CURVES (all models)")
print("=" * 110)

CM_LABELS = ['en', 'ms', 'id']
N_MODELS  = len(MODELS)

# --- Confusion Matrices (2 x 4 grid, last cell empty) ---
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
axes_flat = axes.flatten()

for i, m in enumerate(MODELS):
    cm   = confusion_matrix(y_true, viz[m["key"]]["preds"], labels=CM_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CM_LABELS)
    disp.plot(cmap=plt.cm.Blues, ax=axes_flat[i], colorbar=False)
    axes_flat[i].set_title(m["name"], fontsize=12)

axes_flat[-1].set_visible(False)  # hide the 8th cell
plt.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(CM_PATH, bbox_inches="tight")
plt.close()
print(f" -> Exported '{CM_PATH}'")

# --- ROC Curves (all models on one plot) ---
plt.figure(figsize=(10, 7))
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random baseline')

print(f"\n{'Model':<14} | {'AUC':>6} | {'Optimal Threshold (Youden J)':>28}")
print("-" * 55)

for m, color in zip(MODELS, MODEL_COLORS):
    k      = m["key"]
    binary = viz[k]["binary"]
    scores = viz[k]["scores"]

    # skip if all same label (ROC undefined)
    if len(set(binary)) < 2:
        print(f"{m['name']:<14} | {'N/A':>6} | {'N/A':>28}")
        continue

    fpr, tpr, thresholds = roc_curve(binary, scores)
    roc_auc  = auc(fpr, tpr)
    J        = tpr - fpr
    opt_idx  = np.argmax(J)
    opt_thr  = thresholds[opt_idx]

    plt.plot(fpr, tpr, color=color, lw=2, label=f"{m['name']} (AUC={roc_auc:.2f})")
    plt.scatter(fpr[opt_idx], tpr[opt_idx], marker='o', color=color, zorder=5, s=60)
    print(f"{m['name']:<14} | {roc_auc:>6.4f} | {opt_thr:>28.4f}")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(ROC_PATH)
plt.close()
print(f" -> Exported '{ROC_PATH}'")

print(f"\nEnd of Benchmark Suite. All results have been logged to '{LOG_PATH}'.")
