import os
import sys
import time
import re
import tracemalloc
from collections import defaultdict
from lingua import Language, LanguageDetectorBuilder
import langdetect

# --- New Data Science Imports ---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
class Logger:
    def __init__(self, filename="log2.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout so everything prints to terminal and saves to log2.txt
sys.stdout = Logger("log2.txt")

TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case_6.txt"

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

EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id", "ZH": "zh", "TA": "ta"}
total_cases = len(test_cases)

print(f"Loaded {total_cases} test cases from {TEST_CASE_FILE!r}\n")


# ==============================================================================
# WORD-COUNT BUCKETING (CJK & Punctuation Aware)
# ==============================================================================
CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x30FF),   # Hiragana / Katakana
    (0xAC00, 0xD7A3),   # Hangul
]

def is_cjk_text(text):
    """True if the text is dominated by CJK characters."""
    cjk_count = sum(
        1 for ch in text
        if any(start <= ord(ch) <= end for start, end in CJK_RANGES)
    )
    return cjk_count > 0 and cjk_count >= len(text.replace(" ", "")) * 0.3

def bucket_for(text):
    """Returns one of the 5 bucket labels using robust regex tokenization."""
    if is_cjk_text(text):
        # Strip all non-word characters for pure CJK length counting
        n = len(re.sub(r'[^\w]', '', text))
        
        if n <= 2: return "1 word"
        elif n <= 6: return "2 words"
        elif n <= 15: return "3-7 words"
        elif n <= 48: return "8-16 words"
        else: return "17-50 words"
    else:
        # regex cleanup: Keeps alphanumeric + Tamil Unicode block, drops punctuation and math ($)
        clean_text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
        n = len(clean_text.split())
        
        if n <= 1: return "1 word"
        elif n <= 2: return "2 words"
        elif n <= 7: return "3-7 words"
        elif n <= 16: return "8-16 words"
        else: return "17-50 words"

BUCKET_ORDER = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]
LANGUAGE_ORDER = ["EN", "MY", "ID", "ZH", "TA"]


# ==============================================================================
# 2. BENCHMARK 1: RAM & MEMORY PROFILING
# ==============================================================================
print("=" * 90)
print("BENCHMARK 1: RAM & MEMORY PROFILING")
print("=" * 90)
print("Starting tracemalloc... building detector models in memory.\n")

LANGS = (Language.ENGLISH, Language.MALAY, Language.INDONESIAN, Language.CHINESE, Language.TAMIL)

tracemalloc.start()
detector_high = LanguageDetectorBuilder.from_languages(*LANGS).build()
_, peak_high = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
detector_low = LanguageDetectorBuilder.from_languages(*LANGS).with_low_accuracy_mode().build()
_, peak_low = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  -> Lingua (HIGH accuracy mode): {peak_high / (1024 * 1024):.2f} MB peak RAM to build")
print(f"  -> Lingua (LOW accuracy mode):  {peak_low / (1024 * 1024):.2f} MB peak RAM to build")
print(f"  -> langdetect: not directly comparable (loads profiles lazily per call)\n")


# ==============================================================================
# 3. BENCHMARK 2: RAW PROCESSING SPEED
# ==============================================================================
print("=" * 90)
print("BENCHMARK 2: RAW PROCESSING SPEED (bucketed by word count)")
print("=" * 90)

REPEATS = 100 
texts_by_bucket = defaultdict(list)
for case in test_cases:
    texts_by_bucket[bucket_for(case["text"])].append(case["text"])

print(f"{'Bucket':<14} | {'n':>4} | {'langdetect':>12} | {'lingua-low':>12} | {'lingua-high':>12} | fastest")
print("-" * 90)

for bucket in BUCKET_ORDER:
    texts = texts_by_bucket.get(bucket, [])
    if not texts:
        print(f"{bucket:<14} | (no test cases in this bucket)")
        continue

    total_calls = REPEATS * len(texts)

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts:
            try: langdetect.detect(text)
            except Exception: pass
    ld_ms = (time.perf_counter() - _start) / total_calls * 1000

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts: detector_low.detect_language_of(text)
    low_ms = (time.perf_counter() - _start) / total_calls * 1000

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts: detector_high.detect_language_of(text)
    high_ms = (time.perf_counter() - _start) / total_calls * 1000

    fastest = min([("langdetect", ld_ms), ("lingua-low", low_ms), ("lingua-high", high_ms)], key=lambda x: x[1])[0]
    print(f"{bucket:<14} | {len(texts):>4} | {ld_ms:>10.4f}ms | {low_ms:>10.4f}ms | {high_ms:>10.4f}ms | {fastest}")

print("\nNote: Watch for 'lingua-high' overtaking 'lingua-low' in the longer buckets (early-exit optimization).\n")


# ==============================================================================
# 4. BENCHMARK 3: ACCURACY
# ==============================================================================
print("=" * 90)
print("BENCHMARK 3: ACCURACY BY BUCKET (With Macro-Language Fallbacks)")
print("=" * 90)

bucket_lang_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "ld": 0, "low": 0, "high": 0}))
lang_stats = defaultdict(lambda: {"total": 0, "ld": 0, "low": 0, "high": 0})

# Arrays for Data Science visualizations
y_true = []
y_pred = []
y_true_binary = []
y_scores = []

for case in test_cases:
    text = case["text"]
    expected_label = case["expected"]
    if expected_label not in EXPECTED_TO_ISO: continue

    expected_iso = EXPECTED_TO_ISO[expected_label]
    bucket = bucket_for(text)

    # --- langdetect ---
    try:
        ld_iso = langdetect.detect(text)
        if ld_iso.startswith("zh"): ld_iso = "zh"
    except Exception:
        ld_iso = "unknown"

    # --- lingua ---
    low_lang = detector_low.detect_language_of(text)
    low_iso = low_lang.iso_code_639_1.name.lower() if low_lang else "unknown"

    high_lang = detector_high.detect_language_of(text)
    high_iso = high_lang.iso_code_639_1.name.lower() if high_lang else "unknown"

    confidences = detector_high.compute_language_confidence_values(text)
    top_conf = confidences[0].value if confidences else 0.0

    # Fallback allowance for Langdetect
    ld_ok = (ld_iso == expected_iso) or (expected_iso == 'ms' and ld_iso == 'id')
    low_ok = (low_iso == expected_iso)
    high_ok = (high_iso == expected_iso)

    # Log data for visualizations
    y_true.append(expected_iso)
    y_pred.append(high_iso)
    y_true_binary.append(1 if high_ok else 0)
    y_scores.append(top_conf)

    for stats in (bucket_lang_stats[bucket][expected_label], lang_stats[expected_label]):
        stats["total"] += 1
        stats["ld"] += int(ld_ok)
        stats["low"] += int(low_ok)
        stats["high"] += int(high_ok)

for bucket in BUCKET_ORDER:
    if bucket not in bucket_lang_stats: continue
    print(f"\nBucket: {bucket}")
    print(f"  {'LANG':<5} | {'n':>3} | {'langdetect':>10} | {'lingua-low':>10} | {'lingua-high':>11}")
    for lang in LANGUAGE_ORDER:
        if lang not in bucket_lang_stats[bucket]: continue
        s = bucket_lang_stats[bucket][lang]
        n = s["total"]
        print(f"  {lang:<5} | {n:>3} | {s['ld']/n*100:>9.1f}% | {s['low']/n*100:>9.1f}% | {s['high']/n*100:>10.1f}%")

print("\n" + "=" * 90)
print("BENCHMARK 3b: ACCURACY BY LANGUAGE (all buckets combined)")
print("=" * 90)
print(f"{'LANG':<5} | {'n':>4} | {'langdetect':>10} | {'lingua-low':>10} | {'lingua-high':>11}")
print("-" * 60)
for lang in LANGUAGE_ORDER:
    if lang not in lang_stats: continue
    s = lang_stats[lang]
    n = s["total"]
    print(f"{lang:<5} | {n:>4} | {s['ld']/n*100:>9.1f}% | {s['low']/n*100:>9.1f}% | {s['high']/n*100:>10.1f}%")


# ==============================================================================
# 5. BENCHMARK 4: DATA SCIENCE VISUALIZATIONS & EXACT THRESHOLDING
# ==============================================================================
print("\n" + "=" * 90)
print("BENCHMARK 4: ROC CURVE & OPTIMAL THRESHOLDING")
print("=" * 90)

# 1. Confusion Matrix
labels = ['en', 'ms', 'id', 'zh', 'ta']
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Lingua-High Confusion Matrix (MY vs ID Collision)")
plt.savefig("confusion_matrix.png")
plt.close()
print(" -> Exported 'confusion_matrix.png' successfully.")

# 2. ROC Curve and Youden's J Statistic
fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
roc_auc = auc(fpr, tpr)

# Youden's J statistic = TPR - FPR. The max value represents the optimal threshold.
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Best Threshold: {optimal_threshold:.4f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Incorrect Routings)')
plt.ylabel('True Positive Rate (Correct Routings)')
plt.title('ROC Curve for Lingua Confidence Scores')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()
print(" -> Exported 'roc_curve.png' successfully.")

print(f"\nMATHEMATICAL RECOMMENDATION FOR SERVER ROUTING (Youden's J):")
print(f"Set the the project manual-review threshold to exactly: {optimal_threshold:.4f}")
print("If Lingua's confidence falls below this number, trigger the fallback mechanism.")
print("\nEnd of Benchmark Suite. All results have been logged to log.txt.")