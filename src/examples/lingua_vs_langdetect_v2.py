import os
import sys
import time
from collections import Counter

from lingua import Language, LanguageDetectorBuilder
import langdetect

# ==============================================================================
# 1. SETUP LOGGING TO FILE & CONSOLE
# ==============================================================================
class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout so all print() statements go to both console and log.txt
sys.stdout = Logger("log.txt")

# ==============================================================================
# 2. MAPPING & CONFIGURATION
# ==============================================================================
# Map our custom dataset labels to library-specific outputs for accurate scoring.
LABEL_MAP = {
    "EN": {"lingua": "ENGLISH", "langdetect": ["en"]},
    "MY": {"lingua": "MALAY", "langdetect": ["ms"]},
    "ID": {"lingua": "INDONESIAN", "langdetect": ["id"]},
    "ZH": {"lingua": "CHINESE", "langdetect": ["zh-cn", "zh-tw"]},
    "TA": {"lingua": "TAMIL", "langdetect": ["ta"]},
}

print("Building Lingua detector with ALL supported languages for a fair comparison...")
_build_start = time.perf_counter()
# Prioritize not abandoning other languages: we now load all ~75 languages.
detector = LanguageDetectorBuilder.from_all_languages().build()
_build_seconds = time.perf_counter() - _build_start
print(f"Lingua built in {_build_seconds:.4f}s\n")

# ==============================================================================
# 3. LOAD & PARSE TEST CASES
# ==============================================================================
TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case.txt"

if not os.path.exists(TEST_CASE_FILE):
    print(f"Could not find {TEST_CASE_FILE!r}.")
    sys.exit(1)

test_cases = []
with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Expecting format: "LABEL | text data"
        if "|" in line:
            label, text = line.split("|", 1)
            test_cases.append({"label": label.strip().upper(), "text": text.strip()})
        else:
            # Fallback for unlabelled lines
            test_cases.append({"label": "UNKNOWN", "text": line})

if not test_cases:
    print(f"{TEST_CASE_FILE!r} had no usable lines.")
    sys.exit(1)

print(f"Loaded {len(test_cases)} test case(s) from {TEST_CASE_FILE!r}\n")

# ==============================================================================
# 4. MAIN DETECTION & ACCURACY SCORING
# ==============================================================================
print("=" * 110)
print(f"{'EXPECTED':<10} | {'INPUT':<40} | {'langdetect':<12} | {'lingua (top)':<14} | lingua conf")
print("=" * 110)

lingua_correct = 0
langdetect_correct = 0
scorable_items = 0

lingua_predictions = []
langdetect_predictions = []

for case in test_cases:
    text = case["text"]
    expected_label = case["label"]
    
    # --- langdetect ---
    try:
        ld_res = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        ld_res = "FAILED"
    
    langdetect_predictions.append(ld_res)

    # --- lingua ---
    lingua_lang = detector.detect_language_of(text)
    lingua_res = lingua_lang.name if lingua_lang else "None"
    lingua_predictions.append(lingua_res)

    confidences = detector.compute_language_confidence_values(text)
    top_conf = confidences[0] if confidences else None
    conf_str = f"{top_conf.language.name}={top_conf.value:.2f}" if top_conf else "n/a"

    # --- Accuracy Check ---
    is_ld_correct = False
    is_lg_correct = False

    if expected_label in LABEL_MAP:
        scorable_items += 1
        if ld_res in LABEL_MAP[expected_label]["langdetect"]:
            is_ld_correct = True
            langdetect_correct += 1
        if lingua_res == LABEL_MAP[expected_label]["lingua"]:
            is_lg_correct = True
            lingua_correct += 1
            
    # Truncate text for display
    display_text = text if len(text) <= 38 else text[:35] + "..."
    
    # Mark incorrect predictions with an 'X' in the console output
    ld_display = f"{ld_res} {' ' if is_ld_correct or expected_label == 'UNKNOWN' else '(X)'}"
    lg_display = f"{lingua_res} {' ' if is_lg_correct or expected_label == 'UNKNOWN' else '(X)'}"

    print(f"{expected_label:<10} | {display_text:<40} | {ld_display:<12} | {lg_display:<14} | {conf_str}")

# ==============================================================================
# 5. ACCURACY & DISTRIBUTION REPORT
# ==============================================================================
print("\n" + "=" * 110)
print("PERFORMANCE METRICS & LANGUAGE DISTRIBUTION")
print("=" * 110)

if scorable_items > 0:
    ld_acc = (langdetect_correct / scorable_items) * 100
    lg_acc = (lingua_correct / scorable_items) * 100
    print(f"Accuracy (on {scorable_items} labelled items):")
    print(f"  langdetect: {ld_acc:.1f}%")
    print(f"  lingua:     {lg_acc:.1f}%\n")
else:
    print("No labelled items found. Add labels like 'MY | text' to test_cases.txt to see accuracy.\n")

# Calculate Distribution
ld_counts = Counter(langdetect_predictions)
lg_counts = Counter(lingua_predictions)

total_cases = len(test_cases)

print(f"{'Detected Language':<20} | {'lingua %':<15} | {'langdetect %'}")
print("-" * 60)

# Combine keys from both to show a unified table
all_detected = set(lg_counts.keys()).union(set(ld_counts.keys()))
for lang in sorted(all_detected):
    lg_pct = (lg_counts.get(lang, 0) / total_cases) * 100
    ld_pct = (ld_counts.get(lang, 0) / total_cases) * 100
    print(f"{lang:<20} | {lg_pct:>5.1f}%          | {ld_pct:>5.1f}%")

print("\nMAIN LANGUAGE IDENTIFIED:")
main_lg = lg_counts.most_common(1)[0]
main_ld = ld_counts.most_common(1)[0]
print(f"  According to lingua:     {main_lg[0]} ({main_lg[1]}/{total_cases} texts)")
print(f"  According to langdetect: {main_ld[0]} ({main_ld[1]}/{total_cases} texts)")

# ==============================================================================
# 6. MY vs ID CONFIDENCE GAP & MIXED LANGUAGE (Lingua only)
# ==============================================================================
print("\n" + "=" * 110)
print("MALAY vs INDONESIAN CONFIDENCE GAP (lingua)")
print("=" * 110)

for case in test_cases:
    confidences = detector.compute_language_confidence_values(case["text"])
    if not confidences:
        continue
    top = confidences[0]
    if top.language not in (Language.MALAY, Language.INDONESIAN):
        continue
    runner_up = next((c for c in confidences[1:] if c.language in (Language.MALAY, Language.INDONESIAN)), None)
    gap_str = f"gap={top.value - runner_up.value:.2f} vs {runner_up.language.name}" if runner_up else "no MY/ID runner-up"
    display_text = case["text"] if len(case["text"]) <= 48 else case["text"][:45] + "..."
    print(f"{display_text:<50} -> {top.language.name:<11} ({top.value:.2f}) | {gap_str}")

print("\n" + "=" * 110)
print("MIXED-LANGUAGE SEGMENT DETECTION (lingua only)")
print("=" * 110)

found_any_mixed = False
for case in test_cases:
    segments = detector.detect_multiple_languages_of(case["text"])
    distinct_langs = {seg.language for seg in segments}
    if len(distinct_langs) > 1:
        found_any_mixed = True
        print(f"Input: {case['text'][:70]}...")
        for seg in segments:
            snippet = case["text"][seg.start_index:seg.end_index]
            # Replace newlines for clean printing
            snippet = snippet.replace('\n', ' ')
            print(f"  {seg.language.name:<10} -> {snippet}")
        print()

# ==============================================================================
# 7. SPEED BENCHMARK
# ==============================================================================
print("\n" + "=" * 110)
print("SPEED BENCHMARK")
print("=" * 110)

REPEATS = 50
print(f"Running {REPEATS} pass(es) over {total_cases} line(s)...\n")

_start = time.perf_counter()
for _ in range(REPEATS):
    for case in test_cases:
        try:
            langdetect.detect(case["text"])
        except:
            pass
langdetect_total = time.perf_counter() - _start
langdetect_calls = REPEATS * total_cases
langdetect_avg_ms = (langdetect_total / langdetect_calls) * 1000

_start = time.perf_counter()
for _ in range(REPEATS):
    for case in test_cases:
        detector.detect_language_of(case["text"])
lingua_total = time.perf_counter() - _start
lingua_calls = REPEATS * total_cases
lingua_avg_ms = (lingua_total / lingua_calls) * 1000

print(f"{'Library':<14} | {'Total time':<12} | {'Avg per call':<14} | Calls")
print("-" * 60)
print(f"{'langdetect':<14} | {langdetect_total:>9.4f}s  | {langdetect_avg_ms:>10.4f}ms  | {langdetect_calls}")
print(f"{'lingua':<14} | {lingua_total:>9.4f}s  | {lingua_avg_ms:>10.4f}ms  | {lingua_calls}")