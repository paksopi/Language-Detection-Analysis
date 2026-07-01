import os
import sys
import tracemalloc
from lingua import Language, LanguageDetectorBuilder
import langdetect

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case_4.txt"

if not os.path.exists(TEST_CASE_FILE):
    print(f"Error: Could not find {TEST_CASE_FILE!r}")
    sys.exit(1)

test_cases = []
with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
            
        # Parse ground truth if it exists (e.g., "MY | Tumbuhan")
        if "|" in line:
            label, text = line.split("|", 1)
            test_cases.append({"expected": label.strip().upper(), "text": text.strip()})
        else:
            test_cases.append({"expected": "UNKNOWN", "text": line})

if not test_cases:
    print(f"Error: {TEST_CASE_FILE!r} had no usable lines.")
    sys.exit(1)

EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id", "ZH": "zh", "TA": "ta"}

print(f"Loaded {len(test_cases)} test cases from {TEST_CASE_FILE!r}\n")

# ==============================================================================
# 2. BENCHMARK 1: RAM / MEMORY PROFILING
# ==============================================================================
print("=" * 70)
print("BENCHMARK 1: RAM & MEMORY PROFILING")
print("=" * 70)
print("Starting tracemalloc... building Lingua models in memory.")

# Measure the memory spike required to load Lingua's models
tracemalloc.start()
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.MALAY, Language.INDONESIAN, Language.CHINESE, Language.TAMIL
).build()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

peak_mb = peak_mem / (1024 * 1024)
print(f"-> Peak RAM used to load Lingua models: {peak_mb:.2f} MB\n")

# ==============================================================================
# 3. BENCHMARK 2 & 3: LENGTH STRATIFICATION & CONFIDENCE THRESHOLDS
# ==============================================================================
print("=" * 70)
print("BENCHMARK 2 & 3: PROCESSING DATASET...")
print("=" * 70)

buckets = {
    "Short (1-4 words)":   {"lg_correct": 0, "ld_correct": 0, "total": 0},
    "Medium (5-15 words)": {"lg_correct": 0, "ld_correct": 0, "total": 0},
    "Long (16+ words)":    {"lg_correct": 0, "ld_correct": 0, "total": 0}
}

lingua_conf_correct = []
lingua_conf_incorrect = []

for case in test_cases:
    text = case["text"]
    expected_label = case["expected"]
    
    # Only score items that have a valid label
    if expected_label not in EXPECTED_TO_ISO:
        continue
        
    expected_iso = EXPECTED_TO_ISO[expected_label]
    word_count = len(text.split())
    
    # Determine length bucket
    if word_count <= 4:
        b_key = "Short (1-4 words)"
    elif word_count <= 15:
        b_key = "Medium (5-15 words)"
    else:
        b_key = "Long (16+ words)"
        
    buckets[b_key]["total"] += 1
    
    # Langdetect processing
    try:
        ld_iso = langdetect.detect(text)
        if ld_iso.startswith("zh"): 
            ld_iso = "zh"
    except:
        ld_iso = "unknown"
        
    # Lingua processing
    lg_lang = detector.detect_language_of(text)
    lg_iso = lg_lang.iso_code_639_1.name.lower() if lg_lang else "unknown"
    
    # Lingua Confidence tracking
    confidences = detector.compute_language_confidence_values(text)
    top_conf = confidences[0].value if confidences else 0.0
    
    # Scoring
    if ld_iso == expected_iso:
        buckets[b_key]["ld_correct"] += 1
        
    if lg_iso == expected_iso:
        buckets[b_key]["lg_correct"] += 1
        lingua_conf_correct.append(top_conf)
    else:
        lingua_conf_incorrect.append(top_conf)

print("Processing complete. Generating report...\n")

# ==============================================================================
# 4. FINAL ENGINEERING REPORT OUTPUT
# ==============================================================================
print("=" * 70)
print("LENGTH-STRATIFIED ACCURACY (How length affects AI)")
print("=" * 70)
for b_key, stats in buckets.items():
    total = stats["total"]
    if total == 0: 
        print(f"Bucket: {b_key:<20} | No test cases fit this length.")
        continue
    
    lg_pct = (stats["lg_correct"] / total) * 100
    ld_pct = (stats["ld_correct"] / total) * 100
    
    print(f"Bucket: {b_key:<20} | Total Tests: {total}")
    print(f"  -> Lingua Accuracy:     {lg_pct:.1f}% ({stats['lg_correct']}/{total})")
    print(f"  -> Langdetect Accuracy: {ld_pct:.1f}% ({stats['ld_correct']}/{total})\n")

print("=" * 70)
print("CONFIDENCE THRESHOLDING (Identifying False Positives)")
print("=" * 70)

avg_correct = sum(lingua_conf_correct) / len(lingua_conf_correct) if lingua_conf_correct else 0
avg_incorrect = sum(lingua_conf_incorrect) / len(lingua_conf_incorrect) if lingua_conf_incorrect else 0

print(f"Average Confidence when Lingua is RIGHT: {avg_correct:.4f}")
if lingua_conf_incorrect:
    print(f"Average Confidence when Lingua is WRONG: {avg_incorrect:.4f}")
    
    # Suggest a mathematical threshold (midpoint between right and wrong)
    suggested_threshold = avg_incorrect + ((avg_correct - avg_incorrect) / 2)
    print(f"\nRECOMMENDATION FOR SERVER ROUTING:")
    print(f"If Lingua's confidence is below {suggested_threshold:.4f}, flag the message")
    print("for manual review or prompt the student for more context.")
else:
    print("Lingua got 100% of these domain tests correct! No incorrect confidences to measure.")
print("=" * 70)