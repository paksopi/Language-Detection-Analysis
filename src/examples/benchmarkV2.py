import os
import sys
import time
import tracemalloc
from collections import defaultdict
from lingua import Language, LanguageDetectorBuilder
import langdetect

# ==============================================================================
# 1. SETUP & DATA LOADING
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

# Redirect stdout so everything prints to terminal and saves to log.txt
sys.stdout = Logger("log.txt")

TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case_5.txt"

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
# WORD-COUNT BUCKETING (CJK-aware)
# ==============================================================================
# Python's text.split() counts whitespace-separated tokens, which works for
# EN/MY/ID/TA (all use spaces) but breaks for ZH (Chinese has no spaces, so a
# whole sentence counts as "1 word"). For languages without spaces, this uses
# character count instead, with thresholds calibrated against the actual
# character lengths of this project's own 1w/2w/3-7w/8-16w/17-50w test data
# (see the comments next to each threshold below).
#
# This only affects which bucket a line is grouped into for reporting -- it
# does NOT change what text is sent to the detectors.

CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (covers Chinese)
    (0x3040, 0x30FF),   # Hiragana / Katakana (Japanese, not used here but safe to keep)
    (0xAC00, 0xD7A3),   # Hangul (Korean, not used here but safe to keep)
]


def is_cjk_text(text):
    """True if the text is dominated by CJK characters (no spaces to split on)."""
    cjk_count = sum(
        1 for ch in text
        if any(start <= ord(ch) <= end for start, end in CJK_RANGES)
    )
    letters = sum(1 for ch in text if ch.isalpha() or cjk_count)
    return cjk_count > 0 and cjk_count >= len(text.replace(" ", "")) * 0.3


def bucket_for(text):
    """Returns one of the 5 bucket labels, using char-count for CJK text."""
    if is_cjk_text(text):
        n = len(text.replace(" ", "").replace(",", "").replace("，", "")
                 .replace(".", "").replace("。", "").replace("?", "").replace("？", ""))
        # Calibrated against this project's actual ZH samples:
        # 1w ~1-4 chars, 2w ~3-5 chars, 3-7w ~10-12, 8-16w ~21-45, 17-50w ~53-63
        if n <= 2:
            return "1 word"
        elif n <= 6:
            return "2 words"
        elif n <= 15:
            return "3-7 words"
        elif n <= 48:
            return "8-16 words"
        else:
            return "17-50 words"
    else:
        n = len(text.split())
        if n <= 1:
            return "1 word"
        elif n <= 2:
            return "2 words"
        elif n <= 7:
            return "3-7 words"
        elif n <= 16:
            return "8-16 words"
        else:
            return "17-50 words"


BUCKET_ORDER = ["1 word", "2 words", "3-7 words", "8-16 words", "17-50 words"]
LANGUAGE_ORDER = ["EN", "MY", "ID", "ZH", "TA"]


# ==============================================================================
# 2. BENCHMARK 1: RAM & MEMORY PROFILING (langdetect vs lingua-low vs lingua-high)
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

# langdetect has no comparable build-time memory cost to profile the same way
# (it loads its profile data lazily per-call), so it's noted but not measured
# here on the same axis -- listed for completeness, not a missing number.
print(f"  -> Lingua (HIGH accuracy mode): {peak_high / (1024 * 1024):.2f} MB peak RAM to build")
print(f"  -> Lingua (LOW accuracy mode):  {peak_low / (1024 * 1024):.2f} MB peak RAM to build")
print(f"  -> langdetect: not directly comparable (loads profiles lazily per call, "
      f"no equivalent one-time build step)\n")


# ==============================================================================
# 3. BENCHMARK 2: RAW PROCESSING SPEED -- bucketed, 3-way
# ==============================================================================
print("=" * 90)
print("BENCHMARK 2: RAW PROCESSING SPEED (bucketed by word count, 3-way comparison)")
print("=" * 90)

REPEATS = 100  # per bucket, since each bucket now has its own timing loop
print(f"Running {REPEATS} passes per bucket. This is a relative comparison on YOUR\n"
      f"machine and data -- not directly comparable to lingua's own published\n"
      f"benchmark numbers (different test set, different machine).\n")

texts_by_bucket = defaultdict(list)
for case in test_cases:
    texts_by_bucket[bucket_for(case["text"])].append(case["text"])

print(f"{'Bucket':<14} | {'n':>4} | {'langdetect':>12} | {'lingua-low':>12} | {'lingua-high':>12} | fastest")
print("-" * 90)

speed_results = {}  # bucket -> {"langdetect": ms, "lingua_low": ms, "lingua_high": ms}

for bucket in BUCKET_ORDER:
    texts = texts_by_bucket.get(bucket, [])
    if not texts:
        print(f"{bucket:<14} | (no test cases in this bucket)")
        continue

    total_calls = REPEATS * len(texts)

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts:
            try:
                langdetect.detect(text)
            except Exception:
                pass
    ld_ms = (time.perf_counter() - _start) / total_calls * 1000

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts:
            detector_low.detect_language_of(text)
    low_ms = (time.perf_counter() - _start) / total_calls * 1000

    _start = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts:
            detector_high.detect_language_of(text)
    high_ms = (time.perf_counter() - _start) / total_calls * 1000

    speed_results[bucket] = {"langdetect": ld_ms, "lingua_low": low_ms, "lingua_high": high_ms}
    fastest = min([("langdetect", ld_ms), ("lingua-low", low_ms), ("lingua-high", high_ms)],
                  key=lambda x: x[1])[0]

    print(f"{bucket:<14} | {len(texts):>4} | {ld_ms:>10.4f}ms | {low_ms:>10.4f}ms | {high_ms:>10.4f}ms | {fastest}")

print(
    "\nNote: watch whether 'fastest' stays the same across buckets or changes --\n"
    "that tradeoff (short vs long text speed) matters more than a single overall\n"
    "average, which would hide exactly this kind of crossover.\n"
)


# ==============================================================================
# 4. BENCHMARK 3: ACCURACY -- by bucket AND by language, 3-way
# ==============================================================================
print("=" * 90)
print("BENCHMARK 3: ACCURACY BY BUCKET (3-way comparison)")
print("=" * 90)

# bucket -> lang -> {"total", "ld_correct", "low_correct", "high_correct"}
bucket_lang_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "ld": 0, "low": 0, "high": 0}))
# lang -> {"total", "ld_correct", "low_correct", "high_correct"}  (overall per-language, all buckets combined)
lang_stats = defaultdict(lambda: {"total": 0, "ld": 0, "low": 0, "high": 0})

lingua_conf_correct = []   # high-accuracy detector confidence when correct
lingua_conf_incorrect = []  # high-accuracy detector confidence when wrong

for case in test_cases:
    text = case["text"]
    expected_label = case["expected"]

    if expected_label not in EXPECTED_TO_ISO:
        continue

    expected_iso = EXPECTED_TO_ISO[expected_label]
    bucket = bucket_for(text)

    # --- langdetect ---
    try:
        ld_iso = langdetect.detect(text)
        if ld_iso.startswith("zh"):
            ld_iso = "zh"
    except Exception:
        ld_iso = "unknown"

    # --- lingua low accuracy mode ---
    low_lang = detector_low.detect_language_of(text)
    low_iso = low_lang.iso_code_639_1.name.lower() if low_lang else "unknown"

    # --- lingua high accuracy mode ---
    high_lang = detector_high.detect_language_of(text)
    high_iso = high_lang.iso_code_639_1.name.lower() if high_lang else "unknown"

    # Track high-accuracy confidence for the confidence-thresholding section below
    confidences = detector_high.compute_language_confidence_values(text)
    top_conf = confidences[0].value if confidences else 0.0

    ld_ok = ld_iso == expected_iso
    low_ok = low_iso == expected_iso
    high_ok = high_iso == expected_iso

    if high_ok:
        lingua_conf_correct.append(top_conf)
    else:
        lingua_conf_incorrect.append(top_conf)

    for stats in (bucket_lang_stats[bucket][expected_label], lang_stats[expected_label]):
        stats["total"] += 1
        stats["ld"] += int(ld_ok)
        stats["low"] += int(low_ok)
        stats["high"] += int(high_ok)

# --- print accuracy by bucket, broken down by language within each bucket ---
for bucket in BUCKET_ORDER:
    if bucket not in bucket_lang_stats:
        continue
    print(f"\nBucket: {bucket}")
    print(f"  {'LANG':<5} | {'n':>3} | {'langdetect':>10} | {'lingua-low':>10} | {'lingua-high':>11}")
    for lang in LANGUAGE_ORDER:
        if lang not in bucket_lang_stats[bucket]:
            continue
        s = bucket_lang_stats[bucket][lang]
        n = s["total"]
        print(f"  {lang:<5} | {n:>3} | {s['ld']/n*100:>9.1f}% | {s['low']/n*100:>9.1f}% | {s['high']/n*100:>10.1f}%")

print("\n" + "=" * 90)
print("BENCHMARK 3b: ACCURACY BY LANGUAGE (all buckets combined)")
print("=" * 90)
print(f"{'LANG':<5} | {'n':>4} | {'langdetect':>10} | {'lingua-low':>10} | {'lingua-high':>11}")
print("-" * 60)
for lang in LANGUAGE_ORDER:
    if lang not in lang_stats:
        continue
    s = lang_stats[lang]
    n = s["total"]
    print(f"{lang:<5} | {n:>4} | {s['ld']/n*100:>9.1f}% | {s['low']/n*100:>9.1f}% | {s['high']/n*100:>10.1f}%")

overall_total = sum(s["total"] for s in lang_stats.values())
overall_ld = sum(s["ld"] for s in lang_stats.values())
overall_low = sum(s["low"] for s in lang_stats.values())
overall_high = sum(s["high"] for s in lang_stats.values())
print("-" * 60)
print(f"{'ALL':<5} | {overall_total:>4} | {overall_ld/overall_total*100:>9.1f}% | "
      f"{overall_low/overall_total*100:>9.1f}% | {overall_high/overall_total*100:>10.1f}%")


# ==============================================================================
# 5. BENCHMARK 4: CONFIDENCE THRESHOLDING (high-accuracy mode)
# ==============================================================================
print("\n" + "=" * 90)
print("BENCHMARK 4: CONFIDENCE THRESHOLDING (lingua high-accuracy mode)")
print("=" * 90)

avg_correct = sum(lingua_conf_correct) / len(lingua_conf_correct) if lingua_conf_correct else 0
avg_incorrect = sum(lingua_conf_incorrect) / len(lingua_conf_incorrect) if lingua_conf_incorrect else 0

print(f"Average Confidence when Lingua is RIGHT: {avg_correct:.4f}")
if lingua_conf_incorrect:
    print(f"Average Confidence when Lingua is WRONG: {avg_incorrect:.4f}")

    suggested_threshold = avg_incorrect + ((avg_correct - avg_incorrect) / 2)
    print(f"\nRECOMMENDATION FOR SERVER ROUTING:")
    print(f"If Lingua's confidence is below {suggested_threshold:.4f}, flag the message")
    print("for manual review or prompt the student for more context.")
else:
    print("Lingua got 100% of these domain tests correct! No incorrect confidences to measure.")
print("\n")


# ==============================================================================
# 6. BENCHMARK 5: SESSION STATE MANAGEMENT (SLIDING WINDOW)
# ==============================================================================
print("=" * 90)
print("BENCHMARK 5: SESSION STATE MANAGEMENT (Sliding Window Integration)")
print("=" * 90)
print("Simulating a conversation to test how the project handles code-switching turns...\n")
print("(Uses the lingua HIGH-accuracy detector, since this models a real the project turn.)\n")

detector = detector_high  # alias for the session simulation below


class projectSession:
    def __init__(self):
        self.active_language = None
        self.foreign_language_candidate = None
        self.consecutive_turns = 0

        # The "Percentage" thresholds
        self.HIGH_CONFIDENCE = 0.85  # 85% confidence required to even consider a switch
        self.REQUIRED_TURNS = 2      # Must happen 2 times in a row

    def process_user_message(self, text, turn_number):
        print("-" * 75)
        print(f"TURN {turn_number}: User says -> '{text}'")

        # Get Lingua's prediction and confidence percentage
        confidences = detector.compute_language_confidence_values(text)
        if not confidences:
            return self.active_language

        top_match = confidences[0]
        detected_iso = top_match.language.iso_code_639_1.name.lower()
        confidence_score = top_match.value

        print(f"  Lingua Detects: [{detected_iso}] with {confidence_score*100:.1f}% confidence")

        # SCENARIO A: First message of the conversation (The Anchor)
        if self.active_language is None:
            self.active_language = detected_iso
            print(f"  ACTION: Anchor Set! the project will now speak [{self.active_language}]")
            return self.active_language

        # SCENARIO B: User is speaking the expected active language
        if detected_iso == self.active_language:
            self.consecutive_turns = 0  # Reset the counter
            self.foreign_language_candidate = None
            print(f"  ACTION: Language matches active session. Staying in [{self.active_language}]")
            return self.active_language

        # SCENARIO C: User speaks a DIFFERENT language
        if detected_iso != self.active_language:
            # Did it hit the 85% confidence percentage?
            if confidence_score >= self.HIGH_CONFIDENCE:
                # Is it the same foreign language as the previous turn?
                if detected_iso == self.foreign_language_candidate:
                    self.consecutive_turns += 1
                else:
                    self.foreign_language_candidate = detected_iso
                    self.consecutive_turns = 1

                print(f"  WARNING: High confidence foreign language detected ({self.consecutive_turns}/{self.REQUIRED_TURNS} turns)")

                # Check if we hit the Sliding Window limit!
                if self.consecutive_turns >= self.REQUIRED_TURNS:
                    print(f"  ACTION: SWITCHING OVERRIDE! the project changes from [{self.active_language}] to [{detected_iso}]")
                    self.active_language = detected_iso
                    self.consecutive_turns = 0  # Reset
            else:
                print(f"  ACTION: Ignored. Confidence ({confidence_score*100:.1f}%) is too low. Staying in [{self.active_language}]")
                self.consecutive_turns = 0  # Reset because it was just noise

        return self.active_language


# Simulate a 7-turn conversation
session = projectSession()

mock_conversation = [
    "Cikgu, macam mana nak kira fraction ni?",        # Turn 1: Sets anchor to Malay
    "Faham cikgu, senang je rupanya.",                # Turn 2: Stays Malay
    "ok",                                             # Turn 3: Lingua guesses EN, but low confidence (NOISE)
    "Thanks!",                                        # Turn 4: Lingua guesses EN, high confidence (Turn 1)
    "Baik, saya cuba buat soalan seterusnya.",        # Turn 5: Back to Malay! (Resets the English counter)
    "Can we switch to my English essay now?",         # Turn 6: English high confidence (Turn 1)
    "The essay is about the effects of global warming."  # Turn 7: English high confidence (Turn 2 - TRIGGERS SWITCH)
]

for i, message in enumerate(mock_conversation, 1):
    active_lang = session.process_user_message(message, i)
    print(f"  -> FINAL RESULT: VoxCPM2 & LLM will use [{active_lang}]")

print("-" * 75)
print("End of Benchmark Suite. All results have been logged to log.txt.")
