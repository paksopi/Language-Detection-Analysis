import os
import sys
import time
import tracemalloc
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

TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case_4.txt"

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
# 2. BENCHMARK 1: RAM & MEMORY PROFILING
# ==============================================================================
print("=" * 75)
print("BENCHMARK 1: RAM & MEMORY PROFILING")
print("=" * 75)
print("Starting tracemalloc... building Lingua models in memory.")

tracemalloc.start()
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.MALAY, Language.INDONESIAN, Language.CHINESE, Language.TAMIL
).build()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

peak_mb = peak_mem / (1024 * 1024)
print(f"-> Peak RAM used to load Lingua models: {peak_mb:.2f} MB\n")

# ==============================================================================
# 3. BENCHMARK 2: RAW PROCESSING SPEED
# ==============================================================================
print("=" * 75)
print("BENCHMARK 2: RAW PROCESSING SPEED")
print("=" * 75)

REPEATS = 500
total_calls = REPEATS * total_cases
print(f"Running {REPEATS} isolated passes over {total_cases} lines ({total_calls} total calls)...")

# --- Langdetect Speed Test ---
_start = time.perf_counter()
for _ in range(REPEATS):
    for case in test_cases:
        try:
            langdetect.detect(case["text"])
        except:
            pass
ld_time = time.perf_counter() - _start
ld_avg_ms = (ld_time / total_calls) * 1000

# --- Lingua Speed Test ---
_start = time.perf_counter()
for _ in range(REPEATS):
    for case in test_cases:
        detector.detect_language_of(case["text"])
lingua_time = time.perf_counter() - _start
lingua_avg_ms = (lingua_time / total_calls) * 1000

print(f"  -> Lingua:     {lingua_avg_ms:.4f} ms per call ({lingua_time:.2f}s total)")
print(f"  -> Langdetect: {ld_avg_ms:.4f} ms per call ({ld_time:.2f}s total)\n")

# ==============================================================================
# 4. BENCHMARK 3 & 4: ACCURACY & CONFIDENCE THRESHOLDING
# ==============================================================================
print("=" * 75)
print("BENCHMARK 3: LENGTH-STRATIFIED ACCURACY")
print("=" * 75)

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
    
    if expected_label not in EXPECTED_TO_ISO:
        continue
        
    expected_iso = EXPECTED_TO_ISO[expected_label]
    word_count = len(text.split())
    
    # Determine bucket
    if word_count <= 4:
        b_key = "Short (1-4 words)"
    elif word_count <= 15:
        b_key = "Medium (5-15 words)"
    else:
        b_key = "Long (16+ words)"
        
    buckets[b_key]["total"] += 1
    
    # Process Langdetect
    try:
        ld_iso = langdetect.detect(text)
        if ld_iso.startswith("zh"): ld_iso = "zh"
    except:
        ld_iso = "unknown"
        
    # Process Lingua
    lg_lang = detector.detect_language_of(text)
    lg_iso = lg_lang.iso_code_639_1.name.lower() if lg_lang else "unknown"
    
    # Track Lingua Confidence
    confidences = detector.compute_language_confidence_values(text)
    top_conf = confidences[0].value if confidences else 0.0
    
    # Score
    if ld_iso == expected_iso:
        buckets[b_key]["ld_correct"] += 1
        
    if lg_iso == expected_iso:
        buckets[b_key]["lg_correct"] += 1
        lingua_conf_correct.append(top_conf)
    else:
        lingua_conf_incorrect.append(top_conf)

# Print Accuracy Report
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

print("=" * 75)
print("BENCHMARK 4: CONFIDENCE THRESHOLDING")
print("=" * 75)

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
# 5. BENCHMARK 5: SESSION STATE MANAGEMENT (SLIDING WINDOW)
# ==============================================================================
print("=" * 75)
print("BENCHMARK 5: SESSION STATE MANAGEMENT (Sliding Window Integration)")
print("=" * 75)
print("Simulating a conversation to test how the assistant handles code-switching turns...\n")

class ChatSession:
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
            print(f"  ACTION: ⚓ Anchor Set! Assistant will now speak [{self.active_language}]")
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
                    print(f"  ACTION: 🔄 SWITCHING OVERRIDE! Assistant changes from [{self.active_language}] to [{detected_iso}]")
                    self.active_language = detected_iso
                    self.consecutive_turns = 0  # Reset
            else:
                print(f"  ACTION: Ignored. Confidence ({confidence_score*100:.1f}%) is too low. Staying in [{self.active_language}]")
                self.consecutive_turns = 0 # Reset because it was just noise
                
        return self.active_language


# Simulate a 7-turn conversation
session = ChatSession()

mock_conversation = [
    "Cikgu, macam mana nak kira fraction ni?",        # Turn 1: Sets anchor to Malay
    "Faham cikgu, senang je rupanya.",                # Turn 2: Stays Malay
    "ok",                                             # Turn 3: Lingua guesses EN, but low confidence (NOISE)
    "Thanks!",                                        # Turn 4: Lingua guesses EN, high confidence (Turn 1)
    "Baik, saya cuba buat soalan seterusnya.",        # Turn 5: Back to Malay! (Resets the English counter)
    "Can we switch to my English essay now?",         # Turn 6: English high confidence (Turn 1)
    "The essay is about the effects of global warming." # Turn 7: English high confidence (Turn 2 - TRIGGERS SWITCH)
]

for i, message in enumerate(mock_conversation, 1):
    active_lang = session.process_user_message(message, i)
    print(f"  -> FINAL RESULT: VoxCPM2 & LLM will use [{active_lang}]")

print("-" * 75)
print("End of Benchmark Suite. All results have been logged to log.txt.")