"""
lingua-py vs langdetect — hands-on comparison
Run this locally to see real, structured output for short and code-switched text.

SETUP (run these first):
    pip install lingua-language-detector
    pip install langdetect

PyPI package name note: the import is `lingua`, but the installable package is
`lingua-language-detector` (NOT `lingua` or `lingua-py` — those names are taken
by unrelated packages on PyPI, which is a common first-install mistake).

USAGE:
    Put your test messages in test_case.txt (one message per line, in the same
    folder as this script), then run:
        python lingua_vs_langdetect_sample.py
    Or point it at any other file:
        python lingua_vs_langdetect_sample.py path/to/real_logged_messages.txt
"""

import os
import sys
import time

from lingua import Language, LanguageDetectorBuilder
import langdetect

# --- Project priority languages: Malay (MY), Indonesian (ID), English (EN). ---
# These three are the actual target user base and get equal weight in
# test_case.txt. Chinese and Tamil are kept in as realistic secondary cases
# (a Malaysian classroom plausibly has both) but are intentionally lighter —
# see the secondary-language samples in tiers 3 and 4 of test_case.txt.
#
# Constraining the candidate set is a real accuracy/speed lever (see earlier
# discussion: fewer competing languages = fewer ways for n-gram fingerprints
# to collide). Malay and Indonesian are deliberately both included even
# though they're the closest pair in this set — that's the actual point:
# this list is itself a live test of whether two priority languages confuse
# each other once both are candidates, not just a coverage decision.
LANGUAGES = [Language.ENGLISH, Language.MALAY, Language.INDONESIAN, Language.CHINESE, Language.TAMIL]

_build_start = time.perf_counter()
detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
_build_seconds = time.perf_counter() - _build_start

# --- Load test cases from test_case.txt: one message per line. ---
# Put this file in the same folder as this script (or pass a path as argv[1]).
# Blank lines and lines starting with # are skipped, so you can comment notes
# directly into the file.
TEST_CASE_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_case_2.txt"

if not os.path.exists(TEST_CASE_FILE):
    print(f"Could not find {TEST_CASE_FILE!r}.")
    print("Create it next to this script (one message per line), or run:")
    print(f"    python {os.path.basename(__file__)} path/to/your_file.txt")
    sys.exit(1)

with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
    test_cases = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

if not test_cases:
    print(f"{TEST_CASE_FILE!r} was found but had no usable lines.")
    sys.exit(1)

print(f"Loaded {len(test_cases)} test case(s) from {TEST_CASE_FILE!r}\n")
print(
    "Note: lingua is restricted to "
    f"{', '.join(l.name for l in LANGUAGES)} above.\n"
    "langdetect has no equivalent restriction — it always considers its full\n"
    "~55-language set. This is an inherent asymmetry between the two libraries,\n"
    "not a setup mistake, and is itself one of the things worth comparing.\n"
)

print("=" * 90)
print(f"{'INPUT':<48} | {'langdetect':<12} | {'lingua (top)':<14} | lingua confidence")
print("=" * 90)

for text in test_cases:
    # --- langdetect: bare string, can vary between runs unless seeded ---
    try:
        langdetect_result = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        langdetect_result = "FAILED"  # langdetect throws on very short/ambiguous input

    # --- lingua: structured Language enum + confidence values ---
    lingua_lang = detector.detect_language_of(text)
    lingua_lang_str = lingua_lang.name if lingua_lang else "None"

    confidences = detector.compute_language_confidence_values(text)
    top_conf = confidences[0] if confidences else None
    conf_str = f"{top_conf.language.name}={top_conf.value:.2f}" if top_conf else "n/a"

    print(f"{text:<48} | {langdetect_result:<12} | {lingua_lang_str:<14} | {conf_str}")

print("\n" + "=" * 90)
print("MALAY vs INDONESIAN CONFIDENCE GAP (the project's core priority-language test)")
print("=" * 90)
print(
    "For each line lingua calls MALAY or INDONESIAN, this shows how close the\n"
    "runner-up language's confidence score was. A small gap means lingua is\n"
    "genuinely unsure between your two priority languages, not just choosing\n"
    "one arbitrarily — that distinction matters more here than getting every\n"
    "single line 'right', since MY and ID are linguistically very close.\n"
)

for text in test_cases:
    confidences = detector.compute_language_confidence_values(text)
    if not confidences:
        continue
    top = confidences[0]
    if top.language not in (Language.MALAY, Language.INDONESIAN):
        continue
    runner_up = next((c for c in confidences[1:] if c.language in (Language.MALAY, Language.INDONESIAN)), None)
    gap_str = f"gap={top.value - runner_up.value:.2f} vs {runner_up.language.name}" if runner_up else "no MY/ID runner-up"
    print(f"{text[:50]:<52} -> {top.language.name:<11} ({top.value:.2f}) | {gap_str}")

print("\n" + "=" * 90)
print("MIXED-LANGUAGE SEGMENT DETECTION (lingua only — langdetect has no equivalent)")
print("=" * 90)
print("(Only showing lines from your file that lingua actually splits into 2+ segments)\n")

found_any_mixed = False
for text in test_cases:
    segments = detector.detect_multiple_languages_of(text)
    distinct_langs = {seg.language for seg in segments}
    if len(distinct_langs) > 1:
        found_any_mixed = True
        print(f"Input: {text!r}")
        for seg in segments:
            snippet = text[seg.start_index:seg.end_index]
            print(f"  {seg.language.name:<10} -> {snippet!r}")
        print()

if not found_any_mixed:
    print("No code-switched lines detected in this file — try adding a mixed "
          "Malay/English message to test_case.txt to see this in action.")

print("\n" + "=" * 90)
print("DETERMINISM CHECK — run the same input 5x")
print("=" * 90)

repeat_input = "ok can or not"
print(f"\nInput: {repeat_input!r}")
print("langdetect over 5 runs:", [langdetect.detect(repeat_input) for _ in range(5)])
print("lingua over 5 runs:    ",
      [detector.detect_language_of(repeat_input).name if detector.detect_language_of(repeat_input) else None
       for _ in range(5)])

print("\n" + "=" * 90)
print("SPEED BENCHMARK")
print("=" * 90)
print(
    "This times actual calls on your machine — it is NOT lingua's published\n"
    "benchmark numbers (those came from a different test set, different machine,\n"
    "and a much larger workload: 3000 texts x 75 languages). Treat this as a\n"
    "relative comparison on YOUR data, not something comparable to those figures.\n"
)

REPEATS = 50  # passes through the whole file; raise this for a steadier average

print(f"Detector build / model load time (lingua, one-time cost): {_build_seconds:.4f}s")
print(f"Running {REPEATS} pass(es) over {len(test_cases)} line(s) "
      f"({REPEATS * len(test_cases)} total detections per library)...\n")

# --- langdetect timing ---
_start = time.perf_counter()
for _ in range(REPEATS):
    for text in test_cases:
        try:
            langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            pass  # same failure handling as above — don't let one line break timing
langdetect_total = time.perf_counter() - _start
langdetect_calls = REPEATS * len(test_cases)
langdetect_avg_ms = (langdetect_total / langdetect_calls) * 1000

# --- lingua timing ---
_start = time.perf_counter()
for _ in range(REPEATS):
    for text in test_cases:
        detector.detect_language_of(text)
lingua_total = time.perf_counter() - _start
lingua_calls = REPEATS * len(test_cases)
lingua_avg_ms = (lingua_total / lingua_calls) * 1000

print(f"{'Library':<14} | {'Total time':<12} | {'Avg per call':<14} | Calls")
print("-" * 60)
print(f"{'langdetect':<14} | {langdetect_total:>9.4f}s  | {langdetect_avg_ms:>10.4f}ms  | {langdetect_calls}")
print(f"{'lingua':<14} | {lingua_total:>9.4f}s  | {lingua_avg_ms:>10.4f}ms  | {lingua_calls}")

if lingua_total > 0:
    ratio = langdetect_total / lingua_total
    faster = "lingua" if ratio > 1 else "langdetect"
    print(f"\n{faster} was ~{max(ratio, 1/ratio):.1f}x faster on this file, on this machine.")

print(
    "\nNote: with very few lines in test_case.txt, this mostly measures noise —\n"
    "raise REPEATS, or better, run this against a real sample of logged production\n"
    "messages (hundreds+) for a number worth quoting in your write-up."
)
