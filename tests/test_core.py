"""
tests/test_core.py — unit tests for the pure, model-free functions in voting/core.py.

No models are loaded here (no lingua/langdetect/pycld2 calls) — only the deterministic
helpers: bucketing, voting strategies, path numbering, and accuracy aggregation.
"""

from voting.core import (
    bucket_for, is_cjk, next_path,
    hard_vote, soft_vote, weighted_vote, pick_top,
    overall_accuracy, accuracy_by_lang, binary_correct, pred_labels, pct,
)


# ── bucket_for / is_cjk ──────────────────────────────────────────────────────────
def test_bucket_for_word_counts():
    assert bucket_for("hello") == "1 word"
    assert bucket_for("hello world") == "2 words"
    assert bucket_for("one two three four five") == "3-7 words"
    assert bucket_for(" ".join(["word"] * 10)) == "8-16 words"
    assert bucket_for(" ".join(["word"] * 20)) == "17-50 words"


def test_is_cjk_detects_chinese_text():
    assert is_cjk("你好世界") is True
    assert is_cjk("hello world") is False


def test_bucket_for_cjk_uses_character_count_not_word_count():
    # CJK text has no spaces, so bucketing must fall back to character count
    assert bucket_for("你好") == "1 word"
    assert bucket_for("你好世界你好世界你好") == "3-7 words"


# ── pick_top ─────────────────────────────────────────────────────────────────────
def test_pick_top_returns_highest_probability_language():
    assert pick_top({"en": 0.1, "ms": 0.2, "id": 0.7}) == "id"


def test_pick_top_returns_unknown_when_all_zero():
    assert pick_top({"en": 0.0, "ms": 0.0, "id": 0.0}) == "unknown"


# ── hard_vote ──────────────────────────────────────────────────────────────────
def test_hard_vote_majority_wins():
    assert hard_vote("en", "en", "id") == "en"


def test_hard_vote_three_way_tie_breaks_to_lingua():
    assert hard_vote("en", "id", "ms") == "en"


def test_hard_vote_ignores_unknown_votes():
    assert hard_vote("unknown", "id", "id") == "id"


def test_hard_vote_all_unknown_returns_unknown():
    assert hard_vote("unknown", "unknown", "unknown") == "unknown"


# ── soft_vote / weighted_vote ────────────────────────────────────────────────────
def test_soft_vote_picks_highest_average_probability():
    result = soft_vote(
        {"en": 0.9, "ms": 0.05, "id": 0.05},
        {"en": 0.1, "ms": 0.1, "id": 0.8},
        {"en": 0.2, "ms": 0.1, "id": 0.7},
    )
    assert result == "id"


def test_weighted_vote_respects_custom_weights():
    # model A says 'en' with weight 10, model B says 'id' with weight 1 -> 'en' wins
    result = weighted_vote(
        {"en": 1.0, "id": 0.0}, {"en": 0.0, "id": 1.0}, {"en": 0.0, "id": 0.0},
        weights={"lingua": 10.0, "langdetect": 1.0, "pycld2": 1.0},
    )
    assert result == "en"


# ── next_path ────────────────────────────────────────────────────────────────────
def test_next_path_finds_first_unused_number(tmp_path):
    assert next_path(tmp_path, "log", ".txt") == tmp_path / "log_1.txt"


def test_next_path_skips_existing_files(tmp_path):
    (tmp_path / "log_1.txt").touch()
    (tmp_path / "log_2.txt").touch()
    assert next_path(tmp_path, "log", ".txt") == tmp_path / "log_3.txt"


# ── accuracy helpers ─────────────────────────────────────────────────────────────
SAMPLE_RESULTS = [
    {"expected_lbl": "EN", "expected_iso": "en", "hard": "en"},
    {"expected_lbl": "EN", "expected_iso": "en", "hard": "id"},
    {"expected_lbl": "MY", "expected_iso": "ms", "hard": "ms"},
    {"expected_lbl": "MY", "expected_iso": "ms", "hard": "ms"},
]


def test_overall_accuracy():
    assert overall_accuracy(SAMPLE_RESULTS, "hard") == 0.75


def test_overall_accuracy_empty_results_is_zero():
    assert overall_accuracy([], "hard") == 0.0


def test_accuracy_by_lang():
    by_lang = accuracy_by_lang(SAMPLE_RESULTS, "hard")
    assert by_lang["EN"] == [1, 2]
    assert by_lang["MY"] == [2, 2]


def test_binary_correct_filtered_by_language():
    arr = binary_correct(SAMPLE_RESULTS, "hard", lang_filter="EN")
    assert arr.tolist() == [1, 0]


def test_pred_labels_filtered_by_language():
    assert pred_labels(SAMPLE_RESULTS, "hard", lang_filter="MY") == ["ms", "ms"]


def test_pct_formats_percentage():
    assert pct(3, 4).strip() == "75.0%"


def test_pct_handles_zero_total():
    assert pct(0, 0).strip() == "N/A"
