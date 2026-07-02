"""
tests/test_pipeline_e2e.py — end-to-end pipeline tests.

tests/test_core.py only covers the pure, model-free functions in voting/core.py (bucketing,
vote-counting arithmetic, path numbering). It never loads a real model or runs the actual
voting/scoring pipeline that produces the headline numbers in the report. These tests close that
gap: they run tests/fixtures/e2e_fixture.txt (18 hand-labeled cases, 6 each EN/MY/ID, drawn
verbatim from data/test_case_7_enmyid.txt's "3-7 words" bucket) through the real S1 pipeline
(lingua-high + langdetect + pycld2, via voting.core.run_predictions) and the real S2 pipeline
(lingua-high + openlid-v3 + pycld2, via voting.strategies), and assert accuracy stays within an
expected range.

Scope and honesty about what this does NOT prove:
  - The fixture is 18 cases, not 1,273 — it is a regression guard against pipeline-wiring bugs
    (wrong argument order, broken imports, a vote function silently returning "unknown" for
    everything), not a re-validation of the report's statistical findings. Use the full dataset
    and the scripts in src/voting/ for that.
  - S2 requires models/openlid-v3.bin (1.2 GB), which is excluded from the repo via .gitignore
    (see README's Setup section) and is NOT present in CI. The S2 test is skipped automatically
    when that file is missing — see the CI badge caveat in README.md's Tests section.
"""

from pathlib import Path

import pytest

from voting.core import (
    ROOT, load_dataset, load_lingua, run_predictions, overall_accuracy, lingua_probs, pycld2_probs,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "e2e_fixture.txt"
OPENLID_PATH = ROOT / "models" / "openlid-v3.bin"


@pytest.fixture(scope="module")
def fixture_cases():
    cases = load_dataset(FIXTURE_PATH)
    assert len(cases) == 18
    return cases


def test_s1_pipeline_hard_vote_accuracy_in_expected_range(fixture_cases):
    """Full S1 pipeline (lingua-high + langdetect + pycld2, voting.core.run_predictions):
    hard vote should stay in a healthy range on this fixture (measured: 17/18 = 94.4%)."""
    detector = load_lingua()
    results = run_predictions(fixture_cases, detector)
    acc = overall_accuracy(results, "hard")
    assert 0.75 <= acc <= 1.0, f"S1 hard-vote accuracy {acc:.2%} outside expected [75%, 100%] range"


def test_s1_pipeline_weighted_vote_accuracy_in_expected_range(fixture_cases):
    """Full S1 pipeline, weighted vote (measured: 16/18 = 88.9%)."""
    detector = load_lingua()
    results = run_predictions(fixture_cases, detector)
    acc = overall_accuracy(results, "weighted")
    assert 0.70 <= acc <= 1.0, f"S1 weighted-vote accuracy {acc:.2%} outside expected [70%, 100%] range"


@pytest.mark.skipif(not OPENLID_PATH.exists(), reason="models/openlid-v3.bin not present (excluded via .gitignore, not fetched in CI)")
def test_s2_pipeline_weighted_vote_accuracy_in_expected_range(fixture_cases):
    """Full S2 pipeline (lingua-high + openlid-v3 + pycld2, voting.strategies), weighted vote
    (measured: 18/18 = 100% on this fixture)."""
    import fasttext
    from voting.strategies import openlid_probs
    from voting.strategies.weighted import weighted_vote_3, S2_WEIGHTS

    detector = load_lingua()
    openlid_model = fasttext.load_model(str(OPENLID_PATH))

    correct = 0
    for case in fixture_cases:
        text = case["text"]
        lp = lingua_probs(detector, text)
        op = openlid_probs(openlid_model, text)
        cp = pycld2_probs(text)
        pred = weighted_vote_3(lp, op, cp, S2_WEIGHTS)
        correct += int(pred == case["expected_iso"])

    acc = correct / len(fixture_cases)
    assert 0.70 <= acc <= 1.0, f"S2 weighted-vote accuracy {acc:.2%} outside expected [70%, 100%] range"
