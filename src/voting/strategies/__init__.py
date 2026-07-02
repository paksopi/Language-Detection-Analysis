"""
voting/strategies — Scenario 2 (lingua-high + openlid-v3 + pycld2) voting logic.

Formerly `voting/scenario2/`. Renamed so files are named after the voting technique they
implement rather than an experiment number — "S1"/"S2" only mean something with the report open
next to the code. Old name -> new name, for cross-referencing report sections (§3, §6, §7):

    voting/scenario2/voting_s2.py           -> voting/strategies/run_s2_comparison.py
                                                (hard/soft/weighted vote functions extracted to
                                                 hard.py, soft.py, weighted.py in this package)
    voting/scenario2/voting_s2_two_stage.py -> voting/strategies/run_s2_two_stage.py
                                                (two-stage decision functions extracted to
                                                 two_stage.py in this package)

"S1" in the report means the `lingua-high` + `langdetect` + `pycld2` trio (logic lives in
voting/core.py and voting/voting_two_stage.py, untouched by this rename). "S2" means the
`lingua-high` + `openlid-v3` + `pycld2` trio implemented in this package. Report §7's "Scenario 2"
and §9's "S2 hard / S2 weighted / S2 ts_*" rows all refer to the functions in this package.

This __init__ holds the openlid-v3-specific helpers shared by every strategy module: the
FLORES-200 -> ISO label map and the text preprocessor. TARGET_LANGS, LANGUAGE_ORDER,
BUCKET_ORDER, LINGUA_LANGS, load_dataset, lingua_probs, pycld2_probs, and langdetect_probs are
shared with Scenario 1 and still come from voting.core.
"""

import regex

from voting.core import TARGET_LANGS

# OpenLID FLORES-200 -> ISO mapping (strict — only confirmed mappings)
OPENLID_TO_ISO = {
    "eng_Latn": "en",
    "zsm_Latn": "ms",
    "msa_Latn": "ms",
    "ind_Latn": "id",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "cmn_Hans": "zh",   # Mandarin Chinese (simplified) — FLORES-200 variant
    "cmn_Hant": "zh",
    "tam_Taml": "ta",
}

_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACES = regex.compile(r"\s\s+")


def preprocess_openlid(text):
    text = text.strip().replace("\n", " ").lower()
    text = _SPACES.sub(" ", text)
    text = _NONWORD.sub("", text)
    return text


def openlid_label_to_iso(label):
    return OPENLID_TO_ISO.get(label.replace("__label__", ""), "unknown")


def openlid_probs(model, text):
    """Return {iso: prob} from openlid-v3. Uses top-20 FLORES-200 predictions."""
    probs = {l: 0.0 for l in TARGET_LANGS}
    processed = preprocess_openlid(text)
    if not processed.strip():
        return probs
    try:
        labels, scores = model.predict(processed, k=20)
        for label, score in zip(labels, scores):
            iso = openlid_label_to_iso(label)
            if iso in probs:
                probs[iso] += float(score)
    except Exception:
        pass
    return probs
