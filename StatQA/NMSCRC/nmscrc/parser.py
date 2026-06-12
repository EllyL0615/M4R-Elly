"""Raw-LLM-answer parser — reads `model_answer`. The ONLY place 3B's echo matters.

Parse recipe (preserved verbatim):
  - prepend '{"methods": ["'
  - strip trailing period
  - json.loads
  - keep only names in the 27-set
ADDED: per-row classification into a `flag`, and per-model rates. Scoring is
UNCHANGED — echo / all_unknown / genuine_empty / unparseable all score as the EMPTY set
(F1 = 0, the fair penalty for a failed baseline) and are NEVER dropped. The breakdown is
reported so the raw-LLM row can be read honestly (template-echo fault vs genuine no-prediction).

`is_echo` must NOT over-match: a row with >=1 valid 27-method is always `ok` (never echo), and
assert_echo_pattern_safe() verifies no real method name matches the echo regex (Gate-1).
"""

import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Template/placeholder tokens an LLM emits when it echoes the prompt scaffold instead of answering.
# Each alternative must FULL-match a single name token; the 27 method names are full phrases
# ("Pearson Correlation Coefficient", ...), so these cannot collide (verified by
# assert_echo_pattern_safe at build time, Gate-1).
_ECHO_RE = re.compile(
    r"^(?:"
    r"m\d+"               # m1, m2, m3, ...
    r"|method[_ ]?\d+"    # method_1, method1, method 1
    r"|method[_ ]?name"   # method_name, methodname
    r"|name\d*"           # name, name1
    r"|placeholder\w*"    # placeholder, placeholder1
    r"|<[^>]*>"           # <...>, <method>
    r"|\.{2,}"            # .., ...
    r"|method"            # bare "method"
    r")$",
    re.IGNORECASE,
)

FLAGS = ("ok", "echo", "all_unknown", "genuine_empty", "unparseable")


def _is_echo_name(name: str) -> bool:
    return bool(_ECHO_RE.match(name.strip()))


def assert_echo_pattern_safe(method_list: List[str]) -> None:
    """Gate-1: no real 27-method name may match the echo regex."""
    bad = [m for m in method_list if _is_echo_name(m)]
    if bad:
        raise RuntimeError(f"echo pattern over-matches real method name(s): {bad}")


def _reconstruct(raw) -> Tuple[bool, List[str]]:
    """Legacy reconstruction, returning (parsed_ok, raw_names_before_27_filter).

    Verbatim rule: prepend '{"methods": ["', rstrip, drop trailing '.', json.loads.
    parsed_ok=False only when json.loads raises.
    """
    s = "" if pd.isna(raw) else str(raw)
    cleaned = '{"methods": ["' + s.strip()
    cleaned = cleaned.rstrip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return False, []
    selected = parsed.get("methods", []) if isinstance(parsed, dict) else []
    if not isinstance(selected, list):
        return True, []
    return True, [str(m) for m in selected]


def classify_answer(raw, method_set) -> Tuple[str, List[str]]:
    """Return (flag, valid_methods). valid_methods is the 27-set-filtered prediction.

    Precedence: unparseable -> genuine_empty -> ok (>=1 valid) -> echo -> all_unknown.
    A row with >=1 valid method is ALWAYS ok (echo can never steal an ok row).
    """
    parsed_ok, names = _reconstruct(raw)
    if not parsed_ok:
        return "unparseable", []
    if len(names) == 0:
        return "genuine_empty", []
    valid = [m for m in names if m in method_set]
    if valid:
        return "ok", valid
    # no valid names, but non-empty -> distinguish template-echo from genuine hallucination
    if any(_is_echo_name(n) for n in names):
        return "echo", []
    return "all_unknown", []


def parse_methods(raw, method_set) -> List[str]:
    """Back-compat shim with the legacy parser: the multi-hot prediction only.

    Identical OUTPUT to legacy parse_methods (echo/unknown/empty/unparseable -> []).
    """
    _flag, valid = classify_answer(raw, method_set)
    return valid


def build_answer_matrix(df: pd.DataFrame, method_list: List[str]) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Raw-LLM multi-hot (0/1) over the 27 methods + per-row flag + per-model rate breakdown.

    Failures score as the empty (all-zero) row — NEVER dropped.
    Returns (multi_hot_df[n x 27], flags[n], rates_dict).
    """
    assert_echo_pattern_safe(method_list)
    method_set = set(method_list)

    classified = df["model_answer"].apply(lambda r: classify_answer(r, method_set))
    flags = classified.apply(lambda t: t[0])
    selected_per_row = classified.apply(lambda t: t[1])

    out = pd.DataFrame(
        {m: selected_per_row.apply(lambda lst, m=m: int(m in lst)) for m in method_list},
        columns=method_list,
    )

    n = len(df)
    counts = {f: int((flags == f).sum()) for f in FLAGS}
    rates = {
        "n": n,
        "counts": counts,
        "echo_rate": counts["echo"] / n if n else 0.0,
        "all_unknown_rate": counts["all_unknown"] / n if n else 0.0,
        "genuine_empty_rate": counts["genuine_empty"] / n if n else 0.0,
        "unparseable_rate": counts["unparseable"] / n if n else 0.0,
        "ok_rate": counts["ok"] / n if n else 0.0,
        "all_zero_rows": int((out.sum(axis=1) == 0).sum()),
        "mean_selected": float(out.sum(axis=1).mean()) if n else 0.0,
    }
    return out, flags, rates
