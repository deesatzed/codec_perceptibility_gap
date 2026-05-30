"""Phase 0 regression guard for the F5 cross-family refactor.

Freezes the linear-family proof-ladder and codec-contest metrics so that the
SourceFamily refactor cannot silently change the canonical linear results.
The fixture is deterministic (same seed/estimator -> identical numbers); the
tolerance is therefore tight.
"""
import json
from pathlib import Path

from src.proof_ladder import normalize_cfg, run_ladder
from src.codec_contest import run_codec_contest

FIXTURE = Path(__file__).parent / "fixtures" / "baseline_linear.json"
TOL = 1e-6


def _flatten(obj, prefix=""):
    """Yield (path, float) for every numeric leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        yield (prefix, obj)
    elif isinstance(obj, (int, float)):
        yield (prefix, float(obj))


def _compare(expected, actual):
    exp = dict(_flatten(expected))
    act = dict(_flatten(actual))
    mismatches = []
    for path, ev in exp.items():
        av = act.get(path, None)
        if av is None:
            mismatches.append(f"{path}: missing in actual")
        elif isinstance(ev, bool):
            if ev != av:
                mismatches.append(f"{path}: {ev} != {av}")
        elif abs(ev - av) > TOL:
            mismatches.append(f"{path}: {ev} != {av} (|d|={abs(ev-av):.2e})")
    return mismatches


def test_linear_baseline_reproduced():
    base = json.loads(FIXTURE.read_text())
    cfg = normalize_cfg(base["config"])
    pl = run_ladder(cfg)
    cc = run_codec_contest(cfg)
    mismatches = _compare(base["proof_ladder"], pl) + _compare(base["codec_contest"], cc)
    assert not mismatches, "linear baseline drifted:\n" + "\n".join(mismatches[:30])
