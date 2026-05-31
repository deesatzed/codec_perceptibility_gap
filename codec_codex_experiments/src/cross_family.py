#!/usr/bin/env python3
"""Cross-family orchestration (F5 mitigation, Phase 5).

Runs the proof ladder and codec contest across every configured continuous
source family, then builds a gate matrix marking which gates are family-robust
(PASS on >=2 families) versus family-specific.

Categorical families (e.g. magnetic) keep their own dedicated runner
(`run_magnetic_slope`) because their targets are labels, not continuous theta;
they are surfaced in the same results bundle but not forced through the
continuous nRMSE ladder. No gate threshold is tuned per family: a family FAIL
is reported as an honest result.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .proof_ladder import run_ladder, normalize_cfg
from .codec_contest import run_codec_contest
from .sources import resolve_sources


LADDER_GATES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def run_cross_family(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_cfg(raw_cfg)
    families = resolve_sources(cfg)

    by_source: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        if getattr(fam, "target_kind", "continuous") != "continuous":
            # Categorical families are reported via their own runner elsewhere.
            continue
        ladder = run_ladder(cfg, source=fam)
        contest = run_codec_contest(cfg, source=fam)
        by_source[fam.key] = {"proof_ladder": ladder, "codec_contest": contest}

    matrix = _gate_matrix(by_source)
    audit = _audit_survival(by_source)
    return {
        "families_run": list(by_source.keys()),
        "by_source": by_source,
        "gate_matrix": matrix,
        "codecguard_audit_survival": audit,
        "interpretation": (
            "family_robust gates PASS on >=2 families; family_specific gates "
            "PASS on only one. A FAIL everywhere is an honest negative, not a "
            "threshold to tune. `codecguard_audit_survival` is the headline "
            "cross-family question (critic1 #3): does the disagreement->error "
            "audit signal survive on a non-oscillatory chaotic family? The "
            "hardened LOCO correlation and worst-case catch-rate are the "
            "physics-independent signals; the naive correlation / permutation "
            "null may degrade on the chaotic family and that is reported as-is."
        ),
    }


def _audit_survival(by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-family CodecGuard survival summary so the cross-family report can
    answer 'does the audit signal generalize beyond oscillator physics?'
    without forcing any ladder gate to pass. Pulls the already-computed,
    leakage-hardened metrics straight from each family's codec contest."""
    out: Dict[str, Any] = {}
    for key, bundle in by_source.items():
        a = bundle["codec_contest"]["multi_codec_audit"]
        catch = a.get("catch_rate_top10pct_error_at_top25pct_disagreement")
        out[key] = {
            "naive_corr": a.get("corr_disagreement_vs_ensemble_error"),
            "naive_corr_sd": a.get("corr_disagreement_vs_ensemble_error_sd"),
            "loco_corr": a.get("loco_disagreement_vs_held_codec_error_corr"),
            "all_seeds_clear_permutation_null": a.get("all_seeds_clear_permutation_null"),
            "catch_rate": catch,
            "random_catch_baseline": a.get("random_catch_baseline", 0.25),
            # Physics-independent survival: hardened LOCO is positive AND the
            # worst-error catch-rate beats the random baseline. This is the
            # claim that does not lean on continuous-momentum invariants.
            "audit_signal_survives": bool(
                (a.get("loco_disagreement_vs_held_codec_error_corr") or 0.0) > 0.10
                and (catch or 0.0) > a.get("random_catch_baseline", 0.25)
            ),
        }
    return out


def _gate_matrix(by_source: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(by_source.keys())
    per_gate: Dict[str, Dict[str, bool]] = {}
    for gate in LADDER_GATES:
        per_gate[gate] = {
            k: bool(by_source[k]["proof_ladder"].get(gate, {}).get("passed", False)) for k in keys
        }
    robust: List[str] = []
    specific: List[str] = []
    failing: List[str] = []
    for gate, results in per_gate.items():
        n_pass = sum(1 for v in results.values() if v)
        if n_pass >= 2:
            robust.append(gate)
        elif n_pass == 1:
            specific.append(gate)
        else:
            failing.append(gate)
    return {
        "per_gate_pass_by_family": per_gate,
        "family_robust_gates": robust,
        "family_specific_gates": specific,
        "failing_on_all_families": failing,
    }
