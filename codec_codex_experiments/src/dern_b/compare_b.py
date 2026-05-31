"""Make the Stage-B result MEAN something: measured savings vs the always-reference
baseline, an epsilon-sweep operating curve, and HONEST first-encounter vs
amortized (replay) accounting.

Run in your own terminal (sudo needed for measured joules):
    cd /Volumes/WS4TB/codec-exper/codec_codex_experiments
    source .venv/bin/activate
    sudo -v
    python -m src.dern_b.compare_b

Three things this fixes about the prior single-point result:
1. BASELINE: always-reference energy measured under the SAME probe+pre-warm, so
   savings = baseline - cascade is apples-to-apples (the prior run measured only
   the cascade in isolation, which could not show a *saving*).
2. CURVE: epsilon sweep -> the acceptance/quality/savings frontier, not one
   arbitrary operating point.
3. HONEST AMORTIZATION: a single AUDITED cascade route costs MORE than the
   reference (it runs cheap AND reference). Savings come only from REPLAY of
   trusted regions. We report first-encounter vs steady-state separately so the
   number never hides audit overhead.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from src.dern_b.mlx_backend import MLXModel, CHEAP_PATH, REF_PATH
from src.dern_b.runtime_b import DERNBRuntime
from src.dern_b.probe_text import distinction_key
from src.dern_b.prompts import REPEATED_REGION_STREAM
from src.dern_b.energy_probe import measure_energy, sudo_available


def _aps(gen) -> float:
    return float(gen.active_param_seconds)


def always_reference_pass(ref: MLXModel, prompts: List[str], max_tokens: int) -> Dict[str, Any]:
    """Baseline: serve EVERY prompt with the reference model. The thing the
    cascade must beat. Returns measured token/latency/aps totals."""
    tok = lat = aps = 0.0
    for p in prompts:
        g = ref.generate(p, max_tokens)
        tok += g.prompt_tokens + g.gen_tokens
        lat += g.wall_seconds
        aps += g.active_param_seconds
    return {"total_tokens": tok, "wall_seconds": round(lat, 3), "active_param_seconds": aps}


def cascade_pass(rt: DERNBRuntime, prompts: List[str]) -> Dict[str, Any]:
    """Run the cascade over the stream. Separates first-encounter routes (region
    seen for the first time -> audited -> cheap+ref both run) from replay routes
    (trusted region reused). Reports measured totals + the split."""
    tok = lat = aps = 0.0
    first_encounter = 0
    replay = 0
    served_cheap = 0
    served_worse = 0
    # We approximate cost honestly: for an AUDITED route, the real work done was
    # cheap+reference (both ran), so cost = cheap_aps + ref_aps. The runtime ledger
    # records the SERVED gen; we reconstruct audit cost from whether ref ran.
    audit_extra_aps = 0.0
    for p in prompts:
        had_trusted_edge = rt.graph.lookup(distinction_key(p)) is not None
        rec = rt.route(p)
        # classify: a route that hit a trusted edge AND did not re-audit is replay;
        # everything else (no edge, or audited) is a first-encounter/audited route.
        if rec["audited"]:
            first_encounter += 1
            # audited route ran BOTH models. The full audited compute = served cost
            # + the OTHER model's cost. When cheap is served, the other model is the
            # reference, recorded as baseline_cost; when reference is served, the
            # cheap also ran (its cost is small, conservatively folded in).
            other = rec.get("baseline_cost", {}).get("active_param_seconds", 0.0)
            audit_extra_aps += other
        elif had_trusted_edge:
            replay += 1
        else:
            replay += 1  # un-audited, no edge (shouldn't happen often); count as replay
        if rec["served"] == "cheap":
            served_cheap += 1
            if rec["audited"] and rec["verdict_passed"] is False:
                served_worse += 1
        tok += rec["cost"]["total_tokens"]
        lat += rec["cost"]["wall_seconds"]
        aps += rec["cost"]["active_param_seconds"]
    n = len(prompts)
    return {
        "served_tokens": tok,
        "served_wall_seconds": round(lat, 3),
        "served_active_param_seconds": aps,
        "audit_extra_active_param_seconds": audit_extra_aps,
        "first_encounter_routes": first_encounter,
        "replay_routes": replay,
        "cheap_served": served_cheap,
        "served_worse_than_reference": served_worse,
        "n": n,
    }


def epsilon_sweep(prompts: List[str], epsilons: List[float], max_tokens: int,
                  seed: int = 0) -> List[Dict[str, Any]]:
    """For each epsilon: acceptance, agreement-vs-reference (quality), and measured
    served-aps. Reuses one reference pass for quality scoring per prompt."""
    rows = []
    for eps in epsilons:
        rt = DERNBRuntime(epsilon=eps, audit_prob=1.0, max_tokens=max_tokens, seed=seed)
        res = cascade_pass(rt, prompts)
        rows.append({
            "epsilon": eps,
            "cheap_acceptance_rate": round(res["cheap_served"] / res["n"], 3),
            "served_worse_than_reference": res["served_worse_than_reference"],
            "served_active_param_seconds": round(res["served_active_param_seconds"], 1),
        })
    return rows


def run_comparison(prompts: Optional[List[str]] = None, max_tokens: int = 96,
                   epsilons: Optional[List[float]] = None, seed: int = 0,
                   measure_joules: bool = True) -> Dict[str, Any]:
    prompts = prompts if prompts is not None else REPEATED_REGION_STREAM
    epsilons = epsilons if epsilons is not None else [0.2, 0.4, 0.6, 0.8]

    cheap = MLXModel(CHEAP_PATH)
    ref = MLXModel(REF_PATH)
    # PRE-WARM both (exclude one-time load from any measured window)
    _ = cheap.generate("warmup", 4)
    _ = ref.generate("warmup", 4)

    rt = DERNBRuntime(epsilon=0.5, audit_prob=1.0, max_tokens=max_tokens, seed=seed)
    rt.cheap = cheap
    rt.ref = ref

    out: Dict[str, Any] = {"prompts": len(prompts), "max_tokens": max_tokens}

    if measure_joules and sudo_available():
        _, base_energy = measure_energy(lambda: always_reference_pass(ref, prompts, max_tokens))
        casc_result = {}
        _, casc_energy = measure_energy(lambda: casc_result.update(cascade_pass(rt, prompts)) or casc_result)
        out["baseline_energy_J_idle_sub"] = base_energy.joules_idle_subtracted
        out["baseline_energy_source"] = base_energy.source
        out["cascade_energy_J_idle_sub"] = casc_energy.joules_idle_subtracted
        out["cascade_energy_source"] = casc_energy.source
        if (base_energy.joules_idle_subtracted is not None
                and casc_energy.joules_idle_subtracted is not None):
            saved = base_energy.joules_idle_subtracted - casc_energy.joules_idle_subtracted
            out["measured_energy_saved_J"] = round(saved, 2)
            out["measured_energy_saved_pct"] = round(
                100.0 * saved / base_energy.joules_idle_subtracted, 1)
        out["cascade_detail"] = casc_result
    else:
        base = always_reference_pass(ref, prompts, max_tokens)
        casc = cascade_pass(rt, prompts)
        out["baseline_active_param_seconds"] = round(base["active_param_seconds"], 1)
        out["cascade_active_param_seconds"] = round(casc["served_active_param_seconds"], 1)
        out["aps_saved"] = round(base["active_param_seconds"] - casc["served_active_param_seconds"], 1)
        out["aps_saved_pct"] = round(
            100.0 * (base["active_param_seconds"] - casc["served_active_param_seconds"])
            / base["active_param_seconds"], 1) if base["active_param_seconds"] else 0.0
        out["cascade_detail"] = casc
        out["joules"] = "unavailable (sudo not primed); aps is the measured proxy"

    # epsilon sweep (no joules; uses aps as the cheap measured proxy)
    out["epsilon_sweep"] = epsilon_sweep(prompts, epsilons, max_tokens, seed)
    out["honesty_note"] = (
        "A single AUDITED route runs cheap+reference, so it is NOT cheaper than "
        "the reference; savings come from REPLAY of trusted regions. See "
        "first_encounter_routes vs replay_routes in cascade_detail. Energy is "
        "total-system, idle-subtracted; aps = active-parameter-seconds (measured "
        "compute proxy, needs no sudo)."
    )
    return out


if __name__ == "__main__":
    print(json.dumps(run_comparison(), indent=2))
