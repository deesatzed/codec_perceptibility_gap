"""Standalone measured-energy run for Stage B (RUN THIS IN YOUR OWN TERMINAL).

powermetrics needs sudo, and sudo needs a real TTY to prompt for a password,
which the assistant's non-interactive shell does not have. So run this yourself:

    cd /Volumes/WS4TB/codec-exper/codec_codex_experiments
    source .venv/bin/activate
    sudo -v                       # prompts for YOUR password in YOUR terminal
    python -m src.dern_b.run_energy

It prints, in order:
  1) RAW powermetrics format (so the parser can be verified against your machine)
  2) a measured-energy reading over one real cascade route (total + idle-subtracted)

Nothing is fabricated: if the format is unrecognized or sudo lapses, joules is
reported as None / 'unavailable'. Paste the full output back to fold real numbers
into results/dern_stage_b_results.md.
"""
from __future__ import annotations

import json

from src.dern_b.energy_probe import measure_energy, inspect_format, sudo_available, _component_power_samples
from src.dern_b.runtime_b import DERNBRuntime
from src.dern_b.prompts import FACTUAL


def main() -> int:
    print("=== sudo available (non-interactive)? ===")
    print(sudo_available())

    print("\n=== RAW powermetrics format (verify parser against this) ===")
    raw = inspect_format(n_samples=2, interval_ms=300)
    # print only the lines likely to contain power, to keep it short
    for line in raw.splitlines():
        if "Power" in line or "power" in line or line.strip().endswith("mW"):
            print(line)
    print("\n--- parser sees these per-sample component sums (Watts) ---")
    print(_component_power_samples(raw))

    print("\n=== measured energy over a BATCH of real cascade routes ===")
    print("(batched so the work window spans several seconds; a single 0.27s route")
    print(" is shorter than one 200ms sample and would be noise-dominated.)")
    rt = DERNBRuntime(epsilon=0.5, audit_prob=1.0, max_tokens=256, seed=0)
    holder = {}

    def batch():
        recs = [rt.route(p) for p in FACTUAL]
        holder["recs"] = recs
        return recs

    _, reading = measure_energy(batch, interval_ms=200, idle_samples=5)
    recs = holder.get("recs", [])
    n = max(len(recs), 1)
    served_cheap = sum(1 for r in recs if r["served"] == "cheap")
    out = {
        "n_routes": len(recs),
        "cheap_served": served_cheap,
        "energy_for_batch": {
            "source": reading.source,
            "work_seconds": reading.work_seconds,
            "joules_total": reading.joules_total,
            "joules_idle_subtracted": reading.joules_idle_subtracted,
            "avg_watts_active": reading.avg_watts_active,
            "avg_watts_idle": reading.avg_watts_idle,
            "n_active_samples": reading.n_active_samples,
            "n_idle_samples": reading.n_idle_samples,
            "note": reading.note,
        },
    }
    if reading.joules_idle_subtracted is not None and len(recs):
        out["per_route_joules_idle_subtracted_mean"] = round(reading.joules_idle_subtracted / n, 3)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
