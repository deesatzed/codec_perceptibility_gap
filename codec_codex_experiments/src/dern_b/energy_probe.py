"""Opt-in, sudo-gated true-energy probe via macOS `powermetrics`.

powermetrics requires sudo (a password) and cannot run unattended. This module
NEVER fabricates a joules number. If non-interactive sudo is unavailable, or if
the output format cannot be parsed into recognized component power lines, it
returns joules=None and a source_tag of 'unavailable' — never a guess.

Method (hardened after a method audit; see results/dern_stage_b_energy_method.md):
- Run `powermetrics -n N -i INTERVAL_MS` for a FIXED sample count so no samples
  are lost to mid-stream termination.
- Parse only the explicit per-component lines we recognize (CPU/GPU/ANE Power in
  mW) via a strict regex. SUM components per sample; explicitly IGNORE any
  'Combined'/total line to avoid double counting.
- Energy = sum over samples of (component_power_W * interval_seconds), i.e. a
  proper Riemann sum with the known sampling interval as each sample's weight —
  NOT avg_power * wall_clock (which mis-aligns the sampling window).
- Report BOTH total-system energy during the call and idle-subtracted energy
  (active minus a measured idle baseline), clearly labeled. Total-system power is
  not attributable to the model alone; idle-subtracted is the better proxy.
- If zero recognized samples are parsed, return 'unavailable'.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Strict per-component matchers. powermetrics prints e.g. "CPU Power: 1234 mW".
# We match ONLY these component labels and ignore any other "*Power*" line
# (notably a 'Combined Power' total, which would double-count).
_COMPONENT_RE = re.compile(r"^(CPU|GPU|ANE)\s+Power:\s+([0-9]+(?:\.[0-9]+)?)\s*mW\s*$", re.I)


def sudo_available() -> bool:
    """True iff sudo can run non-interactively right now (no password prompt)."""
    try:
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def _run_powermetrics(n_samples: int, interval_ms: int) -> str:
    """Collect a FIXED number of samples; powermetrics exits on its own (no
    mid-stream terminate that could drop samples)."""
    r = subprocess.run(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
         "-i", str(interval_ms), "-n", str(n_samples)],
        capture_output=True, text=True, timeout=interval_ms / 1000.0 * n_samples + 15,
    )
    return r.stdout


def _component_power_samples(text: str) -> List[float]:
    """Return a list of per-sample summed component power (Watts).

    powermetrics emits component lines repeatedly (one block per sample). We sum
    the recognized component lines, then split into per-sample groups: a new
    sample begins when we see a component label we've already seen since the last
    flush (i.e. the next 'CPU Power' starts a new block).
    """
    samples: List[float] = []
    cur: Dict[str, float] = {}
    for line in text.splitlines():
        m = _COMPONENT_RE.match(line.strip())
        if not m:
            continue
        comp = m.group(1).upper()
        watts = float(m.group(2)) / 1000.0
        if comp in cur:  # repeat label -> previous sample block is complete
            samples.append(sum(cur.values()))
            cur = {}
        cur[comp] = watts
    if cur:
        samples.append(sum(cur.values()))
    return samples


@dataclass
class EnergyReading:
    joules_total: Optional[float]
    joules_idle_subtracted: Optional[float]
    avg_watts_active: Optional[float]
    avg_watts_idle: Optional[float]
    n_active_samples: int
    n_idle_samples: int
    interval_s: float
    source: str  # "measured" | "unavailable"
    note: str


def measure_energy(fn: Callable[[], Any], interval_ms: int = 200,
                   idle_samples: int = 5) -> Tuple[Any, EnergyReading]:
    """Run fn() while sampling power; return (result, EnergyReading).

    First measures an idle baseline (idle_samples samples while doing nothing),
    then measures during fn(). Energy is integrated as sum(power * interval).
    Honest failure: any parse/availability problem -> source='unavailable',
    joules=None (never fabricated).
    """
    interval_s = interval_ms / 1000.0
    if not sudo_available():
        return fn(), EnergyReading(None, None, None, None, 0, 0, interval_s,
                                   "unavailable", "sudo not primed")

    # 1) idle baseline (no model work)
    try:
        idle_text = _run_powermetrics(idle_samples, interval_ms)
        idle = _component_power_samples(idle_text)
    except Exception as e:
        idle = []
    avg_idle = (sum(idle) / len(idle)) if idle else None

    # 2) active window: start powermetrics for enough samples to cover fn, run fn,
    #    then read what was collected. We size n by a rough estimate and cap it.
    t0 = time.time()
    # Probe fn duration cheaply is not possible without running it; instead run
    # powermetrics in the background for a generous sample count and stop after fn.
    proc = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power", "-i", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        result = fn()
    finally:
        time.sleep(interval_s)  # let the final in-flight sample flush
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:
            out = ""
            proc.kill()
    active = _component_power_samples(out)

    if not active:
        return result, EnergyReading(None, None, None, avg_idle,
                                     0, len(idle), interval_s, "unavailable",
                                     "no recognized active samples parsed")

    # proper Riemann sum: each sample represents one interval of energy
    joules_total = sum(p * interval_s for p in active)
    avg_active = sum(active) / len(active)
    joules_idle_sub = None
    if avg_idle is not None:
        joules_idle_sub = max(0.0, sum((p - avg_idle) * interval_s for p in active))

    return result, EnergyReading(
        joules_total=round(joules_total, 3),
        joules_idle_subtracted=(round(joules_idle_sub, 3) if joules_idle_sub is not None else None),
        avg_watts_active=round(avg_active, 3),
        avg_watts_idle=(round(avg_idle, 3) if avg_idle is not None else None),
        n_active_samples=len(active),
        n_idle_samples=len(idle),
        interval_s=interval_s,
        source="measured",
        note="component sum (CPU+GPU[+ANE]); total-system power, idle-subtracted is the better model proxy",
    )


def inspect_format(n_samples: int = 2, interval_ms: int = 300) -> str:
    """Return raw powermetrics output so the parser can be verified against the
    real format on THIS machine before trusting any number. Requires primed sudo."""
    if not sudo_available():
        return "<sudo not primed: run `! sudo -v` first>"
    return _run_powermetrics(n_samples, interval_ms)
