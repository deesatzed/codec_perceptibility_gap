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

    powermetrics prints one block per sample, each STARTING with a 'CPU Power'
    line followed by 'GPU Power' and (on Apple Silicon) 'ANE Power'. It also
    interleaves stray per-cluster 'GPU Power' lines OUTSIDE the block; those must
    not be treated as new samples. So a new sample begins ONLY on a 'CPU Power'
    line; GPU/ANE lines accumulate into the current block, and a repeated GPU/ANE
    before the next CPU is ignored (keep the first, which belongs to the block).

    Verified against real M4 Pro powermetrics output (see
    results/dern_stage_b_energy_method.md).
    """
    samples: List[float] = []
    cur: Dict[str, float] = {}
    for line in text.splitlines():
        m = _COMPONENT_RE.match(line.strip())
        if not m:
            continue
        comp = m.group(1).upper()
        watts = float(m.group(2)) / 1000.0
        if comp == "CPU":
            # CPU line marks the start of a new sample block.
            if cur:
                samples.append(sum(cur.values()))
            cur = {"CPU": watts}
        else:
            # GPU/ANE: attach to the current block only if not already set
            # (ignore stray duplicate cluster lines outside the block).
            if cur and comp not in cur:
                cur[comp] = watts
            # a GPU/ANE line with no open CPU block (stray) is ignored entirely
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
    work_seconds: Optional[float]
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
                                   None, "unavailable", "sudo not primed")

    # 1) idle baseline (no model work)
    try:
        idle_text = _run_powermetrics(idle_samples, interval_ms)
        idle = _component_power_samples(idle_text)
    except Exception as e:
        idle = []
    avg_idle = (sum(idle) / len(idle)) if idle else None

    # 2) active window. We write powermetrics to a FILE (not a pipe) for the whole
    #    window, so samples are not lost to pipe buffering / terminate truncation
    #    (a flaw the first real readings exposed: a backgrounded pipe returned only
    #    ~10 samples for a 53s window). After fn() returns, we stop powermetrics and
    #    read the file. Energy is integrated over the ACTUAL work window, and only
    #    the samples that fall within it are used.
    import tempfile
    import os
    import signal
    tmp = tempfile.NamedTemporaryFile(prefix="pm_", suffix=".txt", delete=False)
    tmp.close()
    t0 = time.time()
    with open(tmp.name, "w") as fh:
        proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power", "-i", str(interval_ms)],
            stdout=fh, stderr=subprocess.DEVNULL, text=True,
        )
        try:
            result = fn()
        finally:
            work_seconds = time.time() - t0          # the ACTUAL work window
            time.sleep(interval_s * 1.5)  # let the final sample flush to the file
            # powermetrics under sudo: terminate the sudo child group.
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    subprocess.run(["sudo", "-n", "kill", str(proc.pid)], timeout=3)
                except Exception:
                    pass
    try:
        with open(tmp.name, "r") as fh:
            out = fh.read()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    active = _component_power_samples(out)

    if not active:
        return result, EnergyReading(None, None, None, avg_idle,
                                     0, len(idle), interval_s, round(work_seconds, 4),
                                     "unavailable", "no recognized active samples parsed")

    # Sample-coverage validity: the captured samples must actually span the work
    # window. If sampled_seconds (n_samples * interval) covers far less than
    # work_seconds, the avg power is from an unrepresentative slice (e.g. only
    # model-load) and stretching it over the full window would be fabrication.
    sampled_seconds = len(active) * interval_s
    coverage = sampled_seconds / work_seconds if work_seconds > 0 else 0.0
    too_short = work_seconds < interval_s
    if coverage < 0.6 and not too_short:
        return result, EnergyReading(
            None, None, round(sum(active) / len(active), 3),
            (round(avg_idle, 3) if avg_idle is not None else None),
            len(active), len(idle), interval_s, round(work_seconds, 4),
            "unavailable",
            f"sample coverage {coverage:.0%} of work window — powermetrics did not "
            f"sample the full {work_seconds:.1f}s; reading would be unrepresentative.",
        )

    # Energy = average measured power x ACTUAL work_seconds (window-aligned).
    avg_active = sum(active) / len(active)
    joules_total = avg_active * work_seconds
    joules_idle_sub = None
    if avg_idle is not None:
        joules_idle_sub = max(0.0, (avg_active - avg_idle) * work_seconds)

    return result, EnergyReading(
        joules_total=round(joules_total, 3),
        joules_idle_subtracted=(round(joules_idle_sub, 3) if joules_idle_sub is not None else None),
        avg_watts_active=round(avg_active, 3),
        avg_watts_idle=(round(avg_idle, 3) if avg_idle is not None else None),
        n_active_samples=len(active),
        n_idle_samples=len(idle),
        interval_s=interval_s,
        work_seconds=round(work_seconds, 4),
        source="measured",
        note=(
            "energy = avg_measured_power x ACTUAL work_seconds (not n_samples x interval). "
            "total-system power; idle-subtracted is the better model proxy. "
            + ("WARNING: work_seconds < one sampling interval -> reading dominated by "
               "sampling noise; measure a longer batch for a meaningful number."
               if too_short else "")
        ),
    )


def inspect_format(n_samples: int = 2, interval_ms: int = 300) -> str:
    """Return raw powermetrics output so the parser can be verified against the
    real format on THIS machine before trusting any number. Requires primed sudo."""
    if not sudo_available():
        return "<sudo not primed: run `! sudo -v` first>"
    return _run_powermetrics(n_samples, interval_ms)
