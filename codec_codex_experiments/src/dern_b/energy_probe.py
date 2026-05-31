"""Opt-in, sudo-gated true-energy probe via macOS `powermetrics`.

powermetrics requires sudo (a password) and cannot run unattended. This module
NEVER fabricates a joules number. If non-interactive sudo is unavailable it
returns source_tag='unavailable' and joules=None. The user can enable measured
energy by priming sudo in the session first (e.g. `! sudo -v`) and then running
the driver.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Optional, Tuple


def sudo_available() -> bool:
    """True iff sudo can run non-interactively right now (no password prompt)."""
    try:
        r = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False


def measure_energy(fn: Callable[[], Any]) -> Tuple[Any, Optional[float], str]:
    """Run fn(); if sudo+powermetrics available, sample CPU+GPU power across the
    call and integrate to joules. Otherwise return (result, None, 'unavailable').

    Returns (result, joules_or_None, source_tag in {'measured','unavailable'}).
    """
    if not sudo_available():
        return fn(), None, "unavailable"

    # Sample power in the background while fn runs, then integrate avg_power*duration.
    # powermetrics prints periodic samples; we average CPU+GPU power over the window.
    t0 = time.time()
    proc = subprocess.Popen(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power", "-i", "200"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        result = fn()
    finally:
        duration = time.time() - t0
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=3)
        except Exception:
            out = ""
            proc.kill()

    # Parse "... Power: NNN mW" lines for CPU+GPU; average and integrate.
    powers_mw = []
    for line in out.splitlines():
        ls = line.strip()
        if ls.endswith("mW") and "Power" in ls:
            try:
                powers_mw.append(float(ls.split(":")[1].strip().split()[0]))
            except Exception:
                pass
    if not powers_mw:
        return result, None, "unavailable"
    avg_w = (sum(powers_mw) / len(powers_mw)) / 1000.0
    joules = avg_w * duration
    return result, float(joules), "measured"
