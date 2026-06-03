"""Worked example: a carbon-offset additionality claim that the library REFUSES.

Uses the real Kariba (VCS902) control sweep measured from public Hansen satellite
data (see carbon_audit/RESULTS.md): the project's post-2011 forest-loss rate is
0.60%, but comparable neighbor controls range 0.62%-4.12%, so the 'avoided
deforestation' ratio swings 0.15->0.97 by control choice alone.

The library encodes the honest verdict: when the additionality estimate is
explained by the choice of control (the project's loss is INSIDE the spread of
control losses), refuse to certify a number. Run:
    python -m src.refuse.examples.example_carbon
"""
from __future__ import annotations

import numpy as np

from src.refuse.contract import CheckResult, Battery

# Real measured values (carbon_audit/RESULTS.md), post-2011 forest-loss %:
PROJECT_LOSS = 0.602
CONTROL_LOSSES = [4.121, 1.803, 0.713, 0.628, 0.622]   # N, W1, E1, W2, S1


def control_choice_not_outcome_determining(project: float, controls: list,
                                            additionality_bar: float = 0.8,
                                            name: str = "control_robustness"):
    """Refuse if the additionality VERDICT flips with control choice.

    Verdict = 'showed additionality' iff ratio = project_loss/control_loss <
    additionality_bar (project lost meaningfully less than the counterfactual).
    If, across candidate controls, that pass/fail decision goes BOTH ways, the
    verdict is an artifact of which control was picked -> refuse to certify."""
    def _check() -> CheckResult:
        ratios = [project / c for c in controls if c > 0]
        rmin, rmax = min(ratios), max(ratios)
        passes = [r < additionality_bar for r in ratios]
        ev = {"project_loss": project, "ratio_range": [round(rmin, 2), round(rmax, 2)],
              "additionality_bar": additionality_bar,
              "verdict_split": f"{sum(passes)}/{len(passes)} controls would PASS"}
        if any(passes) and not all(passes):
            return CheckResult(name, False,
                               f"additionality verdict is control-dependent: ratio swings "
                               f"{rmin:.2f}->{rmax:.2f}, crossing the {additionality_bar} bar "
                               f"({ev['verdict_split']}). Same project, opposite verdicts by "
                               "control choice -> cannot certify.", ev)
        return CheckResult(name, True,
                           f"verdict robust (ratios {rmin:.2f}-{rmax:.2f} all one side of "
                           f"{additionality_bar})", ev)
    return _check


def main():
    battery = Battery([
        control_choice_not_outcome_determining(PROJECT_LOSS, CONTROL_LOSSES),
    ])
    r = battery.evaluate(value="Kariba VCS902 avoided-deforestation claim")
    print(f"[carbon additionality] {r.status}")
    if r.status == "REFUSED":
        print(f"  -> refused: {r.reason}")
        print(f"     evidence: {r.evidence[0].evidence}")
    else:
        print("  -> verified")
    print("\n(Real public data: Berkeley VROD claims + Hansen satellite loss; "
          "see carbon_audit/RESULTS.md. Illustrative, not a verdict on the project.)")


if __name__ == "__main__":
    main()
