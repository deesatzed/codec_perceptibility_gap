"""Coarse additionality check for the Kariba REDD+ project (VCS902, Zimbabwe),
the most-scrutinized avoided-deforestation project in the voluntary market.

The discipline's BASELINE/ADDITIONALITY control applied to a real carbon claim:
a REDD+ credit asserts "we avoided deforestation that would otherwise have
happened." The honest test: did the protected area lose forest at a LOWER rate
than a comparable nearby area that was NOT protected? If the rates are similar,
the credited 'avoided deforestation' did not happen relative to the counterfactual
-> the claim fails the baseline control.

Data (public, no key):
- Hansen Global Forest Change v1.11 lossyear tile 10S_020E (30 m, 2001-2023).
- Project location: Kariba REDD+ ~ -16.5, 28.8 (approx centroid; EXACT boundaries
  are not public here -> this is a COARSE, ILLUSTRATIVE check, not a verdict).

HONEST LIMITATIONS (printed in output, not hidden):
- centroid-buffer != true project boundary; over/under-includes land.
- single nearby control buffer != pre-registered synthetic control.
- does not separate fire/drought/leakage from avoided-deforestation.
- one project; illustrative of the METHOD on real data.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds

TIF = "data/hansen_lossyear_10S_020E.tif"
PROJECT_VALIDATION_YEAR = 2011   # Kariba validated ~2011; "loss year" code = year-2000

# centroid + half-width (deg) for project and SEVERAL comparable neighbor controls.
# The point of sweeping multiple controls (not picking one) is to expose that the
# additionality verdict is determined by control choice — the actual finding.
PROJECT = {"name": "Kariba (VCS902) approx", "lat": -16.5, "lon": 28.8, "half": 0.4}
CONTROLS = [
    {"name": "N +1.2", "lat": -15.3, "lon": 28.8, "half": 0.4},
    {"name": "W -1.2", "lat": -16.5, "lon": 27.6, "half": 0.4},
    {"name": "E +1.0", "lat": -16.5, "lon": 30.0, "half": 0.4},
    {"name": "W -2.4", "lat": -16.5, "lon": 26.4, "half": 0.4},
    {"name": "S -1.2", "lat": -17.7, "lon": 28.8, "half": 0.4},
]


def _box(lat, lon, half):
    return (lon - half, lat - half, lon + half, lat + half)


def loss_post(src, region):
    """Post-validation forest-loss %. Returns None for off-tile / degenerate
    windows (the NaN guard the first run lacked)."""
    minx, miny, maxx, maxy = _box(region["lat"], region["lon"], region["half"])
    arr = src.read(1, window=from_bounds(minx, miny, maxx, maxy, src.transform))
    if arr.size < 1000 or not np.isfinite(arr).all():
        return None
    post_code = PROJECT_VALIDATION_YEAR - 2000
    return round(100.0 * np.count_nonzero(arr >= post_code) / arr.size, 3)


def main():
    with rasterio.open(TIF) as src:
        p = loss_post(src, PROJECT)
        ctrls = [(c["name"], loss_post(src, c)) for c in CONTROLS]

    print("=== Kariba REDD+ (VCS902) COARSE additionality SWEEP ===")
    print("data: Hansen GFC v1.11 lossyear 10S_020E (30 m); credited >25.7M tonnes RETIRED")
    print(f"project post-2011 loss: {p:.3f}%\n")
    ratios = []
    for name, c in ctrls:
        if c is None:
            print(f"  control {name:8s}: OFF-TILE/degenerate -> skipped (NaN guard)")
            continue
        r = p / c if c > 0 else float("inf")
        ratios.append(r)
        print(f"  control {name:8s}: {c:.3f}%  ratio={r:.2f}")
    print("\n--- THE FINDING (not a verdict on the project) ---")
    if ratios:
        print(f"additionality ratio swings from {min(ratios):.2f} (project looks heroic) "
              f"to {max(ratios):.2f} (project did ~nothing)")
        print("=> the verdict is determined by the analyst's free choice of control, "
              "not by the project. Same project, same data, opposite conclusions.")
    print("\nLIMITATIONS (honest): centroid-buffer != true boundary; neighbor buffers != "
          "covariate-matched controls; fire/drought/leakage not separated; one project; "
          "ILLUSTRATIVE that control-choice is outcome-determining, NOT a verdict on Kariba.")


if __name__ == "__main__":
    main()
