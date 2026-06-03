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

# centroid + half-width (degrees) for project and a matched nearby control.
# ~0.4 deg ~ 44 km box. Control offset west, similar latitude band (similar climate/terrain).
PROJECT = {"name": "Kariba (VCS902) approx", "lat": -16.5, "lon": 28.8, "half": 0.4}
CONTROL = {"name": "nearby control (W offset)", "lat": -16.5, "lon": 27.6, "half": 0.4}


def _box(lat, lon, half):
    return (lon - half, lat - half, lon + half, lat + half)  # (minx,miny,maxx,maxy)


def loss_stats(src, region):
    minx, miny, maxx, maxy = _box(region["lat"], region["lon"], region["half"])
    win = from_bounds(minx, miny, maxx, maxy, src.transform)
    arr = src.read(1, window=win)
    total_px = arr.size
    # any loss 2001-2023
    loss_any = int(np.count_nonzero(arr > 0))
    # loss in the post-validation period (year code >= validation_year-2000)
    post_code = PROJECT_VALIDATION_YEAR - 2000
    loss_post = int(np.count_nonzero(arr >= post_code))
    return {
        "region": region["name"],
        "pixels": total_px,
        "loss_any_pct": round(100.0 * loss_any / total_px, 4),
        "loss_post2011_pct": round(100.0 * loss_post / total_px, 4),
    }


def main():
    with rasterio.open(TIF) as src:
        proj = loss_stats(src, PROJECT)
        ctrl = loss_stats(src, CONTROL)

    print("=== Kariba REDD+ (VCS902) COARSE additionality check ===")
    print("data: Hansen GFC v1.11 lossyear 10S_020E (30 m); credited >25.7M tonnes RETIRED")
    for s in (proj, ctrl):
        print(f"  {s['region']:28s} pixels={s['pixels']:>9d} "
              f"loss(all yrs)={s['loss_any_pct']:.3f}%  loss(post-2011)={s['loss_post2011_pct']:.3f}%")

    p, c = proj["loss_post2011_pct"], ctrl["loss_post2011_pct"]
    ratio = (p / c) if c > 0 else float("inf")
    print("\n--- baseline/additionality control ---")
    print(f"post-2011 forest-loss rate: project={p:.3f}%  control={c:.3f}%  ratio={ratio:.2f}")
    if c > 0 and ratio >= 0.8:
        verdict = ("FAILS additionality (coarse): the protected area lost forest at a "
                   f"similar-or-higher rate than the control (ratio {ratio:.2f} >= 0.8) -> "
                   "the credited 'avoided deforestation' is not visible vs the counterfactual.")
    elif c > 0:
        verdict = (f"protected area lost LESS forest (ratio {ratio:.2f} < 0.8) -> consistent "
                   "with some avoided deforestation at this coarse resolution.")
    else:
        verdict = "control had no loss; cannot form a ratio."
    print("VERDICT:", verdict)

    print("\nLIMITATIONS (honest): centroid-buffer != true boundary; single control != "
          "pre-registered synthetic control; fire/drought/leakage not separated; one project; "
          "ILLUSTRATIVE of the method on real data, NOT a verdict on the project.")


if __name__ == "__main__":
    main()
