"""Kariba REDD+ (VCS902) additionality — REAL project polygon edition.

Replaces the earlier approximate centroid-buffer (whose 0.33 ratio was retracted
as a control-choice artifact) with the project's ACTUAL boundary polygon from the
public Renoster/Carbon Direct dataset (Zenodo 11459391, CC-BY-4.0, provenance
'Official'). Forest loss is now measured INSIDE the true footprint, and the
additionality verdict is gated by a PLACEBO/PERMUTATION null over many candidate
controls — so the conclusion cannot be cherry-picked.

Data (all public, no auth):
- claims: project credited as REDD+ avoided deforestation (Berkeley VROD; 25.7M
  credits retired).
- boundary: VCS902 'Project Area' MultiPolygon, ~10,763 km^2 (africa.gpkg).
- ground truth: Hansen GFC v1.11 lossyear tile 10S_020E (30 m).

Method:
1. Rasterize the real project polygon onto the Hansen grid -> in-project loss rate.
2. Generate MANY candidate control regions (random same-size footprints in the
   same latitude band, NOT overlapping the project) -> a DISTRIBUTION of control
   loss rates.
3. additionality 'effect' = control_loss - project_loss. Placebo null: how often
   does a random control beat another random control by as much? If the project's
   effect is INSIDE the placebo distribution, refuse to certify additionality.

Honest limits (printed): single tile; controls matched only on latitude band +
size (not roads/elevation/protection); loss not attributed to cause; one project.
"""
from __future__ import annotations

import warnings

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")

TIF = "data/hansen_lossyear_10S_020E.tif"
GPKG = "data/africa.gpkg"
VALIDATION_YEAR = 2011
POST = VALIDATION_YEAR - 2000


def _project_polygon():
    import geopandas as gpd
    gdf = gpd.read_file(GPKG)
    idcol = [c for c in gdf.columns if c.lower().replace(" ", "") == "projectid"][0]
    geom = gdf[gdf[idcol] == "VCS902"].geometry.iloc[0]
    return geom


def main():
    geom = _project_polygon()
    minx, miny, maxx, maxy = geom.bounds
    with rasterio.open(TIF) as src:
        # window over the project bounds; READ first, then mask to the array's
        # actual shape + matching transform (avoids float-window rounding mismatch).
        win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        win_transform = src.window_transform(win)
        proj_mask = geometry_mask([geom], out_shape=arr.shape, transform=win_transform,
                                  invert=True)  # True INSIDE polygon
        inside = arr[proj_mask]
        proj_loss = float(np.count_nonzero(inside >= POST) / inside.size) if inside.size >= 1000 else None
        if proj_loss is None:
            print("project mask too small; abort")
            return

        # candidate controls: same-size bounding box shifted within the latitude
        # band, not overlapping the project bbox, staying on-tile (lon 20..30, lat -10..-20)
        w = maxx - minx
        h = maxy - miny
        rng = np.random.default_rng(42)
        controls = []
        tries = 0
        while len(controls) < 30 and tries < 400:
            tries += 1
            lon0 = rng.uniform(20.0, 30.0 - w)
            lat0 = rng.uniform(-20.0 + h, -10.0)   # top edge
            # reject if overlapping the project bbox
            if not (lon0 + w < minx or lon0 > maxx or (lat0 - h) > maxy or lat0 < miny):
                continue
            cwin = from_bounds(lon0, lat0 - h, lon0 + w, lat0, src.transform)
            arr = src.read(1, window=cwin)
            if arr.size < 1000:
                continue
            controls.append(float(np.count_nonzero(arr >= POST) / arr.size))

    controls = np.array(controls)
    effect = float(controls.mean() - proj_loss)          # observed 'avoided' loss
    # placebo null: pairwise control-vs-control differences (no real project)
    rng2 = np.random.default_rng(7)
    placebo = []
    for _ in range(5000):
        a, b = rng2.choice(controls, 2, replace=False)
        placebo.append(a - b)
    placebo = np.array(placebo)
    p95 = float(np.percentile(np.abs(placebo), 95))
    beats_placebo = bool(effect > p95)

    print("=== Kariba REDD+ (VCS902) — REAL polygon additionality ===")
    print(f"project area ~10,763 km^2 (Official boundary, Zenodo 11459391)")
    print(f"in-project post-2011 forest loss: {proj_loss*100:.3f}%")
    print(f"control loss (n={len(controls)}): mean {controls.mean()*100:.3f}%  "
          f"range {controls.min()*100:.3f}-{controls.max()*100:.3f}%")
    print(f"observed effect (control_mean - project): {effect*100:+.3f} pp")
    print(f"placebo null |Δ| p95 (control-vs-control by chance): {p95*100:.3f} pp")
    print(f"\nVERDICT: effect {'BEATS' if beats_placebo else 'does NOT beat'} the "
          f"placebo null -> additionality "
          f"{'is distinguishable from control-choice noise' if beats_placebo else 'is NOT distinguishable from control-choice noise (refuse to certify)'}")
    print("\nLIMITS: single Hansen tile; controls matched on latitude-band + size only "
          "(not roads/elevation/protection); loss cause not attributed; one project; "
          "real boundary but coarse counterfactual -> illustrative, not a final verdict.")


if __name__ == "__main__":
    main()
