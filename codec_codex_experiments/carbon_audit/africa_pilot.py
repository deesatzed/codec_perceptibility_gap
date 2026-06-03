"""INTERIM AFRICA PILOT — placebo-null additionality audit, 15 Official-boundary
VCS REDD+ projects (see PREREGISTRATION.md deviation 2026-06-03).

Reduced matching: prior-period loss trend (parallel-trends filter) + same-area +
on-tile, all from Hansen. Roads/DEM/WDPA deferred to the full 66-run. This proves
the pipeline (real polygons -> matched controls -> placebo null -> claimgate
refusal gate) end-to-end before the heavy multi-continent fetch.

Per project:
  - clip Hansen lossyear to the REAL boundary polygon (handles multi-tile).
  - project_pre  = loss fraction BEFORE start year; project_post = loss after.
  - candidate controls: same-area boxes on the covering tile(s), not overlapping,
    retained only if their pre-period loss is within +/-50% of the project's
    (parallel-trends filter). Keep up to N=30 nearest by pre-loss.
  - effect = mean(control_post) - project_post; placebo null = control_i - control_j.
  - claimgate gate: certify additionality only if effect beats placebo p95.
"""
from __future__ import annotations

import csv
import math
import warnings
from datetime import datetime

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "../src")          # use the local claimgate
from claimgate import Battery, CheckResult   # noqa: E402

INDEX = "data/zenodo_index.csv"
GPKG = "data/africa.gpkg"
TILE_FMT = "data/hansen_lossyear_{}.tif"
N_CTRL = 30
TREND_TOL = 0.50          # controls' pre-loss within +/-50% of project's


def _tile_id(lat, lon):
    tlat = int(math.ceil(lat / 10) * 10)
    tlon = int(math.floor(lon / 10) * 10)
    return f"{abs(tlat)}{'N' if tlat>=0 else 'S'}".zfill(3) + "_" + f"{abs(tlon):03d}{'E' if tlon>=0 else 'W'}"


def _start_year(rows, pid):
    for r in rows:
        if r["ProjectID"] == pid and r.get("Project Start Date"):
            try:
                return datetime.strptime(r["Project Start Date"], "%m/%d/%Y").year
            except Exception:
                return None
    return None


def _loss_split(arr, post_code):
    """(pre_frac, post_frac) for a 1D/2D array of lossyear codes (0=no loss)."""
    n = arr.size
    if n < 1000:
        return None
    pre = np.count_nonzero((arr > 0) & (arr < post_code)) / n
    post = np.count_nonzero(arr >= post_code) / n
    return float(pre), float(post)


def _project_stats(src, geom, post_code):
    minx, miny, maxx, maxy = geom.bounds
    win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
    arr = src.read(1, window=win)
    if arr.size < 1000:
        return None
    mask = geometry_mask([geom], out_shape=arr.shape, transform=src.window_transform(win), invert=True)
    inside = arr[mask]
    return _loss_split(inside, post_code)


def _controls(src, geom, post_code, proj_pre, seed=42):
    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    b = src.bounds
    rng = np.random.default_rng(seed)
    cands = []
    tries = 0
    while len(cands) < N_CTRL and tries < 600:
        tries += 1
        lon0 = rng.uniform(b.left, b.right - w)
        lat0 = rng.uniform(b.bottom + h, b.top)            # top edge
        if not (lon0 + w < minx or lon0 > maxx or (lat0 - h) > maxy or lat0 < miny):
            continue                                        # overlaps project
        cwin = from_bounds(lon0, lat0 - h, lon0 + w, lat0, src.transform)
        a = src.read(1, window=cwin)
        sp = _loss_split(a, post_code)
        if sp is None:
            continue
        cpre, cpost = sp
        # parallel-trends filter: pre-loss within +/-TOL of project's pre-loss
        if proj_pre > 0 and abs(cpre - proj_pre) > TREND_TOL * proj_pre:
            continue
        if proj_pre == 0 and cpre > 0.005:
            continue
        cands.append((abs(cpre - proj_pre), cpost))
    cands.sort()
    return [post for _, post in cands]


def audit_project(pid, geom, start_year, rows):
    if start_year is None:
        return {"pid": pid, "status": "DROP", "reason": "no start year"}
    post_code = start_year - 2000
    if not (1 <= post_code <= 23):
        return {"pid": pid, "status": "DROP", "reason": f"start {start_year} out of Hansen range"}
    # primary tile = polygon centroid
    c = geom.centroid
    tile = _tile_id(c.y, c.x)
    path = TILE_FMT.format(tile)
    try:
        src = rasterio.open(path)
    except Exception:
        return {"pid": pid, "status": "DROP", "reason": f"tile {tile} unavailable"}
    with src:
        ps = _project_stats(src, geom, post_code)
        if ps is None:
            return {"pid": pid, "status": "DROP", "reason": "<1000 px in boundary"}
        proj_pre, proj_post = ps
        ctrls = _controls(src, geom, post_code, proj_pre)
    if len(ctrls) < 5:
        return {"pid": pid, "status": "DROP", "reason": f"only {len(ctrls)} trend-matched controls"}
    ctrls = np.array(ctrls)
    effect = float(ctrls.mean() - proj_post)
    rng = np.random.default_rng(7)
    placebo = np.array([rng.choice(ctrls) - rng.choice(ctrls) for _ in range(5000)])
    p95 = float(np.percentile(np.abs(placebo), 95))

    def _gate():
        return CheckResult("beats_placebo", bool(effect > p95),
                           f"effect {effect*100:+.3f}pp vs placebo p95 {p95*100:.3f}pp",
                           {"effect_pp": round(effect*100, 3), "placebo_p95_pp": round(p95*100, 3)})
    res = Battery([_gate]).evaluate(value=pid)
    return {"pid": pid, "status": "CERTIFIED" if res.is_verified() else "REFUSED",
            "start": start_year, "n_ctrl": len(ctrls),
            "proj_post_pct": round(proj_post*100, 3), "ctrl_mean_pct": round(float(ctrls.mean())*100, 3),
            "effect_pp": round(effect*100, 3), "placebo_p95_pp": round(p95*100, 3)}


def main():
    import geopandas as gpd
    rows = list(csv.DictReader(open(INDEX)))
    afr = {r["ProjectID"] for r in rows if r.get("ProjectID","").startswith("VCS")
           and r.get("Project Type")=="AD" and r.get("Processing Approach")=="Official"
           and r.get("Continent")=="Africa"}
    gdf = gpd.read_file(GPKG)
    idc = [c for c in gdf.columns if c.lower().replace(" ","")=="projectid"][0]
    sub = gdf[gdf[idc].isin(afr)]
    results = []
    for _, r in sub.iterrows():
        pid = r[idc]
        results.append(audit_project(pid, r.geometry, _start_year(rows, pid), rows))

    certified = [x for x in results if x["status"]=="CERTIFIED"]
    refused = [x for x in results if x["status"]=="REFUSED"]
    dropped = [x for x in results if x["status"]=="DROP"]
    print("=== INTERIM AFRICA PILOT — additionality audit (15 Official REDD+) ===")
    print(f"CERTIFIED: {len(certified)}  REFUSED: {len(refused)}  DROPPED: {len(dropped)}  of {len(results)}\n")
    for x in sorted(results, key=lambda z: z["status"]):
        if x["status"]=="DROP":
            print(f"  {x['pid']:8s} DROP   ({x['reason']})")
        else:
            print(f"  {x['pid']:8s} {x['status']:9s} proj {x['proj_post_pct']:.2f}% vs ctrl {x['ctrl_mean_pct']:.2f}% "
                  f"effect {x['effect_pp']:+.2f}pp placebo_p95 {x['placebo_p95_pp']:.2f}pp (n_ctrl {x['n_ctrl']})")
    print("\nINTERIM: prior-trend+area matching only (roads/DEM/WDPA deferred); single 30m product; "
          "loss cause not attributed. NOT the paper result; proves the pipeline. See PREREGISTRATION.md.")


if __name__ == "__main__":
    main()
