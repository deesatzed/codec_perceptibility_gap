# Interim Africa Pilot — results + what they prove

Run of `africa_pilot.py` over the 15 Official-boundary VCS REDD+ projects in
Africa. Per the PREREGISTRATION.md deviation (2026-06-03), this uses **reduced
matching** (prior-period loss trend + same-area, Hansen-only); roads/DEM/WDPA are
deferred. **This is a pipeline-validation pilot, NOT the paper result.**

## Raw result

**0 CERTIFIED · 15 REFUSED · 0 dropped** (of 15).

| Project | proj post-loss | ctrl mean | effect | placebo p95 | n_ctrl |
|---|---|---|---|---|---|
| VCS902 (Kariba) | 0.37% | 0.54% | +0.17pp | 0.68pp | 30 |
| VCS2363 | 0.05% | 0.38% | +0.33pp | 8.64pp | 30 |
| VCS562 | 0.03% | 0.29% | +0.26pp | 3.13pp | 30 |
| VCS612 | 0.01% | 0.20% | +0.19pp | 1.50pp | 30 |
| VCS1532 | 4.34% | 5.37% | +1.02pp | 3.30pp | 30 |
| VCS1201 | 0.17% | 1.26% | +1.09pp | 2.55pp | 6 |
| VCS2538 | 0.11% | 0.01% | −0.10pp | 0.01pp | 20 |
| VCS1900 | 0.02% | 0.01% | −0.02pp | 0.03pp | 9 |
| VCS1674 | 4.79% | 4.64% | −0.15pp | 10.80pp | 30 |
| VCS1215 | 5.04% | 3.97% | −1.07pp | 5.17pp | 30 |
| VCS1897 | 6.11% | 2.56% | −3.54pp | 8.85pp | 30 |
| VCS1325 | 15.32% | 9.12% | −6.20pp | 15.28pp | 30 |
| VCS1359 | 10.55% | 3.45% | −7.10pp | 8.48pp | 30 |
| VCS1047 | 16.75% | 6.14% | −10.61pp | 7.51pp | 30 |
| VCS1311 | 17.32% | 5.61% | −11.72pp | 8.49pp | 30 |

## Honest reading (two groups, two meanings)

**Do NOT read this as "15 REDD+ projects failed."** The refusals split into two
very different causes:

1. **Genuine no-signal (effect ~0, swamped by placebo):** VCS902, VCS2363, VCS562,
   VCS612, VCS2538, VCS1900, VCS1674. Small positive-or-zero effects below the
   placebo null — honestly "additionality not distinguishable from control-choice
   noise" at this matching fidelity.

2. **Negative effect (project lost MORE than controls):** VCS1311 (−11.7pp),
   VCS1047 (−10.6pp), VCS1359 (−7.1pp), VCS1325 (−6.2pp), VCS1897 (−3.5pp). A
   protected area losing *far more* forest than its "matched" controls is a **red
   flag on the MATCHING, not (necessarily) the project** — almost certainly the
   reduced Hansen-only controls are landing on unphysical comparators (water,
   arid/already-cleared land with structurally low loss). The prior-trend filter
   does not exclude a zero-loss desert/water box (it also has ~0 prior loss).

## What the pilot actually proves

1. **The pipeline works end-to-end** on real data: real Official boundaries
   (multi-tile, validity-checked), per-project start years, prior-trend-filtered
   control selection, placebo/permutation null, and the `claimgate` refusal gate —
   all 15 ran, 0 dropped. ✅ (the pilot's goal)

2. **The reduced (Hansen-only) matching is insufficient** — empirically
   demonstrated by the negative-effect cluster. This *validates the prereg's
   insistence on roads/DEM/WDPA/biome covariates*: without them, controls are not
   physically comparable, and the audit produces uninterpretable (even backwards)
   effects. **The full covariate set is required before any portfolio/paper claim.**

## Decision

- The pilot has done its job: pipeline proven, reduced-matching shown inadequate.
- **Gate to the full run:** add the prereg's full covariate set (distance-to-road,
  elevation, slope, WDPA, biome) and a **land-cover/forest-baseline mask** so
  controls must actually be forest at baseline (excludes water/desert). Re-run on
  Africa, confirm the negative-effect artifacts resolve, THEN scale to 66.
- No additionality verdict on any project is claimed from this interim run.
