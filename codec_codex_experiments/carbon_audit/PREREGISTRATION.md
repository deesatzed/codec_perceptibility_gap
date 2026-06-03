# Pre-registration — placebo-null additionality audit of REDD+ projects

**Frozen before results are computed.** The entire contribution is cherry-pick
resistance, so the analysis rule is declared here, in advance, and not changed
after seeing outcomes. Any post-hoc deviation will be logged explicitly in the
results as a deviation, with rationale.

Date frozen: 2026-06-03. Author: Wayne A. Satz.

## 1. Question (confirmatory)

For voluntary-market REDD+ ("avoided deforestation", VCS type `AD`) projects with
publicly available official boundaries, does each project's observed in-boundary
forest-loss reduction, relative to **covariate-matched** control areas, exceed
what a **placebo/permutation null** over matched controls produces by chance? I.e.
is the credited additionality *distinguishable from control-selection noise*?

This is a **measurement-methodology + portfolio audit**, NOT a fraud finding and
NOT a re-derivation of any single project's true carbon accounting.

## 2. Sample (fixed)

- All VCS REDD+ (`Project Type == AD`) projects in the public Renoster/Carbon
  Direct boundary dataset (Zenodo 11459391, CC-BY-4.0) with
  `Processing Approach == "Official"` (registry-sourced boundary, highest trust).
- Count at freeze: **66 projects** (39 South America, 15 Africa, 6 Asia, 5 North
  America, 1 Oceania; Brazil 19, Peru 9, Colombia 7 lead).
- **Exclusions (declared now):** a project is dropped (and reported as dropped,
  with reason) if: (a) its boundary polygon is invalid/empty; (b) the required
  Hansen tile(s) or covariate layers are unavailable for its location; (c) the
  project validation/start year cannot be determined; (d) <1000 valid 30 m pixels
  inside the boundary. No project is dropped for its result.

## 3. Outcome (fixed)

Per project, per 30 m Hansen pixel: forest loss in the **post-validation period**
(loss-year code >= validation_year − 2000). Outcome = fraction of in-boundary
pixels lost post-validation. Same definition for controls.

## 4. Covariates for matching (fixed set, all public)

Controls must resemble the project on the variables that actually predict
deforestation:
- **distance to nearest road** (OSM / GRIP global roads)
- **elevation** and **slope** (SRTM 30 m or Copernicus DEM)
- **prior-period forest-loss rate** (Hansen loss BEFORE validation year — the
  pre-treatment trend)
- **protected-area status / proximity** (WDPA)
- **latitude band + biome proxy** (coarse climate control)

## 5. Control selection rule (fixed)

For each project: generate candidate control regions of the **same area** as the
project boundary, within the same country (or same Hansen tile if cross-border),
**not overlapping** the project. Retain the **N=30** candidates whose covariate
vector is closest (standardized Euclidean / nearest-neighbor on the covariates in
§4) to the project's mean covariate vector. Controls must have a **pre-period loss
trend within ±50%** of the project's pre-period trend (parallel-trends proxy);
candidates failing this are rejected before matching.

## 6. Test statistic + null (fixed)

- **Observed effect** = mean(control post-loss) − project post-loss (positive =
  apparent avoided deforestation).
- **Placebo null** = distribution of (control_i − control_j) over matched control
  pairs (5000 draws). This is "how different are two equally-valid controls by
  chance."
- **Refusal gate (via `claimgate`)**: a project's additionality is **certified
  only if** the observed effect exceeds the placebo null's 95th percentile of
  |Δ| AND the project's matched controls passed the parallel-trends filter. Else:
  **REFUSED** (effect indistinguishable from control-choice noise).

No threshold will be tuned after seeing results. 95th percentile is fixed now.

## 7. Portfolio reporting (fixed)

- Headline = **count and fraction of the 66 that PASS vs REFUSE**, plus the count
  dropped (with reasons).
- Dollar/credit weighting: also report the result **weighted by retired credits**
  (Berkeley VROD `Total Credits Retired`), since retired credits are the ones used
  for net-zero claims.
- Full per-project table (effect, placebo p95, verdict) published.

## 8. Pre-committed honest outcomes

- If **most projects PASS** → additionality is largely robust to control choice on
  this sample; report it plainly (a positive for the market).
- If **most REFUSE** → additionality is largely indistinguishable from
  control-selection noise; report it as a measurement finding, explicitly NOT as
  proof of fraud, and positioned as complementary to West et al. (2023, Science)
  and its rebuttals.
- If **coverage is poor** (many drops) → report the audit as inconclusive at
  portfolio scale and publish only the method + what was reachable.

## 9. Known limitations (declared up front)

- Hansen loss ≠ deforestation cause (fire, drought, agriculture, legal logging
  not separated); leakage/displacement not measured (acknowledged, not corrected).
- Boundary precision varies even among "Official" records; pre-period-trend filter
  partially mitigates mismatch.
- Single 30 m product (Hansen); no cross-sensor validation.
- This audits **observed additionality signal vs a counterfactual**, not the
  project's full carbon-accounting methodology, permanence, or co-benefits.

## 10. Deviations log

(Any change from the above, made after freeze, is recorded here with date +
reason. Empty at freeze.)
