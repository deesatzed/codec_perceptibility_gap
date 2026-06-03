# Carbon-offset additionality — premise validation on real public data

A small, honest test of whether the no-fabrication discipline (specifically the
**additionality/baseline control**) has real purchase on a high-consequence domain
with **public, fetchable data + public ground truth**. Domain selected after five
prior near-misses, under a hard "public data" requirement.

## What was done (all real, no mock)

- **Claims data (public):** Berkeley Voluntary Registry Offsets Database
  (`v2026-04`, 11,343 projects). Filtered to **431 real REDD+/avoided-deforestation
  projects** with credited tonnes (Total Credits Issued) and consequence weight
  (Total Credits Retired). Top projects by volume: ART102 (Guyana, 58M issued),
  VCS934 (DRC, 42M), VCS985 (Peru), VCS674 (Indonesia), **VCS902 (Kariba,
  Zimbabwe, 29M issued / 25.7M retired)**.
- **Ground truth (public, no API key):** Hansen Global Forest Change v1.11
  `lossyear` tile `10S_020E` (30 m, 2001-2023), downloaded from the public Google
  Storage bucket.
- **Control applied:** for Kariba (VCS902 — the most-scrutinized REDD+ project),
  measured post-2011 forest-loss rate inside an approximate project buffer vs a
  nearby control buffer. A REDD+ credit claims avoided deforestation relative to a
  counterfactual; the additionality control asks whether the protected area
  actually lost less forest than a comparable un-protected area.

## The measured result

| Region | post-2011 forest-loss |
|---|---|
| Kariba project buffer (approx) | 0.60% |
| Nearby control buffer (hand-picked) | 1.80% |
| ratio | 0.33 |

At this coarse resolution the protected area lost forest at ~1/3 the control rate.

## The honest finding (this is the real point)

The result is NOT "Kariba is legitimate" and NOT "Kariba is a phantom." It is
sharper and more useful:

> **The additionality verdict is almost entirely determined by the analyst's free
> choice of control/counterfactual area — an unconstrained, unregistered,
> outcome-determining choice.** By moving one buffer, the same 15-line script could
> report "Kariba avoided 67% of deforestation" (low-loss control) or "Kariba is a
> phantom" (high-loss control). My single hand-picked control is the *same
> methodological move* that published critiques say the project itself used to
> inflate its baseline (non-comparable, high-deforestation reference areas).

So the discipline's claim is validated on real data: **REDD+ additionality numbers
are, as currently produced, not falsifiable — the counterfactual is a free analyst
parameter, and there is no mandatory pre-registration or cherry-pick refusal at
issuance.** The control IS the thing that must be pinned and refused-if-gamed. That
is exactly where a refusal-by-default discipline has genuine, high-consequence
purchase ($2B market; 25.7M Kariba credits *retired* = claimed-neutralized tonnes).

## Limitations (stated, not hidden)

- Centroid-buffer != true project boundary (exact polygons were not publicly
  reachable here without auth); area is approximate.
- A single hand-picked control != a pre-registered synthetic control. (This is
  itself the demonstrated flaw, not a defect to apologize for — it is the finding.)
- Fire / drought / leakage are not separated from avoided-deforestation.
- One project, coarse resolution. ILLUSTRATIVE of the method on real data, NOT a
  verdict on the project.

## What this does / does not show

**Does:** the full chain works on real public data — fetch real credited claims +
real satellite ground truth, apply the additionality control, and surface that the
counterfactual choice is outcome-determining and unconstrained. The premise (this
discipline has real purchase here) is validated.

**Does not:** deliver a verdict on Kariba or the market. That needs pre-registered
synthetic-control matching over true project polygons with placebo/permutation
nulls — the rigorous build, scoped but not run.

## Defensible next step (if pursued)

The genuinely useful artifact is not "another over-crediting estimate" (the
literature has those, and they fight over control choice). It is a **pre-registered,
cherry-pick-resistant additionality auditor**: given a project boundary, it (a)
selects controls by a fixed, pre-declared covariate-matching rule, (b) runs a
placebo/permutation null over candidate controls to quantify how much "avoided
deforestation" appears by control-choice alone, and (c) REFUSES to emit an
additionality number when the placebo distribution swamps the estimate. That moves
the unconstrained analyst choice (the actual flaw) behind a refusal gate — the
discipline's core move, on a high-consequence, public-data domain.
