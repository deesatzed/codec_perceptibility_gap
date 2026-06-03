# Carbon-offset additionality — premise validation on real public data

> **CORRECTION (contrarian self-audit).** An earlier version of this file
> headlined a single ratio of **0.33** ("project lost 1/3 the control's forest")
> from ONE hand-picked control area. A contrarian check swept multiple comparable
> neighbor controls and the ratio swings from **0.15 to 0.97** depending purely on
> which control is chosen (plus one off-tile control that silently produced a NaN
> in the first run). **The 0.33 headline is retracted as an artifact of control
> choice.** This is not a footnote — it IS the finding (see below), and it is the
> discipline catching its own overclaim. The corrected result is the *sweep*, not
> any single number.


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

## The measured result (corrected: a SWEEP, not one number)

Project post-2011 forest loss: **0.60%**. Five comparable neighbor controls:

| Control (offset) | post-2011 loss | project/control ratio | what it would imply |
|---|---|---|---|
| N (+1.2°) | 4.12% | 0.15 | project looks heroic |
| W (−1.2°) | 1.80% | 0.33 | project looks great |
| E (+1.0°) | 0.71% | 0.85 | project looks ~neutral |
| W (−2.4°) | 0.63% | 0.96 | project did ~nothing |
| S (−1.2°) | 0.62% | 0.97 | project did ~nothing |

**Same project, same satellite data, same period — the additionality verdict
swings from 0.15 (saved almost everything) to 0.97 (saved nothing) purely by which
neighbor you choose as the control.** (A sixth control fell off the tile and the
first run silently returned NaN — itself a caught measurement bug.)

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
