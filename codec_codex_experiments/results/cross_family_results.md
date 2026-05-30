# Cross-Family Generalization Results (F5 mitigation)

Implementation of `docs/PLAN_F5_cross_family.md`. Closes finding F5 in `critical_review.md`: every gate previously ran on one linear source family only, so a PASS could not show cross-family reach. The ladder + codec contest now run across multiple source families behind a single config switch, and a gate matrix marks which gates are family-robust.

**Run:** `python -m src.run_all --config configs/confirmatory_local.json --out results/cross_family` (8 seeds, n_train=1200, Ridge). Wall clock ~26 s on M4 Pro. `pytest -q` -> 11 passed.

---

## What was built

| Phase | Change | Gate result |
|---|---|---|
| 0 | Froze linear baseline (`tests/fixtures/baseline_linear.json`) + regression test | PASS (bit-for-bit reproducible) |
| 1 | `src/sources.py`: `SourceFamily` protocol + registry; `LinearOscillator` delegates to existing code | PASS (linear byte-identical) |
| 2 | `make_dataset`, `run_ladder`, `run_codec_contest` accept optional `source`; `src/cross_family.py` loops families and builds gate matrix | PASS (single-family identical) |
| 3 | `NonlinearOscillator` (Duffing cubic stiffness, `+beta*x^3`) added to registry | PASS (stable, non-degenerate S7 slope) |
| 4 | Magnetic family reported in same bundle via its own categorical runner (not forced through continuous nRMSE ladder) | n/a by design |
| 5 | Gate matrix in `results.json` + `report.md` "Section 5"; `configs/confirmatory_local.json` sources list | PASS |

The critical guard is `tests/test_sources.py::test_no_target_leakage`, which runs for every registered family and makes the F1 leakage class un-reintroducible as sources are added.

---

## The gate matrix (confirmatory, 8 seeds)

| gate | claim | linear_oscillator | nonlinear_oscillator | classification |
|---|---|---|---|---|
| S0 | sanity | PASS | PASS | family-robust |
| S1 | distinction-dependence | PASS | PASS | family-robust |
| S2 | throughput-invariance | PASS | PASS | family-robust |
| S3 | wrong-lever | PASS | PASS | family-robust |
| S4 | corruption-decay | PASS | PASS | family-robust |
| S5 | additive-access | FAIL | PASS | **family-specific** |
| S6 | shared-placement | PASS | PASS | family-robust |
| S7 | dimensionality-slope | FAIL | FAIL | **fails on all** |
| S8 | multi-codec-audit | PASS | PASS | family-robust |

**Family-robust (PASS on both): S0-S4, S6, S8 (7 of 9).**

---

## Interpretation (honest, not tuned)

1. **Seven of nine gates generalize across a linear and a nonlinear source family.** This is the cross-family evidence F5 required: the core measurement logic (sanity, distinction-dependence, throughput-invariance, wrong-lever, corruption-decay, shared-placement, multi-codec audit) is not an artifact of the single linear calibration source.

2. **S5 (additive-access) is family-specific, and that is the most informative new result.** On the linear family the cross-oscillator coupling channel carries no signal unique from native (it FAILs, as established in the mitigation pass). On the **nonlinear** family it PASSES: the cubic interaction makes cross-oscillator coupling genuinely informative beyond per-oscillator statistics. The single-family setup could never have surfaced this contrast. It directly supports the manuscript's premise that additive access matters more in nonlinear regimes (HE2/HE4 motivation), while showing it is family-dependent rather than universal.

3. **S7 (dimensionality-slope) fails on both families - an honest negative.** Even with cubic nonlinearity, recovery error still *drops* as oscillators are added (nonlinear trained 0.473 -> 0.336 across 4D->8D; linear 0.409 -> 0.298). Diagnosis: the engineered-feature dimensionality grows with `n_osc` (12 -> 26 -> 40 features), so each added oscillator supplies more informative features faster than the nonlinearity adds difficulty. The gate as written conflates "more latent dimensions" with "more observable features." This is reported as a FAIL, not tuned away. A genuine fix (future work) would hold observable-feature count fixed while raising latent complexity, or score per-target recovery normalized by feature count.

Per the plan's rule, no gate threshold was changed to make any family pass. A FAIL is a published negative.

---

## Artifacts

- `results/cross_family/results.json` - full `by_source` blocks + `gate_matrix`.
- `results/cross_family/report.md` - Section 5 renders the matrix.
- `src/sources.py`, `src/cross_family.py` - the seam and orchestrator.
- `tests/test_sources.py`, `tests/test_cross_family.py`, `tests/test_regression_linear.py` - 7 new tests.

## Remaining open items (future work, not blocking F5 closure)

- **S7 redesign**: decouple observable-feature count from latent dimensionality so the dimensionality slope is measurable. (New finding from this work.)
- **Magnetic in the unified matrix**: currently reported via its own categorical runner; a categorical-aware gate matrix could fold it in (routing metric by `target_kind`).
- **Confirmatory MLP/Transformer estimators**: separate P2 gap; F5 used Ridge/KNN only.
