# Mitigation Plan — F5 Cross-Family Generalization Structural Gap

**Status:** Proposed (not yet approved/implemented). Targets finding **F5** in `results/critical_review.md`.
**Owner:** TBD. **Scope:** medium. **Risk class:** experiment-spec change → requires user approval before merge (per workspace build-checklist rule).

---

## 1. The gap, precisely

Every proof-ladder stage and both codec experiments draw from **one** generator at a time, and that generator is hardcoded:

- `src/proof_ladder.py:212` — `make_dataset()` calls `sample_theta()` + `simulate()`, which are the **linear coupled-oscillator family only**.
- `src/proof_ladder.py:241` — `run_ladder()` builds `base = {seed: make_dataset(2, cfg, seed)}` with `n_osc=2` fixed; every stage S0–S8 reads from this single linear `base`.
- `src/codec_contest.py:65,150` — `multi_codec_audit_one` and `codec_robustness_one` both call `make_dataset(2, cfg, seed)` directly (linear only).
- `src/magnetic_pendulum.py` — has its own separate simulator; it is the *only* nonlinear source and is never run through the proof-ladder or codec stages.

**Consequence (as stated in F5):** a PASS demonstrates the measurement logic behaves sensibly on one linear generator. It is **not** evidence that any gate generalizes across source families — yet the manuscript framing ("source-family hierarchy", "nonlinear stress test for HE2/HE4") implies cross-family reach. The S7 FAIL is itself a symptom: the linear family cannot produce the difficulty slope S7 needs, so the gate is structurally unsatisfiable there.

**What "done" looks like:** the full ladder + codec stages run on ≥2 source families behind a single switch; `results.json` carries per-source results; each gate reports PASS/FAIL per family; and a cross-family summary makes explicit which gates are family-robust vs family-specific. The manuscript's cross-family claim then becomes supportable (or is falsified and reported as such).

---

## 2. Design principle: one seam, source-agnostic everywhere else

The codebase is already 80% ready for this. The feature channels (`native_feats`, `coupling_feats`, `engineered_feats`, `direct_feats`, `quantize`) all operate on a generic trajectory array `x: (B, T, N)` and a target array `theta: (B, D)` — they do **not** know which simulator produced them. The estimator (`fit_predict`), metrics (`nrmse`, `per_target_nrmse`), and controls (`phase_randomized_surrogate`) are likewise generic.

**Therefore the only thing that must change is the data-production seam.** Introduce a `SourceFamily` protocol with a single registry, route `make_dataset` through it, and loop the ladder over a configured list of families. No stage logic changes. This keeps the diff small and the risk contained.

> **Non-negotiable carried-forward constraint (from F1):** no source family may expose the scored target as a feature. Each family must declare its `targets` and its `native`/`expanded_physics` channels separately, and the F1 anti-leakage rule (`expanded_physics` excludes the target; any leaked channel is a labeled `*_leak_positive_control`) is enforced per family.

---

## 3. Target source families (start with 3; the switch supports N)

| key | family | nature | why it's in the set | targets (theta) |
|---|---|---|---|---|
| `linear_oscillator` | existing forced/damped coupled oscillators | linear, fully recoverable | calibration/ablation baseline; must keep reproducing current numbers exactly | springs…, coupling g, damping c |
| `nonlinear_oscillator` | Duffing-type / cubic-stiffness coupled oscillators (add `beta * x^3` term to `simulate`) | mildly nonlinear, same observable shape | the **minimal** change that should let S7 produce a real difficulty slope without leaving the oscillator observable family | springs…, coupling g, damping c, nonlinearity beta |
| `magnetic` | existing magnetic-pendulum simulator | chaotic / multistable | the genuine nonlinear stress test; already implemented, just not wired into the ladder | basin label, finite-time divergence (categorical + continuous) |

Rationale for `nonlinear_oscillator` as the second family: it shares the exact `(B, T, N)` trajectory observable shape with the linear family, so it slots into the existing continuous-nRMSE ladder with **zero** stage changes — it is the cheapest way to get a true second family and directly tests the S7 hypothesis ("a nonlinear source will generate the dimensionality difficulty slope the linear one cannot"). `magnetic` is the harder integration (categorical targets) and is staged second.

---

## 4. Phased build checklist

Each phase is independently validated. **Do not proceed to the next phase until the current phase's gate passes.** No phase may relax an existing gate to make a new family pass.

### Phase 0 — Pin the baseline (regression guard) ✅ gate before any refactor
- [ ] Record current canonical numbers from `results/confirmatory_mitigated/results.json` as a frozen fixture (`tests/fixtures/baseline_linear.json`): S0–S8 metrics, CodecGuard corr/LOCO/null, CodecBench AUC gap, magnetic accuracies.
- [ ] Add a regression test `tests/test_regression_linear.py` asserting the refactored `linear_oscillator` path reproduces these within a tight tolerance (e.g. abs ≤ 1e-6 for same seed/estimator).
- **Gate:** new test passes against current code (it must, before refactoring).

### Phase 1 — Introduce the `SourceFamily` seam (no behavior change)
- [ ] Create `src/sources.py` with a `SourceFamily` protocol:
  ```python
  class SourceFamily(Protocol):
      key: str
      target_labels: list[str]          # for per-target reporting (F8)
      target_kind: str                  # "continuous" | "categorical" | "mixed"
      def sample(self, n: int, cfg, seed) -> tuple[np.ndarray, np.ndarray]:
          """Return (trajectory (B,T,N), theta (B,D)). NEVER puts theta in the trajectory (F1)."""
      def native_channel(self, x) -> np.ndarray
      def expanded_physics_channel(self, x) -> np.ndarray   # excludes target (F1)
  ```
- [ ] Move the existing linear `simulate`/`sample_theta`/`native_feats`/`coupling_feats` into a `LinearOscillator` implementation; keep the old module-level functions as thin shims so nothing else breaks.
- [ ] Add a `SOURCE_REGISTRY = {"linear_oscillator": LinearOscillator(), ...}` and a `get_source(cfg)` resolver reading `cfg["source"]` (default `"linear_oscillator"`).
- [ ] Route `make_dataset()` through `get_source(cfg).sample(...)`.
- **Gate:** Phase-0 regression test still passes (identical numbers); `pytest -q` → all green.

### Phase 2 — Loop the ladder + codec stages over families
- [ ] Change `cfg["source"]` to accept either a string or a list; add `cfg["sources"]` (default `["linear_oscillator"]`).
- [ ] In `run_ladder`, `run_codec_contest`: wrap the existing body in a `for family in resolve_sources(cfg)` loop; key results by family in `results.json` (`{"by_source": {"linear_oscillator": {...S0..S8}, ...}}`).
- [ ] Keep a top-level `results["proof_ladder"]` pointing at the first/primary family for backward compatibility with `report.py` and the smoke tests.
- **Gate:** single-family runs produce byte-identical metrics to Phase 1; `pytest -q` green.

### Phase 3 — Add the `nonlinear_oscillator` family
- [ ] Implement `NonlinearOscillator`: copy `simulate` and add a cubic stiffness term (`a += -beta * x**3`), extend `sample_theta` to draw `beta` in a small range kept stable on the M4 (clip accelerations as the magnetic sim does).
- [ ] Add `beta` to `target_labels`; native/expanded channels reuse the oscillator feature functions (already generic).
- [ ] Run the ladder on `["linear_oscillator", "nonlinear_oscillator"]` at demo scale.
- **Gate:** runs without numerical blowup; **S7 produces a non-degenerate slope on the nonlinear family** (the core F5 hypothesis — record PASS or honest FAIL, do not tune the gate to force PASS).

### Phase 4 — Wire `magnetic` into the cross-family report (categorical targets)
- [ ] Wrap the existing magnetic simulator in a `MagneticPendulum(SourceFamily)` with `target_kind="categorical"` (basin) + a continuous divergence target.
- [ ] For categorical families, the ladder uses the magnetic module's KNN classify/log-loss path instead of Ridge/nRMSE; gate the metric choice on `family.target_kind` (no new math, just routing).
- [ ] Ensure the F1 separation holds: `expanded_physics` vs `oracle_leak_positive_control` already exist in `magnetic_pendulum.py` — reuse them.
- **Gate:** magnetic runs through the unified path and reproduces `results/confirmatory_mitigated` magnetic numbers within tolerance.

### Phase 5 — Cross-family synthesis + report + manuscript
- [ ] Add a cross-family summary to `results.json`: for each gate, `{family: PASS/FAIL}` plus a `family_robust: bool` flag (PASS on ≥2 families) and a `family_specific` list.
- [ ] Extend `src/report.py` with a "Cross-family gate matrix" table (gates × families) and a one-line interpretation per gate.
- [ ] Update `configs/confirmatory_local.json` with `"sources": ["linear_oscillator","nonlinear_oscillator","magnetic"]`; keep `configs/demo.json` to 2 families for speed. Keep all defaults M4-friendly (scaling knobs in config only).
- [ ] Update the manuscript (`../perceptibility_gap_paper_v1_3.md`, → v1.4): replace the implied cross-family reach with the measured gate matrix; close Appendix F.5; record whichever gates proved family-robust vs family-specific.
- **Gate:** `pytest -q` green; `results/cross_family/results.json` + `report.md` exist with the gate matrix; manuscript numbers trace to that file.

---

## 5. Test plan (each is a hard gate, ≥1 per phase)

| Test | Asserts | Phase |
|---|---|---|
| `test_regression_linear` | refactor reproduces frozen baseline within 1e-6 | 0/1/2 |
| `test_source_registry` | every registered family returns shapes `(B,T,N)`,`(B,D)` and `target_labels` length matches D | 1 |
| `test_no_target_leakage` | for each family, no column of `native`/`expanded_physics` equals (or correlates > 0.999 with) any target column | 1 (enforces F1) |
| `test_ladder_multi_source` | `by_source` has one entry per configured family; each has S0,S8 | 2 |
| `test_nonlinear_s7_nondegenerate` | nonlinear S7 slope spans a real range (trained error not monotone-trivial) | 3 |
| `test_magnetic_unified_matches` | magnetic via unified path == standalone magnetic within tolerance | 4 |
| `test_cross_family_matrix` | gate matrix present, booleans well-formed | 5 |

`test_no_target_leakage` is the most important new test: it makes the F1 leakage class **un-reintroducible** as families are added.

---

## 6. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Refactor silently changes linear numbers | Med | Phase-0 frozen-fixture regression test gates every phase |
| Nonlinear sim numerically blows up on M4 | Med | Reuse magnetic sim's `np.clip` acceleration/position bounds; keep `beta` range small; winsorize targets |
| Categorical/continuous metric mismatch in unified ladder | Med | Route metric by `family.target_kind`; no shared-axis averaging (matches paper §4.2 discipline) |
| New family accidentally re-leaks target | Low/High-impact | `test_no_target_leakage` runs for every family in CI |
| Scope creep into confirmatory MLP/Transformer (a *separate* gap) | Low | Out of scope here; F5 uses existing Ridge/KNN estimators only |
| Tuning a gate to make a new family pass | Low/High-impact | Explicit rule: gates are frozen; a family FAIL is reported as an honest result, never rescued |

---

## 7. Honest scope boundaries

- This plan does **not** add the MLP/Transformer confirmatory estimators (a separate P2 gap) — F5 is solved with the estimators already in use.
- This plan does **not** touch the human-subject protocol; it is a developmental-simulation change only.
- A cross-family **FAIL** for some gate is an acceptable, publishable outcome — it would sharpen the manuscript's claim ladder (which gates are family-robust), not weaken it. Success is defined as *measuring* cross-family behavior, not as every gate passing everywhere.
- All scaling knobs stay in `configs/`; defaults remain small enough for a local M4 machine.

---

## 8. Definition of done (acceptance)

- [ ] `pytest -q` green including the 7 new tests.
- [ ] `results/cross_family/results.json` contains `by_source` for ≥3 families and a cross-family gate matrix.
- [ ] `results/cross_family/report.md` contains the gate matrix table and per-gate interpretation; figures referenced.
- [ ] `test_no_target_leakage` passes for every family (F1 cannot regress).
- [ ] Manuscript updated so cross-family claims trace to the measured matrix; Appendix F.5 closed.
- [ ] No existing gate threshold changed; any family FAIL reported as an honest result.
