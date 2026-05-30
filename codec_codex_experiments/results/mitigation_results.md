# Mitigation Results — Codec Codex Experiments

This documents the code changes made in response to `critical_review.md`, and the before/after evidence from re-running the bundle. All changes were approved before implementation. The confirmatory comparison uses 8 seeds (raised from 5 per F6).

Output dirs:
- Pre-mitigation: `results/demo_run`, `results/confirmatory_local`
- Post-mitigation: `results/demo_mitigated`, `results/confirmatory_mitigated`

`pytest -q` → **4 passed** after all changes. Confirmatory bundle wall clock ~13 s (8 seeds, M4 Pro).

---

## F1 — Magnetic label leakage → honest physics channel + explicit leakage control (CRITICAL)

**Change** (`src/magnetic_pendulum.py`): added `_physics_expanded_features()` — a channel built only from the preview trajectory (velocity profile, speed extremes, radial dynamics, cross-coordinate moment), containing **no label and no divergence target**. The old channel (native + noisy-label one-hot + noisy-divergence target) is retained but renamed `oracle_leak_positive_control` and reported separately. Report and plot updated to show all four channels.

**Before → after (basin accuracy, confirmatory):**

| m | native | OLD "expanded" (leaked) | NEW expanded_physics | direct | oracle-leak ctrl |
|---|---|---|---|---|---|
| 3 | 0.946 | 0.968 | **0.925 ± 0.017** | 0.967 | 0.968 |
| 4 | 0.924 | 0.936 | **0.884 ± 0.008** | 0.952 | 0.939 |
| 5 | 0.889 | 0.902 | **0.855 ± 0.015** | 0.927 | 0.906 |

**Result:** the leaked "expanded beats native" effect **disappears** once the target is removed. Honest expanded-physics is *slightly below* native (mean delta −0.029); `direct` (full trajectory) is best. The oracle-leak control sits at its noise ceiling (~1−flip_prob), now clearly labeled as not-evidence-of-access. **The prior report's "clearest positive signal" claim is retracted and the artifact is fully neutralized.**

---

## F2 — Self-contradictory magnetic slope gate → honest graceful-degradation gate (High)

**Change:** removed the `mean(expanded) > mean(native) + 0.10` condition (unsatisfiable: RHS exceeded 1.0 because native is near-ceiling). New gate: `min(expanded_physics) > chance + 0.20` **AND** `expanded[first] − expanded[last] ≤ 0.15` (no collapse across magnet count). The expanded-minus-native delta is reported as information, not required for PASS.

**Result (confirmatory):**
```
slope_gate_passed: True
{"expanded_min_above_chance_plus_0.20": true,
 "graceful_no_collapse_le_0.15": true,
 "expanded_minus_native_mean_delta": -0.029}
```
The gate now PASSES on an honest criterion: the physics channel stays well above chance and degrades gracefully (0.925 → 0.855, drop 0.07 < 0.15). It no longer requires leakage to pass.

---

## F3 — CodecGuard semi-circularity → LOCO + permutation null (High)

**Change** (`src/codec_contest.py`): added (a) **leave-one-codec-out (LOCO)** correlation — for each held-out codec, predict *its* error from the disagreement of the *other* codecs (removes self-reference entirely); and (b) a **permutation null** — shuffle the disagreement vector and report the 95th-percentile null correlation, plus a per-seed `clears_permutation_null` flag and an `all_seeds_clear_permutation_null` summary.

**Result (confirmatory):**
```
corr_disagreement_vs_ensemble_error      = 0.751 ± 0.032
loco_disagreement_vs_held_codec_error_corr = 0.615     <- no self-reference, still strong
permutation_null_p95                     = 0.073
all_seeds_clear_permutation_null         = True
mean_pairwise_codec_error_correlation    = 0.565   (overconfidence warning, unchanged)
```
**Result:** the disagreement→error signal **survives both checks**. LOCO (0.615) confirms it is not a std/|mean−truth| artifact; the observed correlation clears the permutation null (0.073) on every seed. The finding is now defensible. The correlated-error warning (0.565) is retained.

---

## F4 — S5 independent-shuffle sham → phase-randomized surrogate + unique-contribution test (Medium)

**Change** (`src/proof_ladder.py`): added `phase_randomized_surrogate()` (distribution-preserving per-column row permutation) replacing the old independent train/test shuffle. Added a **unique-contribution test**: fit native→z, then test whether the coupling channel predicts the *native residual* (`coupling_on_native_residual_nrmse`, `coupling_has_unique_signal`).

**Result (confirmatory):**
```
native_only = 0.533   expanded = 0.529   sham(surrogate) = 0.534
coupling_on_native_residual_nrmse = 1.003   -> coupling_has_unique_signal: False
```
**Result:** the unique-contribution test is decisive. Coupling features predict the native residual at nRMSE 1.003 — **no better than the mean** (1.0 = mean-only). This sharpens the prior "redundant at scale" conclusion into a precise statement: **the coupling channel carries no information unique from native** on this generator. The S5 FAIL is now explained mechanistically, not just by a thin margin.

---

## F6 — Seed/uncertainty discipline (Medium)

**Change:** confirmatory config seeds raised 5 → **8** (both proof-ladder and magnetic blocks). Magnetic per-magnet metrics now report across-seed **SD** alongside the mean; CodecGuard reports `corr_..._sd`.

**Result:** across-seed SDs are small (magnetic accuracy SD ~0.01–0.02; CodecGuard corr SD 0.032), indicating the point estimates are stable across data-generating seeds — uncertainty is now visible rather than implied.

---

## F7 — Gate-source / reporting clarity (Low)

**Change:** `report.py` now prints the magnetic gate's three sub-conditions (`slope_gate_conditions`) inline, so a PASS/FAIL is self-explaining and tied to the single gate computed in the module. Added an explicit note distinguishing `expanded-physics` from the `oracle-leak` control in the report body.

---

## F8 — Scalar nRMSE hides per-target behavior (Low)

**Change:** added `per_target_nrmse()` and surfaced it on S0.

**Result (confirmatory):**
```
labels:           [spring1, spring2, coupling_g, damping_c]
per_target_nrmse: [0.008,   0.239,   0.008,      0.181]
```
**Result:** the scalar S0 mean (0.23) concealed that spring1 and coupling_g are recovered nearly perfectly (0.008), while spring2 (0.239) and damping_c (0.181) are materially harder. This is now visible and could inform a per-target gate later.

---

## Net effect

The mitigations **improved the experiment's integrity**, not its headline numbers — which is the correct outcome:

- One **invalid** positive result (magnetic expanded-beats-native) was identified as leakage and **removed**; replaced by an honest channel that gives a clean negative ("hand-engineered geometry does not beat the native preview; the full trajectory carries the basin information").
- One **borderline** result (S5) was sharpened from "redundant" to a decisive "no unique signal" via the residual test.
- One **at-risk** result (CodecGuard) was **defended** with LOCO + a permutation null and now clears both.
- Uncertainty and per-target structure are now reported, so future readers cannot be misled by scalar summaries.

Remaining open item (not implemented, lower priority): **F5** — multi-source-family generalization (parameterize the generator behind a `source` key and run the ladder on ≥2 families). This is a larger structural change; recommend as the next milestone if cross-family claims are intended.
