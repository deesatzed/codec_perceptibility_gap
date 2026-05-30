# Codec Codex Experiments — Handoff Packet
**Generated:** 2026-05-30 (updated: F5 cross-family gap CLOSED)
**Branch:** N/A (not a git repository) @ N/A
**Last Commit:** N/A — version control not initialized for this directory

> **Update:** The P1 F5 cross-family generalization gap is now **implemented and closed** (plan `docs/PLAN_F5_cross_family.md` executed Phases 0–5). New: `src/sources.py`, `src/cross_family.py`, 3 test files (11 tests total, all pass), `results/cross_family/`, `results/cross_family_results.md`. Manuscript bumped to v1.4 (`../perceptibility_gap_paper_v1_4.md`). Gate matrix: **7/9 gates family-robust**; S5 now family-specific (informative), S7 fails on both (honest negative, flagged for redesign). See `results/cross_family_results.md`.

---

## Quick Resume Checklist
- [ ] `cd /Volumes/WS4TB/codec-exper/codec_codex_experiments`
- [ ] Environment: `source .venv/bin/activate` (venv already exists; otherwise `python3 -m venv .venv && source .venv/bin/activate`)
- [ ] Install deps (only if recreating venv): `pip install -r requirements.txt`
- [ ] Verify: `pytest -q` — expect **4 passed**
- [ ] Review "Current Blockers" and "Next Steps" below

## AI Continuity Checklist
- [x] Latest handoff reviewed — none existed prior to this one
- [x] Open assumptions imported — see "Open Questions / Decisions Needed"
- [x] Open debt items imported — see "Known Issues & Tech Debt"
- [x] Open error references imported — none open (no error log file in repo)
- [x] Verification suite executed — `pytest -q` → 4 passed (2026-05-30)
- [x] Next actions prioritized (P0/P1/P2)

---

## What This Project Does
A local, runnable simulation backbone for the **Perceptibility Gap / CodecGuard / CodecBench** research program (companion to the manuscript `../perceptibility_gap_paper_v1_3.md`). It executes four developmental smoke tests: a gated **proof ladder** (S0–S8) for distinction-vs-throughput claims, **CodecGuard** (multi-codec disagreement as a reliability/audit signal), **CodecBench** (codec-robustness recovery curve), and a **magnetic-pendulum** nonlinear stress test. These are simulation/software design-validity checks — explicitly **not** human-subject or confirmatory evidence.

**Tech Stack:** Python 3.13, numpy, scikit-learn (Ridge + KNN estimators), matplotlib, pandas, pytest. No LLM/API/network/database dependencies.
**Architecture Pattern:** CLI / library. Single-command orchestrator (`python -m src.run_all`) over three independent experiment modules + a report writer.

---

## Project Structure
```
codec_codex_experiments/
├── goal / GOAL.md         # Codex mission + definition of done
├── README.md              # Human setup instructions
├── MANIFEST.md            # File-by-file manifest
├── Makefile               # test / demo / confirmatory / clean targets
├── pytest.ini             # sets pythonpath=. so `src` imports under pytest
├── requirements.txt       # numpy, scikit-learn, matplotlib, pandas, pytest (floor pins)
├── configs/
│   ├── demo.json          # fast local run (1 seed)
│   └── confirmatory_local.json  # larger run (8 seeds — raised from 5 in mitigation)
├── src/
│   ├── run_all.py         # orchestrator entry point
│   ├── proof_ladder.py    # simulator + S0–S8 gates + estimators + metrics
│   ├── codec_contest.py   # CodecGuard (audit) + CodecBench (robustness)
│   ├── magnetic_pendulum.py # nonlinear stress test
│   └── report.py          # report.md writer + matplotlib figures
├── tests/test_smoke.py    # 4 smoke tests
├── examples/tasks.jsonl   # starter tasks for OPTIONAL LLM/Ollama port (unused)
├── prompts/               # OPTIONAL LLM port prompt templates (unused)
├── docs/                  # experiment_notes.md, reporting_template.md
└── results/
    ├── demo/              # original bundled sample output
    ├── demo_run/          # my first verified demo run (pre-mitigation)
    ├── confirmatory_local/# pre-mitigation 5-seed run (+ report_auto.md, curated report.md)
    ├── demo_mitigated/    # post-mitigation demo run
    ├── confirmatory_mitigated/  # post-mitigation 8-seed run (CURRENT canonical results)
    ├── critical_review.md # 8-finding methodological audit (F1–F8)
    └── mitigation_results.md  # before/after evidence for the applied fixes
```

**Entry Points:**
- `src/run_all.py` — `python -m src.run_all --config <cfg> --out <dir>`; runs all three experiments, writes `results.json` + `report.md` + figures.

**Key Modules:**
| Module | Path | Purpose | Status |
|--------|------|---------|--------|
| Orchestrator | `src/run_all.py` | One-command bundle runner | ✅ |
| Proof ladder | `src/proof_ladder.py` | Simulator + S0–S8 gates, Ridge estimator, nRMSE, surrogate/residual tests | ✅ |
| CodecGuard/Bench | `src/codec_contest.py` | Disagreement-error audit + LOCO + permutation null; recovery curve | ✅ |
| Magnetic | `src/magnetic_pendulum.py` | Nonlinear basin/divergence test; honest physics vs leak-control channels | ✅ |
| Report writer | `src/report.py` | report.md tables + 2 figures | ✅ |

---

## How to Run

### Local Development
```bash
# Setup (one-time; venv already present)
cd /Volumes/WS4TB/codec-exper/codec_codex_experiments
source .venv/bin/activate
pip install -r requirements.txt   # only if recreating the venv

# Run demo (fast, 1 seed)
python -m src.run_all --config configs/demo.json --out results/demo_run

# Run confirmatory (8 seeds, ~13s on M4 Pro)
python -m src.run_all --config configs/confirmatory_local.json --out results/confirmatory_mitigated
```
**Expected output:** four "Running…/Wrote…" lines; `results.json`, `report.md`, and `figures/*.png` appear under `--out`.

Makefile shortcuts: `make test`, `make demo`, `make confirmatory`, `make clean`.

### Tests
```bash
pytest -q
```
**Current Status:** **4 passing, 0 failing, 0 skipped** (verified 2026-05-30).
**Known Failures:** none.

### Verification Suite
```bash
source .venv/bin/activate && pytest -q && \
python -m src.run_all --config configs/demo.json --out results/demo_run && \
test -f results/demo_run/results.json && test -f results/demo_run/report.md && echo "VERIFY_OK"
```
**Pass Condition:** `4 passed` from pytest AND the final line prints `VERIFY_OK`.

---

## Current State Assessment

### What's Working ✅
- All four experiments run end-to-end on demo and 8-seed confirmatory configs — `results/confirmatory_mitigated/` is the canonical post-mitigation output.
- `pytest -q` → 4 passed.
- Figures render (`codecbench_recovery_curve.png`, `magnetic_basin_accuracy.png`).
- Mitigations from the critical review are implemented and validated (see `results/mitigation_results.md`): F1 leakage fix, F2 gate, F3 LOCO+null, F4 S5 surrogate+residual, F6/F8 reporting.
- Companion manuscript updated to v1.3 (`../perceptibility_gap_paper_v1_3.md`) and aligned to these numbers.

### What's Incomplete ⚠️
- **Optional LLM/Ollama port** — `prompts/` + `examples/tasks.jsonl` describe a path to reimplement CodecGuard against a real local model (prose/JSON/code/causal-graph projections). Not implemented; no code references these files. Requires the user to select the model (see Open Questions).
- **Confirmatory estimators (MLP + small Transformer)** — named in the manuscript Study-1 plan but **not run**; developmental code uses Ridge + KNN only.

### What's Broken ❌
- None known.

### Current Blockers 🚧
- **Model selection for the LLM port** — per user policy, the user selects all LLM model versions (via OpenRouter). The Ollama port cannot proceed until a model is named.

### Feature Completion Matrix
| Feature | Status | Evidence | Gap to Done | Priority |
|---------|--------|----------|-------------|----------|
| Proof ladder S0–S8 | ✅ | `src/proof_ladder.py:215`; `results/confirmatory_mitigated/results.json` | S5 & S7 are honest informative FAILs (by design) | — |
| CodecGuard audit + LOCO/null | ✅ | `src/codec_contest.py:63` | none | — |
| CodecBench recovery curve | ✅ | `src/codec_contest.py:148` | none | — |
| Magnetic honest physics + leak control | ✅ | `src/magnetic_pendulum.py:179` | none | — |
| Report writer + figures | ✅ | `src/report.py:101` | none | — |
| Cross-family generalization (F5) | ❌ | `results/critical_review.md` F5 | parameterize generator behind `source` key, run ladder on ≥2 families | P1 |
| LLM/Ollama CodecGuard port | ⚠️ | `prompts/codecguard_ollama_prompts.md` | full implementation + model choice | P2 |
| Confirmatory MLP/Transformer run | ⚠️ | manuscript Sec 4.5 | set `estimator: "mlp"` in config; Transformer not yet coded | P2 |

---

## Recent Changes
Not a git repo, so changes are tracked by session artifacts rather than commits. This session's work, newest first:

| Date | Artifact | Change | Why |
|------|----------|--------|-----|
| 2026-05-30 | `../perceptibility_gap_paper_v1_3.md` | Manuscript v1.2b→v1.3 | Align paper to corrected re-run evidence; retract leakage-based magnetic result |
| 2026-05-30 | `src/*.py`, `configs/confirmatory_local.json` | Implemented mitigations F1–F4, F6, F8 | Fix label leakage, weak controls, tautology risk, reporting gaps |
| 2026-05-30 | `results/mitigation_results.md` | Before/after evidence doc | Document that fixes improved integrity, not headline numbers |
| 2026-05-30 | `results/critical_review.md` | 8-finding methodological audit | Adversarial review of methods + results |
| 2026-05-30 | `results/{demo_run,confirmatory_local}/` | Initial verified runs + curated report | Goal-required end-to-end run with environment summary |

**Uncommitted Changes:** N/A (no VCS). All edits are on disk.
**Stashed Work:** none.

---

## Configuration & Secrets

### Environment Variables
| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| (none) | No secrets/API keys required | — |

### External Dependencies
| Service | Purpose | Local Alternative |
|---------|---------|-------------------|
| (none) | Fully self-contained; numpy/sklearn only | N/A |
| Ollama (optional) | Only if implementing the LLM CodecGuard port | local Ollama install + user-selected model |

---

## Known Issues & Tech Debt
- [ ] **F5 single source family (P1)** — every ladder stage draws from one generator at a time; no cross-family generalization. Largest structural limitation. (`results/critical_review.md` §F5)
- [ ] **S5 is an honest FAIL** — coupling channel carries no signal unique from native on the linear family (residual nRMSE 1.003). This is correct/intended, not a bug; do not "fix" by loosening the gate. (`src/proof_ladder.py` S5)
- [ ] **S7 is an expected boundary FAIL** — linear source does not get harder with dimensions; resolved only by a nonlinear generator (overlaps F5). (`src/proof_ladder.py:380`)
- [ ] **`results/demo/`** is the original bundled sample; `results/demo_run/` and `results/*_mitigated/` are this session's runs. Consider pruning stale dirs to avoid confusion.
- [ ] No git history — consider `git init` to track future changes.

---

## Next Steps (Priority Order)
1. **(P1) Implement cross-family generalization (F5).** Detailed phased plan written: `docs/PLAN_F5_cross_family.md` (6-phase build checklist, 7-test plan, regression-guarded refactor through a `SourceFamily` seam). Add a `source`/`sources` config key (`linear_oscillator`, `nonlinear_oscillator`, `magnetic`) and run the full ladder across ≥2 families, reporting each gate per source. Keep defaults small for local M4. "Done" = ladder runs on ≥2 families with per-source results + a cross-family gate matrix in `results.json`, and the manuscript's cross-family claim becomes supportable. **Needs user approval before implementing (experiment-spec change).**
2. **(P2) Confirmatory estimator pass.** Run `estimator: "mlp"` config variant and add a small-Transformer estimator to `fit_predict`; report numbers separately. "Done" = MLP/Transformer results exist and the manuscript can replace "not yet run" with real numbers.
3. **(P2) Optional LLM/Ollama CodecGuard port.** Requires user to select a model first. Implement prose/JSON/code/causal-graph projection parsing per `prompts/codecguard_ollama_prompts.md`, validate against `examples/tasks.jsonl`. "Done" = disagreement metrics computed from a real local model.
4. **(P3) Housekeeping.** `git init`; prune redundant `results/` dirs; optionally fold the review's failure-classification logic into `src/report.py` (F6 suggestion #6).

---

## Key Files Reference
| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/proof_ladder.py` | Simulator, S0–S8 gates, estimators, nRMSE/surrogate/residual helpers | Adding stages, source families (F5), estimators |
| `src/codec_contest.py` | CodecGuard audit (+LOCO/null), CodecBench curve | Audit metric changes, new codecs |
| `src/magnetic_pendulum.py` | Nonlinear test; `_physics_expanded_features` vs leak control | Nonlinear source changes, gate tuning |
| `src/report.py` | report.md tables + figures | New metrics to surface, plot changes |
| `configs/*.json` | All scaling knobs (seeds, n_train/test, magnets, noise) | Tuning run size; keep M4-friendly defaults |
| `results/critical_review.md` | The methodological audit (F1–F8) | Reference before changing controls/gates |
| `results/mitigation_results.md` | Before/after evidence | Reference for what each fix did |

---

## Open Questions / Decisions Needed
- **LLM model for the Ollama port** — user must select (policy: user selects all LLM versions via OpenRouter, models change weekly). Port is blocked until then.
- **Prune stale results dirs?** — `results/demo/` (bundled sample) and `results/confirmatory_local/` (pre-mitigation) are superseded by `*_mitigated/`. Keep for provenance or remove?
- **Initialize git?** — currently untracked; recommended before further changes.

---

## Appendix: Machine-Readable Summary
```json
{
  "project": "codec_codex_experiments",
  "generated": "2026-05-30",
  "repo": {
    "branch": null,
    "commit": null,
    "commit_date": null,
    "uncommitted_changes": false,
    "stashed_work": 0,
    "version_control": "none"
  },
  "stack": {
    "language": "python",
    "language_version": "3.13.9",
    "framework": "numpy+scikit-learn",
    "framework_version": "numpy 2.4.6 / scikit-learn 1.8.0"
  },
  "health": {
    "tests_passing": 4,
    "tests_failing": 0,
    "tests_skipped": 0,
    "lint_clean": null,
    "type_check_clean": null
  },
  "status": {
    "working": ["proof_ladder", "codecguard", "codecbench", "magnetic_pendulum", "report_writer"],
    "incomplete": ["llm_ollama_port", "confirmatory_mlp_transformer"],
    "broken": [],
    "blockers": ["llm_model_selection_required_for_ollama_port"]
  },
  "continuity": {
    "previous_handoff_loaded": false,
    "assumptions_imported": 0,
    "debt_items_imported": 0,
    "error_refs_imported": 0
  },
  "feature_completion_matrix": [
    {"feature": "proof_ladder_s0_s8", "status": "✅", "evidence": "src/proof_ladder.py:215", "priority": "P0"},
    {"feature": "codecguard_audit_loco_null", "status": "✅", "evidence": "src/codec_contest.py:63", "priority": "P0"},
    {"feature": "codecbench_recovery", "status": "✅", "evidence": "src/codec_contest.py:148", "priority": "P0"},
    {"feature": "magnetic_physics_vs_leak", "status": "✅", "evidence": "src/magnetic_pendulum.py:179", "priority": "P0"},
    {"feature": "cross_family_generalization", "status": "❌", "evidence": "results/critical_review.md#F5", "priority": "P1"},
    {"feature": "llm_ollama_port", "status": "⚠️", "evidence": "prompts/codecguard_ollama_prompts.md", "priority": "P2"},
    {"feature": "confirmatory_mlp_transformer", "status": "⚠️", "evidence": "perceptibility_gap_paper_v1_3.md#4.5", "priority": "P2"}
  ],
  "verification_suite": {
    "command": "source .venv/bin/activate && pytest -q && python -m src.run_all --config configs/demo.json --out results/demo_run && test -f results/demo_run/results.json && test -f results/demo_run/report.md && echo VERIFY_OK",
    "pass_condition": "pytest reports '4 passed' and final line is 'VERIFY_OK'",
    "result": "pass"
  },
  "next_steps": [
    {"task": "Implement cross-family generalization (source config key, run ladder on >=2 families)", "priority": "P1", "scope": "medium"},
    {"task": "Run confirmatory MLP + add small-Transformer estimator", "priority": "P2", "scope": "medium"},
    {"task": "Implement optional LLM/Ollama CodecGuard port (blocked on model selection)", "priority": "P2", "scope": "large"},
    {"task": "git init, prune stale results dirs", "priority": "P3", "scope": "small"}
  ]
}
```
