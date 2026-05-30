# Codec Codex Experiments

Local runnable package for the two software embodiments of the codec-contest principle plus the simulation backbone from the Perceptibility Gap program.

## What this repo runs

1. **Proof ladder** (`src/proof_ladder.py`): gated simulation checks for distinction-dependence, throughput-invariance, wrong-lever behavior, additive-access proxy, shared placement, dimensionality slope, and multi-codec audit signal.
2. **CodecGuard smoke test** (`src/codec_contest.py`): multi-codec disagreement as a reliability signal; reports disagreement-error correlation, leave-one-codec-out correlation, permutation null, AUC, catch-rate, calibration, and correlated-error risk.
3. **CodecBench smoke test** (`src/codec_contest.py`): codec-robustness curve; reports recoverable structure retained as distinction count is compressed.
4. **Magnetic-pendulum stress test** (`src/magnetic_pendulum.py`): nonlinear source-family probe for basin-label and finite-time divergence targets under a Lyapunov-time discipline, with an honest `expanded_physics` channel separated from an explicit `oracle_leak_positive_control`.
5. **Cross-family generalization** (`src/cross_family.py`, `src/sources.py`): runs the ladder + codec contest across a registry of source families (linear and Duffing-nonlinear oscillators) and reports a gate matrix marking which gates are family-robust.

These are simulation and software smoke tests. They are useful because they make the measurement logic executable, not because they confirm the human-subject hypotheses. For the full story (the audit that found a leakage bug, the fixes, and the cross-family results) see the narrative writeup at `../README.md`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.run_all --config configs/demo.json --out results/demo
```

Then open:

```text
results/demo/report.md
results/demo/results.json
```

## Larger / cross-family run

```bash
# 8-seed, two-family (linear + nonlinear) run with the cross-family gate matrix
python -m src.run_all --config configs/confirmatory_local.json --out results/cross_family
```

`configs/confirmatory_local.json` declares `"sources": ["linear_oscillator", "nonlinear_oscillator"]`; when more than one source is listed, the bundle additionally emits a `cross_family` block (per-family gate results + a robustness matrix) into `results.json` and a "Cross-family generalization" section into `report.md`.

On an M4 mini, start with `configs/demo.json` (single family, fast); increase `n_train`, `n_test`, `seeds`, and add families only after the smoke run succeeds. Scaling knobs live in the config files only.

## Source families

`src/sources.py` defines a `SourceFamily` protocol and registry. To add a family, implement `sample`, `native_channel`, and `expanded_physics_channel`, register it, and add its key to a config `sources` list. **Invariant:** no channel may expose the scored target (enforced by `tests/test_sources.py::test_no_target_leakage`).

## Optional LLM port

The prompts in `prompts/codecguard_ollama_prompts.md` describe the direct path to a local Ollama implementation: ask the same model for prose, JSON schema, executable code, and causal graph projections; parse them to canonical claim atoms; diff them; then validate disagreement against known-answer tasks from `examples/tasks.jsonl`.
