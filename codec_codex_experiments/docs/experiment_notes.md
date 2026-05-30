# Experiment notes

This package operationalizes two shippable ideas:

1. **CodecGuard**: a runtime reliability layer. The proxy implementation uses independent feature projections; the LLM implementation should use prose, schema, executable code, and causal graph projections parsed into canonical claims.
2. **CodecBench**: an offline robustness benchmark. The proxy implementation degrades a structured channel through lower distinction counts; the LLM implementation compresses answers through prose -> bullets -> schema -> graph -> single contrastive feature and measures claim retention.

Important claim hygiene:

- These are reach-frontier tests, not horizon tests.
- The magnetic pendulum must be simulated for ground truth; a physical toy is motivation/demo only.
- In chaotic systems, use basin labels, finite-time divergence, and short-horizon prediction within the Lyapunov time. Do not score long-horizon exact-path recovery.
- Multi-codec consistency is an audit layer, not an alignment solution. Correlated errors remain the main risk.

## Integrity audit + cross-family (current state)

Two structural passes have been applied since the first run; both are documented with before/after evidence in `../results/`.

1. **Integrity audit** (`../results/critical_review.md`, `../results/mitigation_results.md`): found and removed target leakage in the magnetic "expanded" channel (a noisy copy of the label/divergence target was being fed back as a feature). The honest channel is now `expanded_physics`; the leaked one is retained only as `oracle_leak_positive_control`. Controls hardened (phase-randomized surrogate + native-residual unique-contribution test); CodecGuard de-tautologized (leave-one-codec-out + permutation null); per-target and across-seed reporting added; seeds raised to eight.
2. **Cross-family generalization** (`../results/cross_family_results.md`, plan in `PLAN_F5_cross_family.md`): the ladder + codec contest now run across a source-family registry. On linear + nonlinear oscillators, seven of nine gates are family-robust. S5 (additive access) is family-specific (FAIL linear, PASS nonlinear); S7 fails on both (honest negative, gate flagged for redesign).

No gate threshold has ever been tuned to force a PASS. A FAIL is reported as a negative.
