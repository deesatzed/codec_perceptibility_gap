"""DERN Stage B — real-model distinction-governed cascade (mlx-lm, local).

Bounded-loss cascade: a cheap local model is served when its output is verified
within epsilon of the reference model; otherwise the reference is served. Cost is
MEASURED (tokens, wall-clock, active-parameter-seconds). True joules are an
opt-in, sudo-gated measurement and are never fabricated. See
docs/plans/2026-05-31-dern-stage-b-design-and-plan.md.
"""
