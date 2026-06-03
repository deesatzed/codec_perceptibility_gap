# Changelog

## 0.1.0 — initial release

First public release of `claimgate` — gate a quantitative claim behind adversarial
controls and get back `Verified(value, receipts)` or `Refused(reason, evidence)`.

- **Contract** (`Verified | Refused`) with enforced extraction (Rust-`Result`-style):
  `unwrap()`/`expect()` raise on a refusal, `.value` is trapped, `match()` forces
  both arms, `unwrap_or(default)` is the explicit out.
- **Typestate**: `Earned` brand (mintable only from a `Verified`) and
  `require_verified()` boundary guard so downstream functions can demand earned
  numbers and reject raw ones.
- **Explicit pipeline**: `and_then`/`map` short-circuit a refusal; `Battery`
  validates its configuration at construction.
- **Controls** (all standard statistics): `base_rate`, `beats_difficulty`
  (bootstrap-CI AUC gap), `permutation_null`, `collinearity`, `coverage`,
  `improvement_beats_noise` ("+1% is not enough" paired-bootstrap guardrail).
- **Worked examples**: the `+1%` over-claim refused; a difficulty-proxy refused;
  a control-dependent carbon-additionality claim refused (real public data).

Honest scope: this is rigor packaged for reuse, not a novel technique. Every
control is standard statistics; the value is the composition + the refuse-by-default
contract.
