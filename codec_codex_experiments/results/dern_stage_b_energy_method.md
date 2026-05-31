# Stage-B energy-probe method audit

Before trusting any measured-joules number, the `powermetrics` probe was audited.
Six flaws were found in the first version and fixed in `src/dern_b/energy_probe.py`.

| # | Flaw (v1) | Fix (v2) |
|---|---|---|
| 1 | Parser format assumed (`"...Power: NNN mW"`), unverified against the real machine | Strict per-component regex (`CPU/GPU/ANE Power: NNN mW`); a `--inspect` mode dumps the real format to verify before trusting numbers; unrecognized format -> `unavailable`, never a guess |
| 2 | Averaging all `mW` lines could mix a 'Combined' total with components -> double counting | Match ONLY CPU/GPU/ANE component lines, SUM them per sample, explicitly ignore any 'Combined'/total line (asserted by `test_parser_sums_components_and_ignores_combined`) |
| 3 | `avg_power * wall_clock` is wrong integration | Riemann sum: sum(power_sample * sampling_interval) with the known `-i` interval as each sample's weight |
| 4 | Sampling window misaligned with `fn` (powermetrics startup lag) | Energy integrated over the SAMPLED window; a flush `sleep(interval)` before terminate so the last sample lands |
| 5 | No idle baseline; total-system power over-attributes to the model | Measure idle baseline first; report BOTH total-system and idle-subtracted joules, clearly labeled as total-system (not model-only) |
| 6 | `terminate()` could drop buffered samples (esp. short fn) | Idle phase uses fixed `-n` count (exits cleanly); active phase flushes before terminate; zero parsed samples -> `unavailable` |

## Honesty boundary

- `powermetrics` reports **total-system** power, not the model's alone. Even
  idle-subtracted, attribution is approximate (background load shifts during a
  multi-second run). Reported as "total-system energy during the cascade,
  idle-subtracted," NOT "energy of the model."
- Requires sudo on a real TTY. The assistant's non-interactive shell cannot
  prompt for a password, so the measured run is executed by the user in their own
  terminal via `python -m src.dern_b.run_energy` (password never enters the
  session). Until that is run, joules stays `unavailable` — never fabricated.

## Measured reading

(To be filled in from the user's terminal run of `python -m src.dern_b.run_energy`.)
