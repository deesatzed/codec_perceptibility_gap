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

## Two more flaws found during real measurement (and fixed)

The first two real runs exposed flaws the synthetic tests could not:

| # | Flaw (found on real hardware) | Fix |
|---|---|---|
| 7 | `avg_power * (n_samples * interval)` inflated energy ~7x: 10 background samples (~2s) integrated for a 0.27s route | Energy = `avg_power * ACTUAL work_seconds` (window-aligned); too-short guard if work < one interval |
| 8 | Backgrounded powermetrics piped to `communicate()` returned only ~10 samples for a 53s window (pipe buffering + terminate truncation) -> avg from an unrepresentative 2s slice; and the 53s was dominated by one-time 21GB model LOADING | (a) write powermetrics to a FILE for the full window; (b) **sample-coverage guard**: return `unavailable` if captured samples cover <60% of the work window (never stretch a sparse sample); (c) **pre-warm** both models before the measured batch so energy reflects inference, not loading |

The parser was also corrected against real output (a stray per-cluster `GPU Power`
line outside the CPU->GPU->ANE block created phantom samples; sample now starts
only on a `CPU Power` line).

## Measured reading (validated)

Run: `python -m src.dern_b.run_energy` on Apple M4 Pro, 64 GB, mlx-lm 0.31.3.
Models pre-warmed (load excluded). Batch = 6 factual prompts, gemma-4 e4b -> 31B
cascade, ε=0.5.

| Quantity | Value |
|---|---|
| work window | 39.70 s |
| active samples / coverage | 177 samples x 200 ms = 35.4 s = **89% coverage** (>60% guard -> valid) |
| avg power (active) | **18.19 W** (total system) |
| avg power (idle baseline) | 0.66 W |
| batch energy, total-system | 722.1 J |
| batch energy, idle-subtracted | **695.9 J** |
| per-route mean, idle-subtracted | **~116 J** |
| routes served by cheap (verified) | 5 / 6 |

**Honest scope:** total-system power (not model-only); idle-subtracted is the
better proxy. This is the energy of running the *cascade* (6 routes, mostly
cheap-served with 1 reference audit) over ~40 s, with models already loaded. It
is NOT a per-token figure and NOT a comparison vs always-reference energy (that
would require measuring an always-31B batch under the same probe — a clean next
step). Coverage 89% and avg-power/idle separation (18.2 W vs 0.66 W) make the
reading trustworthy; the value supersedes the earlier ~15.5 J and ~393 J readings,
which were the flawed-probe artifacts documented above.
