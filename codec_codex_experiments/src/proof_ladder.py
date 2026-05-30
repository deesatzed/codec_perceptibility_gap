#!/usr/bin/env python3
"""
Progressive experimental proof ladder.

This is a local, runnable simulation backbone. It is intentionally small enough
for a consumer workstation. Treat all outputs as developmental smoke-test
signals unless the config, seeds, and analysis plan are frozen in advance.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CONFIG: Dict[str, Any] = dict(
    n_train=450,
    n_test=180,
    T=12.0,
    dt=0.06,
    drive_amp=0.3,
    drive_w=0.7,
    seeds=[0, 1],
    mlp_hidden=(48,),
    mlp_iter=100,
    estimator="ridge",
    ridge_alpha=1.0,
    n_boot=120,
    eff_floor=0.02,
    delta_eq=0.04,
)


def normalize_cfg(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = dict(CONFIG)
    if cfg:
        out.update(cfg)
    # JSON cannot represent tuples; sklearn accepts tuples for hidden layers.
    if isinstance(out.get("mlp_hidden"), list):
        out["mlp_hidden"] = tuple(out["mlp_hidden"])
    return out


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return normalize_cfg({})
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = normalize_cfg(data)
    return base


# ----------------------------- simulator -----------------------------
def simulate(theta: np.ndarray, n_osc: int, cfg: Dict[str, Any], seed: int) -> np.ndarray:
    """theta: (B, n_osc+2) = [springs..., coupling g, damping c]."""
    B = theta.shape[0]
    springs = theta[:, :n_osc]
    g = theta[:, n_osc]
    c = theta[:, n_osc + 1]
    steps = int(cfg["T"] / cfg["dt"])
    x = np.zeros((B, n_osc), dtype=float)
    v = np.zeros((B, n_osc), dtype=float)
    x[:, 0] = 1.0
    out = np.empty((B, steps, n_osc), dtype=float)
    for t in range(steps):
        lap = np.zeros_like(x)
        if n_osc > 1:
            lap[:, 1:] += x[:, :-1] - x[:, 1:]
            lap[:, :-1] += x[:, 1:] - x[:, :-1]
        drive = cfg["drive_amp"] * np.sin(cfg["drive_w"] * t * cfg["dt"])
        a = -springs * x - c[:, None] * v + g[:, None] * lap + drive
        v = v + cfg["dt"] * a
        x = x + cfg["dt"] * v
        out[:, t, :] = x
    return out


def sample_theta(B: int, n_osc: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(7 + seed)
    springs = rng.uniform(0.5, 4.0, size=(B, n_osc))
    g = rng.uniform(0.05, 1.0, size=(B, 1))
    c = rng.uniform(0.10, 0.60, size=(B, 1))
    return np.concatenate([springs, g, c], axis=1)


# ----------------------------- channels ------------------------------
def native_feats(x: np.ndarray) -> np.ndarray:
    """Per-oscillator statistics. Deliberately omits richer cross-oscillator relations."""
    f = [
        x.mean(1),
        x.std(1),
        x[:, -1, :],
        np.abs(x).mean(1),
        x[:, : x.shape[1] // 2, :].std(1) - x[:, x.shape[1] // 2 :, :].std(1),
    ]
    return np.concatenate([a.reshape(x.shape[0], -1) for a in f], axis=1)


def coupling_feats(x: np.ndarray) -> np.ndarray:
    """Cross-oscillator structure: correlations and interaction moments."""
    B, _, N = x.shape
    if N < 2:
        return np.zeros((B, 1), dtype=float)
    cols = []
    for i in range(N - 1):
        a, b = x[:, :, i], x[:, :, i + 1]
        am, bm = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        corr = (am * bm).mean(1) / (am.std(1) * bm.std(1) + 1e-8)
        cols.append(corr)
        cols.append((a * b).mean(1))
    return np.stack(cols, axis=1)


def engineered_feats(x: np.ndarray) -> np.ndarray:
    return np.concatenate([native_feats(x), coupling_feats(x)], axis=1)


def direct_feats(x: np.ndarray, q: int = 8) -> np.ndarray:
    return x[:, ::q, :].reshape(x.shape[0], -1)


def quantize(F: np.ndarray, k: int, ref: np.ndarray) -> np.ndarray:
    """Ordinal quantization to k levels using train-set quantiles."""
    Q = np.empty_like(F, dtype=float)
    for j in range(F.shape[1]):
        edges = np.quantile(ref[:, j], np.linspace(0, 1, k + 1)[1:-1])
        Q[:, j] = np.digitize(F[:, j], edges)
    return Q.astype(float)


# ----------------------------- estimator -----------------------------
def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
    """Fit a small estimator and return predictions on the original target scale.

    The demo defaults to Ridge because Codex/local M4 runs need fast iteration.
    Set ``estimator: "mlp"`` in a config for the slower nonlinear dry run.
    """
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr)
    Xtr_s = xs.transform(Xtr)
    Xte_s = xs.transform(Xte)
    ytr_s = ys.transform(ytr)

    if str(cfg.get("estimator", "ridge")).lower() == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=cfg["mlp_hidden"],
            max_iter=int(cfg["mlp_iter"]),
            early_stopping=True,
            random_state=int(seed),
        )
        fit_y = ytr_s.ravel() if ytr_s.shape[1] == 1 else ytr_s
        model.fit(Xtr_s, fit_y)
        pred_scaled = model.predict(Xte_s).reshape(len(Xte), -1)
    else:
        model = Ridge(alpha=float(cfg.get("ridge_alpha", 1.0)), random_state=int(seed))
        model.fit(Xtr_s, ytr_s)
        pred_scaled = model.predict(Xte_s).reshape(len(Xte), -1)

    return ys.inverse_transform(pred_scaled)


def nrmse(yt: np.ndarray, yp: np.ndarray) -> float:
    yt = yt.reshape(len(yt), -1)
    yp = yp.reshape(len(yp), -1)
    return float(np.mean(np.sqrt(((yt - yp) ** 2).mean(0)) / (yt.std(0) + 1e-8)))


def per_target_nrmse(yt: np.ndarray, yp: np.ndarray) -> list:
    """F8: per-target normalized RMSE vector, so a gate cannot pass on the
    back of one easy target while another is unlearned."""
    yt = yt.reshape(len(yt), -1)
    yp = yp.reshape(len(yp), -1)
    v = np.sqrt(((yt - yp) ** 2).mean(0)) / (yt.std(0) + 1e-8)
    return [round(float(x), 3) for x in v]


def phase_randomized_surrogate(F: np.ndarray, seed: int) -> np.ndarray:
    """F4: distribution-preserving surrogate. Independently permutes the ROWS of
    each column, preserving each feature's marginal distribution but destroying
    its cross-feature / cross-row alignment with the target. Unlike the old
    independent-shuffle sham, the SAME surrogate construction is applied so the
    control isolates 'does this channel carry target-aligned information' rather
    than 'is this channel pure noise'."""
    rng = np.random.default_rng(seed)
    S = np.empty_like(F)
    for j in range(F.shape[1]):
        S[:, j] = F[rng.permutation(len(F)), j]
    return S


def boot_gap(yt: np.ndarray, a: np.ndarray, b: np.ndarray, cfg: Dict[str, Any]) -> Tuple[float, Tuple[float, float]]:
    """95% CI for nRMSE(a) - nRMSE(b), paired bootstrap over test rows."""
    n = len(yt)
    idx = np.arange(n)
    gaps = []
    rng = np.random.default_rng(99)
    for _ in range(int(cfg.get("n_boot", 120))):
        s = rng.choice(idx, n, replace=True)
        gaps.append(nrmse(yt[s], a[s]) - nrmse(yt[s], b[s]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(np.mean(gaps)), (float(lo), float(hi))


# ----------------------------- data cache ----------------------------
def make_dataset(n_osc: int, cfg: Dict[str, Any], seed: int, source: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce (train_traj, train_theta, test_traj, test_theta).

    Default (source=None) is the linear oscillator family, byte-identical to the
    original implementation. When a SourceFamily is supplied, datasets are drawn
    from it instead; `n_osc` is taken from the family's own configuration so the
    S7 dimensionality sweep can vary it per family.
    """
    if source is None:
        th_tr = sample_theta(int(cfg["n_train"]), n_osc, seed)
        th_te = sample_theta(int(cfg["n_test"]), n_osc, seed + 500)
        return (
            simulate(th_tr, n_osc, cfg, seed),
            th_tr,
            simulate(th_te, n_osc, cfg, seed + 500),
            th_te,
        )
    xtr, ttr = source.sample(int(cfg["n_train"]), cfg, seed)
    xte, tte = source.sample(int(cfg["n_test"]), cfg, seed + 500)
    return xtr, ttr, xte, tte


def avg(fn: Callable[[int], float], cfg: Dict[str, Any]) -> float:
    return float(np.mean([fn(int(s)) for s in cfg["seeds"]]))


def _round(x: Any, digits: int = 3) -> Any:
    if isinstance(x, (float, np.floating)):
        return round(float(x), digits)
    if isinstance(x, dict):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, digits) for v in x]
    return x


# ------------------------------ stages -------------------------------
def run_ladder(raw_cfg: Dict[str, Any] | None = None, source: Any = None) -> Dict[str, Any]:
    """Run the proof ladder. source=None uses the linear oscillator family
    (byte-identical to the original). A SourceFamily routes all stages through
    that family; S7's dimensionality sweep uses per-dimension instances of the
    same family type."""
    cfg = normalize_cfg(raw_cfg)
    rep: Dict[str, Any] = {}
    base = {int(s): make_dataset(2, cfg, int(s), source=source) for s in cfg["seeds"]}

    # S0 sanity
    def s0(seed: int) -> Tuple[float, float]:
        xtr, ttr, xte, tte = base[seed]
        yp = fit_predict(direct_feats(xtr), ttr, direct_feats(xte), cfg, seed)
        uni = np.repeat(ttr.mean(0)[None], len(tte), 0)
        return nrmse(tte, yp), nrmse(tte, uni)

    d, u = np.mean([s0(int(s)) for s in cfg["seeds"]], axis=0)
    # F8: per-target breakdown on the first seed, so a strong mean cannot hide an
    # unlearned target (theta = [spring1, spring2, coupling g, damping c]).
    _xtr0, _ttr0, _xte0, _tte0 = base[int(cfg["seeds"][0])]
    _yp0 = fit_predict(direct_feats(_xtr0), _ttr0, direct_feats(_xte0), cfg, int(cfg["seeds"][0]))
    rep["S0"] = dict(
        name="sanity",
        direct=d,
        uninformative=u,
        per_target_nrmse=per_target_nrmse(_tte0, _yp0),
        per_target_labels=["spring1", "spring2", "coupling_g", "damping_c"],
        passed=bool(d < 0.55 and 0.70 <= u <= 1.30 and d + cfg["eff_floor"] < u),
    )

    # S1 distinction-dependence
    def s1(seed: int, k: int) -> float:
        xtr, ttr, xte, tte = base[seed]
        Ftr = engineered_feats(xtr)
        Fte = engineered_feats(xte)
        return nrmse(tte, fit_predict(quantize(Ftr, k, Ftr), ttr, quantize(Fte, k, Ftr), cfg, seed))

    k_lo, k_hi = avg(lambda s: s1(s, 2), cfg), avg(lambda s: s1(s, 16), cfg)
    rep["S1"] = dict(name="distinction-dependence", k2=k_lo, k16=k_hi, passed=bool(k_lo - k_hi >= cfg["eff_floor"]))

    # S2 throughput-invariance (fixed k=8, replicate redundant bits)
    def s2(seed: int, m: int) -> float:
        xtr, ttr, xte, tte = base[seed]
        Ntr = native_feats(xtr)
        Nte = native_feats(xte)
        Ftr = quantize(Ntr, 8, Ntr)
        Fte = quantize(Nte, 8, Ntr)
        rng = np.random.default_rng(100 + m)
        Xtr = np.concatenate([Ftr + 0.01 * rng.standard_normal(Ftr.shape) for _ in range(m)], axis=1)
        Xte = np.concatenate([Fte + 0.01 * rng.standard_normal(Fte.shape) for _ in range(m)], axis=1)
        return nrmse(tte, fit_predict(Xtr, ttr, Xte, cfg, seed))

    t1, t8 = avg(lambda s: s2(s, 1), cfg), avg(lambda s: s2(s, 8), cfg)
    rep["S2"] = dict(
        name="throughput-invariance",
        thru1x=t1,
        thru8x=t8,
        delta=abs(t1 - t8),
        passed=bool(abs(t1 - t8) <= cfg["delta_eq"]),
    )

    # S3 wrong-lever
    def s3(seed: int) -> Tuple[float, float]:
        xtr, ttr, xte, tte = base[seed]
        Ftr, Fte = native_feats(xtr), native_feats(xte)
        lo_k = quantize(Ftr, 2, Ftr)
        lo_k_te = quantize(Fte, 2, Ftr)
        rng = np.random.default_rng(3)
        hiT = np.concatenate([lo_k + 0.01 * rng.standard_normal(lo_k.shape) for _ in range(6)], axis=1)
        hiT_te = np.concatenate([lo_k_te + 0.01 * rng.standard_normal(lo_k_te.shape) for _ in range(6)], axis=1)
        a = nrmse(tte, fit_predict(hiT, ttr, hiT_te, cfg, seed))
        hi_k = quantize(Ftr, 16, Ftr)
        hi_k_te = quantize(Fte, 16, Ftr)
        b = nrmse(tte, fit_predict(hi_k, ttr, hi_k_te, cfg, seed))
        return a, b

    hiT, hiK = np.mean([s3(int(s)) for s in cfg["seeds"]], axis=0)
    rep["S3"] = dict(
        name="wrong-lever",
        hi_throughput_lo_k=hiT,
        lo_throughput_hi_k=hiK,
        passed=bool(hiK + cfg["eff_floor"] <= hiT),
    )

    # S4 divergence/corruption decay. Name deliberately avoids overclaiming basis divergence.
    def s4(seed: int, delta: float) -> float:
        xtr, ttr, xte, tte = base[seed]
        Ftr = engineered_feats(xtr)
        Fte = engineered_feats(xte)
        rng = np.random.default_rng(int(delta * 1000) + 4)
        sc = delta * 3.0 * Ftr.std(0, keepdims=True)
        Xtr = Ftr + sc * rng.standard_normal(Ftr.shape)
        Xte = Fte + sc * rng.standard_normal(Fte.shape)
        return nrmse(tte, fit_predict(Xtr, ttr, Xte, cfg, seed))

    d0, d5, d10 = avg(lambda s: s4(s, 0.0), cfg), avg(lambda s: s4(s, 0.5), cfg), avg(lambda s: s4(s, 1.0), cfg)
    rep["S4"] = dict(name="corruption-decay", delta0=d0, delta0_5=d5, delta1=d10, passed=bool(d0 < d5 < d10))

    # S5 additive hidden-variable access proxy (target = latent z = g/c)
    def s5(seed: int) -> Tuple[float, float, float, float]:
        xtr, ttr, xte, tte = base[seed]
        ztr = (ttr[:, 2] / ttr[:, 3]).reshape(-1, 1)
        zte = (tte[:, 2] / tte[:, 3]).reshape(-1, 1)
        nat_tr, nat_te = native_feats(xtr), native_feats(xte)
        cp_tr, cp_te = coupling_feats(xtr), coupling_feats(xte)
        native = nrmse(zte, fit_predict(nat_tr, ztr, nat_te, cfg, seed))
        expanded = nrmse(
            zte,
            fit_predict(np.concatenate([nat_tr, cp_tr], axis=1), ztr, np.concatenate([nat_te, cp_te], axis=1), cfg, seed),
        )
        # F4 fix: distribution-preserving surrogate of the coupling channel.
        sh_tr = phase_randomized_surrogate(cp_tr, 5 + seed)
        sh_te = phase_randomized_surrogate(cp_te, 6 + seed)
        sham = nrmse(
            zte,
            fit_predict(np.concatenate([nat_tr, sh_tr], axis=1), ztr, np.concatenate([nat_te, sh_te], axis=1), cfg, seed),
        )
        # F4 fix: unique-contribution test. Fit native -> z, then test whether the
        # coupling channel predicts the NATIVE RESIDUAL. This separates "coupling
        # has no information" from "coupling info is redundant with native".
        nat_pred_tr = fit_predict(nat_tr, ztr, nat_tr, cfg, seed)
        nat_pred_te = fit_predict(nat_tr, ztr, nat_te, cfg, seed)
        resid_tr = ztr - nat_pred_tr
        resid_te = zte - nat_pred_te
        resid_nrmse = nrmse(resid_te, fit_predict(cp_tr, resid_tr, cp_te, cfg, seed))
        return native, expanded, sham, resid_nrmse

    nat, exp, sham, resid = np.mean([s5(int(s)) for s in cfg["seeds"]], axis=0)
    # Unique contribution: coupling explains residual variance below 1.0 (mean-only).
    unique_contribution = bool(resid + cfg["eff_floor"] <= 1.0)
    rep["S5"] = dict(
        name="additive-access-proxy",
        native_only=nat,
        expanded=exp,
        sham=sham,
        coupling_on_native_residual_nrmse=resid,
        coupling_has_unique_signal=unique_contribution,
        passed=bool(exp + cfg["eff_floor"] <= nat and exp + cfg["eff_floor"] <= sham),
    )

    # S6 shared rate-distortion placement
    def s6(seed: int) -> Tuple[float, float, float]:
        xtr, ttr, xte, tte = base[seed]
        direct = nrmse(tte, fit_predict(direct_feats(xtr), ttr, direct_feats(xte), cfg, seed))
        Etr, Ete = engineered_feats(xtr), engineered_feats(xte)
        trained = nrmse(tte, fit_predict(quantize(Etr, 6, Etr), ttr, quantize(Ete, 6, Etr), cfg, seed))
        Ntr, Nte = native_feats(xtr), native_feats(xte)
        symbolic = nrmse(tte, fit_predict(quantize(Ntr, 3, Ntr), ttr, quantize(Nte, 3, Ntr), cfg, seed))
        return direct, trained, symbolic

    dir_, trn, sym = np.mean([s6(int(s)) for s in cfg["seeds"]], axis=0)
    rep["S6"] = dict(name="shared-placement-proxy", direct=dir_, trained_proxy=trn, symbolic=sym, passed=bool(dir_ < trn < sym))

    # S7 dimensionality slope. Default: linear oscillator family by n_osc.
    # With a source: per-dimension instances of the same family type, so the
    # slope is measured within that family (the F5 fix for the linear failure).
    slope: Dict[str, Dict[str, Any]] = {}
    for n_osc, dlabel in [(2, "4D"), (4, "6D"), (6, "8D")]:
        dim_source = None
        if source is not None:
            dim_source = type(source)(n_osc=n_osc)

        def s7(seed: int, n_osc: int = n_osc, dim_source: Any = dim_source) -> Tuple[float, float]:
            xtr, ttr, xte, tte = make_dataset(n_osc, cfg, seed, source=dim_source)
            Etr, Ete = engineered_feats(xtr), engineered_feats(xte)
            trained = nrmse(tte, fit_predict(quantize(Etr, 8, Etr), ttr, quantize(Ete, 8, Etr), cfg, seed))
            ctrl = nrmse(tte, np.repeat(ttr.mean(0)[None], len(tte), 0))
            return trained, ctrl

        tr_, ct_ = np.mean([s7(int(s)) for s in cfg["seeds"]], axis=0)
        slope[dlabel] = dict(trained=tr_, control=ct_, above_ctrl=bool(tr_ + cfg["eff_floor"] < ct_))
    rep["S7"] = dict(
        name="linear-dimensionality-slope-proxy",
        slope=slope,
        passed=bool(slope["4D"]["trained"] < slope["8D"]["trained"] and slope["8D"]["above_ctrl"]),
        note="A fail here can be informative: linear systems may not become harder as dimensions are added.",
    )

    # S8 multi-codec audit: does inter-projection disagreement track true error?
    def s8(seed: int) -> float:
        xtr, ttr, xte, tte = base[seed]
        p1 = fit_predict(native_feats(xtr), ttr, native_feats(xte), cfg, seed)
        cftr, cfte = coupling_feats(xtr), coupling_feats(xte)
        if cftr.shape[1] <= 1:
            cftr, cfte = direct_feats(xtr), direct_feats(xte)
        p2 = fit_predict(cftr, ttr, cfte, cfg, seed + 1)
        disagree = np.linalg.norm(p1 - p2, axis=1)
        true_err = np.linalg.norm(0.5 * (p1 + p2) - tte, axis=1)
        return float(np.corrcoef(disagree, true_err)[0, 1])

    corr = avg(s8, cfg)
    rep["S8"] = dict(name="multi-codec-audit-proxy", disagreement_error_corr=corr, passed=bool(corr > 0.20))

    return _round(rep)


def format_ladder_table(rep: Dict[str, Any]) -> str:
    lines = ["| Stage | Name | Passed | Key metrics |", "|---|---|---:|---|"]
    for stage in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]:
        r = rep.get(stage, {})
        name = r.get("name", "")
        passed = "PASS" if r.get("passed") else "FAIL"
        details = {k: v for k, v in r.items() if k not in {"name", "passed"}}
        lines.append(f"| {stage} | {name} | {passed} | `{json.dumps(details)}` |")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run_ladder(CONFIG)
    print("\n=== PROGRESSIVE PROOF LADDER ===\n")
    print(format_ladder_table(rep))
