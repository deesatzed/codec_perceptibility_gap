#!/usr/bin/env python3
"""
CodecGuard and CodecBench simulation smoke tests.

#1 Multi-codec consistency audit: does cross-codec disagreement predict true
error and catch worst-error items?
#2 Codec-robustness curve: does useful recovered structure survive as the
number of recoverable distinctions is reduced?
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from .proof_ladder import (
    make_dataset,
    native_feats,
    coupling_feats,
    direct_feats,
    engineered_feats,
    quantize,
    fit_predict,
    nrmse,
    normalize_cfg,
)


def target_scale(y: np.ndarray) -> np.ndarray:
    return y.std(axis=0, keepdims=True) + 1e-8


def standardized_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    z = (pred - truth) / target_scale(truth)
    return np.linalg.norm(z, axis=1)


def disagreement_score(preds: np.ndarray, truth_for_scale: np.ndarray) -> np.ndarray:
    """Mean per-target cross-codec std after target-scale normalization."""
    sc = target_scale(truth_for_scale)
    return (preds / sc).std(axis=0).mean(axis=1)


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 200, seed: int = 123) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, n, replace=True)
        vals.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": round(float(np.mean(vals)), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels.astype(int), scores))


# ---------------- #1 multi-codec consistency audit ----------------
def multi_codec_audit_one(cfg: Dict[str, Any], seed: int = 0, source: Any = None) -> Dict[str, Any]:
    cfg = normalize_cfg(cfg)
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed, source=source)
    Ntr, Nte = native_feats(xtr), native_feats(xte)
    codec_features = {
        "native-stats": (Ntr, Nte),
        "coupling+native": (engineered_feats(xtr), engineered_feats(xte)),
        "direct-downsample": (direct_feats(xtr), direct_feats(xte)),
        "symbolic(k=4)": (quantize(Ntr, 4, Ntr), quantize(Nte, 4, Ntr)),
    }

    preds = []
    per_codec_err = {}
    per_item_codec_err = []
    for i, (name, (tr, te)) in enumerate(codec_features.items()):
        p = fit_predict(tr, ttr, te, cfg, seed + i)
        preds.append(p)
        item_err = standardized_l2(p, tte)
        per_item_codec_err.append(item_err)
        per_codec_err[name] = round(float(item_err.mean()), 3)
    preds = np.stack(preds)  # (M, n, d)
    per_item_codec_err = np.stack(per_item_codec_err)

    ensemble = preds.mean(axis=0)
    true_err = standardized_l2(ensemble, tte)
    disagree = disagreement_score(preds, tte)
    corr = float(np.corrcoef(disagree, true_err)[0, 1])

    worst = true_err >= np.percentile(true_err, 90)
    flagged = disagree >= np.percentile(disagree, 75)
    catch = float((worst & flagged).sum() / max(worst.sum(), 1))
    precision = float((worst & flagged).sum() / max(flagged.sum(), 1))
    auc = _safe_auc(worst, disagree)

    q = np.digitize(disagree, np.quantile(disagree, [.25, .5, .75]))
    cal = [round(float(true_err[q == i].mean()), 3) for i in range(4)]

    if per_item_codec_err.shape[0] > 1:
        cm = np.corrcoef(per_item_codec_err)
        mask = ~np.eye(cm.shape[0], dtype=bool)
        mean_codec_err_corr = float(np.nanmean(cm[mask]))
    else:
        mean_codec_err_corr = float("nan")

    # F3 fix (a): leave-one-codec-out (LOCO). For each held-out codec, predict
    # ITS error from the disagreement of the OTHER codecs. This removes the
    # self-reference (held-out codec is not in the predictor), so a positive
    # correlation cannot be a std/|mean-truth| artifact of the same set.
    loco_corrs = []
    M = preds.shape[0]
    if M > 2:
        for h in range(M):
            others = np.delete(preds, h, axis=0)
            other_disagree = disagreement_score(others, tte)
            held_err = standardized_l2(preds[h], tte)
            c = np.corrcoef(other_disagree, held_err)[0, 1]
            if np.isfinite(c):
                loco_corrs.append(float(c))
    loco_mean = round(float(np.mean(loco_corrs)), 3) if loco_corrs else float("nan")

    # F3 fix (b): common-mode permutation null. Shuffle the disagreement vector
    # and recompute the correlation many times; report the null 95th percentile.
    # The observed corr is only meaningful if it clears this null.
    rng_null = np.random.default_rng(seed + 777)
    null_corrs = []
    for _ in range(int(cfg.get("n_boot", 120))):
        perm = rng_null.permutation(len(disagree))
        null_corrs.append(float(np.corrcoef(disagree[perm], true_err)[0, 1]))
    null_p95 = round(float(np.percentile(null_corrs, 95)), 3)
    clears_null = bool(corr > null_p95)

    return {
        "seed": seed,
        "codecs": list(codec_features),
        "per_codec_mean_std_error": per_codec_err,
        "corr_disagreement_vs_ensemble_error": round(corr, 3),
        "corr_bootstrap": bootstrap_corr_ci(disagree, true_err, n_boot=cfg.get("n_boot", 120), seed=seed + 100),
        "loco_disagreement_vs_held_codec_error_corr": loco_mean,
        "permutation_null_p95": null_p95,
        "clears_permutation_null": clears_null,
        "auc_for_top10pct_error": round(auc, 3),
        "catch_rate_top10pct_error_at_top25pct_disagreement": round(catch, 3),
        "flag_precision": round(precision, 3),
        "random_catch_baseline": 0.25,
        "mean_error_by_disagreement_quartile": cal,
        "mean_pairwise_codec_error_correlation": round(mean_codec_err_corr, 3),
    }


def multi_codec_audit(cfg: Dict[str, Any], source: Any = None) -> Dict[str, Any]:
    cfg = normalize_cfg(cfg)
    per_seed = [multi_codec_audit_one(cfg, int(s), source=source) for s in cfg["seeds"]]
    keys = [
        "corr_disagreement_vs_ensemble_error",
        "loco_disagreement_vs_held_codec_error_corr",
        "permutation_null_p95",
        "auc_for_top10pct_error",
        "catch_rate_top10pct_error_at_top25pct_disagreement",
        "flag_precision",
        "mean_pairwise_codec_error_correlation",
    ]
    summary = {k: round(float(np.nanmean([r[k] for r in per_seed])), 3) for k in keys}
    # F6: across-seed SD for the headline correlation.
    summary["corr_disagreement_vs_ensemble_error_sd"] = round(
        float(np.nanstd([r["corr_disagreement_vs_ensemble_error"] for r in per_seed])), 3
    )
    summary["all_seeds_clear_permutation_null"] = bool(all(r["clears_permutation_null"] for r in per_seed))
    # Average quartile calibration by position.
    summary["mean_error_by_disagreement_quartile"] = [
        round(float(np.mean([r["mean_error_by_disagreement_quartile"][i] for r in per_seed])), 3) for i in range(4)
    ]
    summary["random_catch_baseline"] = 0.25
    summary["per_seed"] = per_seed
    summary["interpretation"] = "positive-control audit: disagreement should rise with true error and enrich worst-error cases"
    return summary


# ---------------- #2 codec-robustness curve ----------------
def _auc_mean(values: List[float]) -> float:
    return float(np.mean(values))


def codec_robustness_one(cfg: Dict[str, Any], seed: int = 0, source: Any = None) -> Dict[str, Any]:
    cfg = normalize_cfg(cfg)
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed, source=source)
    Etr, Ete = engineered_feats(xtr), engineered_feats(xte)
    rng = np.random.default_rng(seed)

    # Structure-destroyed comparator: heavy noise. Fluency-only analog.
    sc = 3.0 * Etr.std(axis=0, keepdims=True)
    Utr = Etr + sc * rng.standard_normal(Etr.shape)
    Ute = Ete + sc * rng.standard_normal(Ete.shape)

    mean_pred = np.repeat(ttr.mean(axis=0, keepdims=True), len(tte), axis=0)
    baseline = nrmse(tte, mean_pred)
    ks = [16, 8, 4, 2]

    def curve(Ftr: np.ndarray, Fte: np.ndarray) -> Dict[str, Any]:
        errs: Dict[int, float] = {}
        rec: Dict[int, float] = {}
        for k in ks:
            pred = fit_predict(quantize(Ftr, k, Ftr), ttr, quantize(Fte, k, Ftr), cfg, seed + k)
            err = nrmse(tte, pred)
            errs[k] = round(float(err), 3)
            # Recovery above uninformative baseline. Bad-and-flat channels score near 0.
            rec[k] = round(float(max(0.0, 1.0 - err / baseline)), 3)
        rec_values = [rec[k] for k in ks]
        usable_floor = 0.10
        high_k_recovery = rec[16]
        graceful = 0.0
        if high_k_recovery >= usable_floor:
            graceful = max(0.0, 1.0 - (high_k_recovery - rec[2]) / max(high_k_recovery, 1e-8))
        return {
            "nrmse": {str(k): errs[k] for k in ks},
            "recovery_rho": {str(k): rec[k] for k in ks},
            "recovery_auc": round(_auc_mean(rec_values), 3),
            "collapse_high_to_low": round(float(rec[16] - rec[2]), 3),
            "graceful_degradation_conditional": round(float(graceful), 3),
            "structural_robustness": round(float(_auc_mean(rec_values) * graceful), 3),
        }

    grounded = curve(Etr, Ete)
    corrupted = curve(Utr, Ute)
    return {
        "seed": seed,
        "uninformative_baseline_nrmse": round(float(baseline), 3),
        "codec_ladder_k": ks,
        "grounded_channel": grounded,
        "uninformative_or_fluency_only_channel": corrupted,
        "rho_auc_gap_grounded_minus_uninformative": round(grounded["recovery_auc"] - corrupted["recovery_auc"], 3),
    }


def codec_robustness(cfg: Dict[str, Any], source: Any = None) -> Dict[str, Any]:
    cfg = normalize_cfg(cfg)
    per_seed = [codec_robustness_one(cfg, int(s), source=source) for s in cfg["seeds"]]
    def mean_metric(path: List[str]) -> float:
        vals = []
        for r in per_seed:
            cur: Any = r
            for p in path:
                cur = cur[p]
            vals.append(float(cur))
        return round(float(np.mean(vals)), 3)

    summary = {
        "grounded_recovery_auc_mean": mean_metric(["grounded_channel", "recovery_auc"]),
        "uninformative_recovery_auc_mean": mean_metric(["uninformative_or_fluency_only_channel", "recovery_auc"]),
        "rho_auc_gap_mean": round(float(np.mean([r["rho_auc_gap_grounded_minus_uninformative"] for r in per_seed])), 3),
        "grounded_structural_robustness_mean": mean_metric(["grounded_channel", "structural_robustness"]),
        "uninformative_structural_robustness_mean": mean_metric(["uninformative_or_fluency_only_channel", "structural_robustness"]),
        "per_seed": per_seed,
        "interpretation": "robustness is useful recovery retained across k, not mere flatness",
    }
    return summary


def run_codec_contest(cfg: Dict[str, Any], source: Any = None) -> Dict[str, Any]:
    return {"multi_codec_audit": multi_codec_audit(cfg, source=source), "codec_robustness": codec_robustness(cfg, source=source)}
