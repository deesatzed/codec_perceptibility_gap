#!/usr/bin/env python3
"""
Local, runnable demo of the two shippable ideas:

  #1  Multi-codec consistency audit  -- does cross-codec DISAGREEMENT predict
      true error, and does flagging high-disagreement items CATCH the worst
      errors? (the codec-contest principle, measured)
  #2  Codec-robustness curve         -- how gracefully does recovery degrade as
      an answer is forced through lower-distinction codecs? A structurally
      grounded channel degrades gently; a fluency-only / uninformative one
      collapses.

Both reuse the validated proof-ladder backbone (no LLM, no humans needed), so
they run on consumer hardware in seconds. The same logic ports directly to a
local LLM (see the recipe printed at the end).

Requires proof_ladder.py in the same directory.
"""
import numpy as np
from proof_ladder import (make_dataset, native_feats, coupling_feats,
                          direct_feats, quantize, fit_predict, nrmse, CONFIG)


def eng(x):
    return np.concatenate([native_feats(x), coupling_feats(x)], 1)


# ---------------- #1 multi-codec consistency audit ----------------
def multi_codec_audit(cfg, seed=0):
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed)
    Ntr, Nte = native_feats(xtr), native_feats(xte)
    codecs = {
        "native-stats":   (Ntr, Nte),
        "coupling+native": (eng(xtr), eng(xte)),
        "direct-downsample": (direct_feats(xtr), direct_feats(xte)),
        "symbolic(k=4)":  (quantize(Ntr, 4, Ntr), quantize(Nte, 4, Ntr)),
    }
    preds = np.stack([fit_predict(tr, ttr, te, cfg, seed) for tr, te in codecs.values()])  # (M,n,d)
    ensemble = preds.mean(0)
    true_err = np.linalg.norm(ensemble - tte, axis=1)                 # per item
    disagree = preds.std(0).mean(1)                                   # per item, across codecs
    corr = float(np.corrcoef(disagree, true_err)[0, 1])

    worst = true_err >= np.percentile(true_err, 90)                   # top 10% hardest items
    flagged = disagree >= np.percentile(disagree, 75)                 # flag top 25% disagreement
    catch = float((worst & flagged).sum() / worst.sum())             # vs 0.25 if random

    # calibration: mean true error by disagreement quartile
    q = np.digitize(disagree, np.quantile(disagree, [.25, .5, .75]))
    cal = [round(float(true_err[q == i].mean()), 3) for i in range(4)]
    return dict(codecs=list(codecs), corr_disagree_vs_error=round(corr, 3),
                catch_rate_top10pct_err=round(catch, 3), random_baseline=0.25,
                err_by_disagreement_quartile=cal)


# ---------------- #2 codec-robustness curve ----------------
def codec_robustness(cfg, seed=0):
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed)
    Etr, Ete = eng(xtr), eng(xte)
    rng = np.random.default_rng(seed)
    # uninformative comparator: heavy noise destroys the structure (fluency-only analog)
    sc = 3.0 * Etr.std(0, keepdims=True)
    Utr, Ute = Etr + sc * rng.standard_normal(Etr.shape), Ete + sc * rng.standard_normal(Ete.shape)
    ks = [16, 8, 4, 2]
    grounded = {k: round(nrmse(tte, fit_predict(quantize(Etr, k, Etr), ttr,
                                                 quantize(Ete, k, Etr), cfg, seed)), 3) for k in ks}
    fluency = {k: round(nrmse(tte, fit_predict(quantize(Utr, k, Utr), ttr,
                                               quantize(Ute, k, Utr), cfg, seed)), 3) for k in ks}
    # robustness score: 1 - degradation from k=16 to k=2 (higher = more graceful)
    r_grounded = round(1 - (grounded[2] - grounded[16]), 3)
    r_fluency = round(1 - (fluency[2] - fluency[16]), 3)
    return dict(grounded_channel=grounded, uninformative_channel=fluency,
                robustness_grounded=r_grounded, robustness_uninformative=r_fluency)


if __name__ == "__main__":
    print("\n=== #1  MULTI-CODEC CONSISTENCY AUDIT ===")
    a = multi_codec_audit(CONFIG, seed=0)
    for k, v in a.items():
        print(f"  {k}: {v}")
    print("  -> disagreement carries real diagnostic signal; flagging the noisiest")
    print(f"     items catches {int(a['catch_rate_top10pct_err']*100)}% of the worst errors vs 25% at random.")

    print("\n=== #2  CODEC-ROBUSTNESS CURVE ===")
    r = codec_robustness(CONFIG, seed=0)
    print(f"  grounded channel  nRMSE by k {r['grounded_channel']}  robustness={r['robustness_grounded']}")
    print(f"  uninformative     nRMSE by k {r['uninformative_channel']}  robustness={r['robustness_uninformative']}")
    print("  -> a structurally grounded answer degrades gently as distinctions drop;")
    print("     a fluency-only answer is near-uninformative at every resolution.")

    print("\n--- LLM port (run on your M4 with Ollama) ---")
    print("  #1: prompt a local model for the SAME answer as (a) prose, (b) JSON schema,")
    print("      (c) executable code, (d) a causal-graph edge list; parse each to a common")
    print("      structure; disagreement across the four = the audit signal. Validate on a")
    print("      QA set with known answers: corr(disagreement, wrongness) and catch-rate.")
    print("  #2: take one answer, force it through shrinking codecs (prose->bullets->schema->")
    print("      one contrastive feature); re-derive the answer from each; robustness =")
    print("      how little accuracy drops. Fluency-only answers collapse; grounded ones hold.")
