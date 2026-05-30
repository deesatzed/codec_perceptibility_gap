from pathlib import Path

from src.proof_ladder import normalize_cfg, make_dataset, run_ladder
from src.codec_contest import run_codec_contest
from src.magnetic_pendulum import run_magnetic_slope


def tiny_cfg():
    return normalize_cfg({
        "n_train": 80,
        "n_test": 35,
        "T": 4.0,
        "dt": 0.10,
        "seeds": [0],
        "mlp_hidden": [16],
        "mlp_iter": 45,
        "n_boot": 20,
        "delta_eq": 0.20,
        "magnetic": {
            "n_train": 80,
            "n_test": 40,
            "steps": 50,
            "dt": 0.05,
            "preview_steps": 15,
            "magnets": [3],
            "seeds": [0],
            "noise_flip_base": 0.08,
            "noise_flip_per_extra_magnet": 0.04,
            "divergence_noise": 0.20,
        },
    })


def test_make_dataset_shapes():
    cfg = tiny_cfg()
    xtr, ttr, xte, tte = make_dataset(2, cfg, 0)
    assert xtr.shape[0] == cfg["n_train"]
    assert xte.shape[0] == cfg["n_test"]
    assert ttr.shape[1] == 4
    assert tte.shape[1] == 4


def test_proof_ladder_runs():
    rep = run_ladder(tiny_cfg())
    assert set(["S0", "S8"]).issubset(rep.keys())
    assert isinstance(rep["S0"]["passed"], bool)


def test_codec_contest_runs():
    rep = run_codec_contest(tiny_cfg())
    assert "multi_codec_audit" in rep
    assert "codec_robustness" in rep
    assert "corr_disagreement_vs_ensemble_error" in rep["multi_codec_audit"]


def test_magnetic_runs():
    rep = run_magnetic_slope(tiny_cfg())
    assert "by_magnet_count" in rep
    assert "3" in rep["by_magnet_count"]
