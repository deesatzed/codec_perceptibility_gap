import numpy as np
from src.crc.embedder import Embedder


def test_embedder_semantic_spread_real():
    e = Embedder()
    v = e.encode(["Paris", "The capital is Paris.", "Lyon"])
    assert v.shape[0] == 3 and v.ndim == 2

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # same meaning closer than different meaning -> the metric is real
    assert cos(v[0], v[1]) > cos(v[0], v[2])
