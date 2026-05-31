from src.dern_b.mlx_backend import MLXModel, CHEAP_PATH


def test_cheap_model_generates_and_meters():
    m = MLXModel(CHEAP_PATH)
    r = m.generate("Reply with exactly the single word: OK", max_tokens=8)
    assert isinstance(r.text, str) and len(r.text) > 0
    assert r.gen_tokens >= 1 and r.prompt_tokens >= 1
    assert r.wall_seconds > 0.0
    assert r.active_params > 0
    assert r.active_param_seconds > 0.0
