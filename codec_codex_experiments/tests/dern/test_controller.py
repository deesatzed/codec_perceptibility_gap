import inspect
import numpy as np
from src.dern.controller import OnlineController


def test_controller_update_takes_only_verdict_and_cost():
    sig = set(inspect.signature(OnlineController.update).parameters)
    assert "reward" in sig or "cost" in sig
    assert "self_score" not in sig


def test_controller_converges_to_low_cost_action_under_reward():
    rng = np.random.default_rng(0)
    c = OnlineController(actions=["cheap", "mid", "full"], seed=0, epsilon=0.1)
    rewards = {"cheap": -1.0, "mid": -2.5, "full": -5.0}
    for _ in range(500):
        a = c.choose()
        c.update(a, reward=rewards[a] + 0.05 * rng.standard_normal())
    assert c.greedy() == "cheap"


def test_controller_reward_not_derived_from_own_proposal():
    c1 = OnlineController(actions=["a", "b"], seed=1, epsilon=0.0)
    c2 = OnlineController(actions=["a", "b"], seed=1, epsilon=0.0)
    for _ in range(50):
        c1.update("a", reward=1.0)
        c2.update("a", reward=-1.0)
    assert c1.value("a") != c2.value("a")
