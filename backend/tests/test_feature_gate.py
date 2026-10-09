"""Feature-distance gate maths (ood.py)."""
import numpy as np

from ood import FeatureGate


def _gate(threshold=20.0):
    means = np.array([[0.0, 0.0], [10.0, 10.0]])
    return FeatureGate(means, np.eye(2), threshold, {})


def test_score_is_min_squared_distance_to_any_class_mean():
    g = _gate()
    assert np.isclose(g.score(np.array([1.0, 0.0])), 1.0)       # near class 0
    assert np.isclose(g.score(np.array([9.0, 10.0])), 1.0)      # near class 1


def test_far_point_is_out_of_distribution():
    ok, score = _gate().is_in_distribution(np.array([100.0, -100.0]))
    assert not ok and score > 20


def test_near_point_is_in_distribution():
    ok, _ = _gate().is_in_distribution(np.array([1.0, 1.0]))
    assert ok


def test_precision_matrix_is_used():
    g = FeatureGate(np.zeros((1, 2)), np.diag([4.0, 1.0]), 1.0, {})
    assert np.isclose(g.score(np.array([1.0, 0.0])), 4.0)


def test_load_roundtrip_and_missing_file(tmp_path):
    p = tmp_path / "stats.npz"
    np.savez(p, means=np.zeros((2, 3)), precision=np.eye(3), threshold=5.5, percentile=99.0)
    g = FeatureGate.load_if_available(str(p))
    assert g.threshold == 5.5 and g.means.shape == (2, 3)
    assert FeatureGate.load_if_available(str(tmp_path / "nope.npz")) is None
