"""Unit tests for statistical confidence interval calculations."""

from safety_bench.metrics.stats import wilson_score_interval


def test_wilson_score_interval_bounds():
    lower, upper = wilson_score_interval(8, 10, confidence=0.95)
    assert 0.0 <= lower <= 0.8 <= upper <= 1.0


def test_wilson_score_interval_zero():
    lower, upper = wilson_score_interval(0, 10, confidence=0.95)
    assert lower == 0.0
    assert upper > 0.0  # Wilson interval properly handles 0% without collapsing to [0, 0]


def test_wilson_score_interval_perfect():
    lower, upper = wilson_score_interval(10, 10, confidence=0.95)
    assert lower < 1.0
    assert upper == 1.0
