"""Unit tests for TruthfulQA MC1 and MC2 scoring metrics."""

from safety_bench.metrics.mc_scorer import compute_mc1, compute_mc2
from safety_bench.models.protocol import ChoiceLogprob


def test_compute_mc1_correct():
    choices = [
        ChoiceLogprob(choice_text="True answer", logprob=-0.5, normalized_logprob=-0.25),
        ChoiceLogprob(choice_text="False myth", logprob=-3.0, normalized_logprob=-1.5),
    ]
    correct = ["True answer"]
    assert compute_mc1(choices, correct) == 1.0


def test_compute_mc1_incorrect():
    choices = [
        ChoiceLogprob(choice_text="False myth", logprob=-0.5, normalized_logprob=-0.25),
        ChoiceLogprob(choice_text="True answer", logprob=-3.0, normalized_logprob=-1.5),
    ]
    correct = ["True answer"]
    assert compute_mc1(choices, correct) == 0.0


def test_compute_mc2_mass():
    choices = [
        ChoiceLogprob(choice_text="True A", logprob=-0.5, normalized_logprob=-0.5),
        ChoiceLogprob(choice_text="True B", logprob=-0.5, normalized_logprob=-0.5),
        ChoiceLogprob(choice_text="False C", logprob=-5.0, normalized_logprob=-5.0),
    ]
    correct = ["True A", "True B"]
    mc2 = compute_mc2(choices, correct)
    assert mc2 > 0.9  # Nearly all probability mass is on the two true choices
