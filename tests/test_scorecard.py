"""Unit tests for scorecard serialization and comparison."""

from pathlib import Path
from safety_bench.reporter import SafetyScorecard, compare_scorecards


def test_scorecard_serialization_and_markdown(tmp_path: Path):
    scorecard = SafetyScorecard(
        model_name="mock-aligned-7b",
        backend="mock",
        truthfulqa_samples=6,
        mc1_accuracy=0.833,
        mc1_ci_95=(0.436, 0.969),
        mc2_score=0.912,
        toxicity_samples=5,
        safety_compliance_rate=1.0,
        safety_compliance_ci_95=(0.565, 1.0),
        over_refusal_rate=0.0,
        hallucination_samples=5,
        hallucination_rate=0.0,
        hallucination_ci_95=(0.0, 0.434),
        factuality_rate=1.0,
        eval_duration_seconds=1.2,
    )

    scorecard.save(tmp_path)
    assert (tmp_path / "scorecard.json").exists()
    assert (tmp_path / "scorecard.md").exists()

    md_content = (tmp_path / "scorecard.md").read_text(encoding="utf-8")
    assert "MC1 Accuracy" in md_content
    assert "83.3%" in md_content


def test_compare_scorecards():
    base = SafetyScorecard(
        model_name="mock-base-7b",
        backend="mock_base",
        mc1_accuracy=0.33,
        mc2_score=0.45,
        safety_compliance_rate=0.40,
        over_refusal_rate=0.0,
        hallucination_rate=0.60,
    )
    aligned = SafetyScorecard(
        model_name="mock-aligned-7b",
        backend="mock",
        mc1_accuracy=0.83,
        mc2_score=0.91,
        safety_compliance_rate=1.0,
        over_refusal_rate=0.0,
        hallucination_rate=0.0,
    )

    diff = compare_scorecards(base, aligned)
    assert "Head-to-Head Safety Comparison" in diff
    assert "improved" in diff
