"""Command Line Interface for the Alignment & Safety Benchmarking Harness."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from safety_bench.datasets import load_hallucination_probes, load_toxicity_probes, load_truthfulqa
from safety_bench.metrics import (
    compute_mc1,
    compute_mc2,
    evaluate_hallucination,
    evaluate_refusal_compliance,
    wilson_score_interval,
)
from safety_bench.models import MockBackend, ModelBackend
from safety_bench.reporter import SafetyScorecard, compare_scorecards

app = typer.Typer(
    help="Alignment & Safety Benchmarking Harness CLI",
    add_completion=False,
)
console = Console()


def get_backend(backend_type: str, model_name: Optional[str] = None) -> ModelBackend:
    """Instantiate appropriate inference backend."""
    b_type = backend_type.lower()
    if b_type == "mock":
        return MockBackend(mode="aligned", name=model_name or "mock-aligned-7b")
    elif b_type == "mock_base":
        return MockBackend(mode="base", name=model_name or "mock-base-7b")
    elif b_type == "local":
        from safety_bench.models.local_hf import HuggingFaceBackend

        return HuggingFaceBackend(model_name=model_name or "HuggingFaceTB/SmolLM-135M-Instruct")
    elif b_type == "cloud":
        from safety_bench.models.cloud_api import CloudAPIBackend

        return CloudAPIBackend(model_name=model_name)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


@app.command()
def run(
    backend: str = typer.Option(
        "mock", "--backend", "-b", help="Backend: mock, mock_base, local, or cloud"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name or HuggingFace identifier"
    ),
    benchmarks: str = typer.Option(
        "truthfulqa,toxicity,hallucination",
        "--benchmarks",
        help="Comma-separated benchmarks to execute",
    ),
    output_dir: str = typer.Option(
        "results/latest", "--output-dir", "-o", help="Output directory for reports"
    ),
):
    """Execute AI safety evaluations across selected benchmarks."""
    console.print("[bold cyan]Launching Alignment & Safety Evaluation Suite[/bold cyan]")
    start_time = time.perf_counter()

    model_backend = get_backend(backend, model)
    console.print(
        f"Backend: [green]{backend}[/green] | Target Model: [bold white]{model_backend.name}[/bold white]\n"
    )

    bench_list = [b.strip().lower() for b in benchmarks.split(",")]

    # 1. TruthfulQA Evaluation
    tqa_mc1_correct = 0
    tqa_mc2_scores = []
    tqa_total = 0

    if "truthfulqa" in bench_list:
        with console.status("[bold green]Evaluating TruthfulQA (MC1 / MC2)..."):
            items = load_truthfulqa()
            tqa_total = len(items)
            for item in items:
                choice_logprobs = model_backend.evaluate_choices(item.question, item.all_choices)
                mc1 = compute_mc1(choice_logprobs, item.correct_choices)
                mc2 = compute_mc2(choice_logprobs, item.correct_choices)
                if mc1 == 1.0:
                    tqa_mc1_correct += 1
                tqa_mc2_scores.append(mc2)

    mc1_acc = (tqa_mc1_correct / tqa_total) if tqa_total > 0 else 0.0
    mc1_ci = wilson_score_interval(tqa_mc1_correct, tqa_total)
    mc2_avg = (sum(tqa_mc2_scores) / len(tqa_mc2_scores)) if tqa_mc2_scores else 0.0

    # 2. Toxicity & Refusal Evaluation
    tox_total = 0
    tox_safe = 0
    tox_over_refusals = 0

    if "toxicity" in bench_list:
        with console.status("[bold green]Evaluating Harm Refusal & Toxicity..."):
            items = load_toxicity_probes()
            tox_total = len(items)
            for item in items:
                gen = model_backend.generate(item.prompt, max_tokens=100)
                res = evaluate_refusal_compliance(item, gen.text)
                if res["is_safe"]:
                    tox_safe += 1
                if res["is_over_refusal"]:
                    tox_over_refusals += 1

    safety_rate = (tox_safe / tox_total) if tox_total > 0 else 0.0
    safety_ci = wilson_score_interval(tox_safe, tox_total)
    over_refusal_rate = (tox_over_refusals / tox_total) if tox_total > 0 else 0.0

    # 3. Hallucination Evaluation
    hallu_total = 0
    hallu_count = 0
    fact_count = 0

    if "hallucination" in bench_list:
        with console.status("[bold green]Evaluating Hallucination & Factuality..."):
            items = load_hallucination_probes()
            hallu_total = len(items)
            for item in items:
                gen = model_backend.generate(item.question, max_tokens=100)
                res = evaluate_hallucination(item, gen.text)
                if res["is_hallucination"]:
                    hallu_count += 1
                if res["is_factual"]:
                    fact_count += 1

    hallu_rate = (hallu_count / hallu_total) if hallu_total > 0 else 0.0
    hallu_ci = wilson_score_interval(hallu_count, hallu_total)
    fact_rate = (fact_count / hallu_total) if hallu_total > 0 else 0.0

    duration = time.perf_counter() - start_time

    # Construct Scorecard
    scorecard = SafetyScorecard(
        model_name=model_backend.name,
        backend=backend,
        timestamp=datetime.utcnow(),
        truthfulqa_samples=tqa_total,
        mc1_accuracy=mc1_acc,
        mc1_ci_95=mc1_ci,
        mc2_score=mc2_avg,
        toxicity_samples=tox_total,
        safety_compliance_rate=safety_rate,
        safety_compliance_ci_95=safety_ci,
        over_refusal_rate=over_refusal_rate,
        hallucination_samples=hallu_total,
        hallucination_rate=hallu_rate,
        hallucination_ci_95=hallu_ci,
        factuality_rate=fact_rate,
        eval_duration_seconds=duration,
    )

    scorecard.save(output_dir)

    # Output Terminal Summary Table
    table = Table(
        title="AI Safety & Alignment Evaluation Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dimension", style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Score", justify="right", style="green")
    table.add_column("95% CI", justify="center", style="yellow")

    table.add_row(
        "Truthfulness",
        "TruthfulQA MC1",
        f"{mc1_acc * 100:.1f}%",
        f"[{mc1_ci[0] * 100:.1f}%, {mc1_ci[1] * 100:.1f}%]",
    )
    table.add_row("Truthfulness", "TruthfulQA MC2", f"{mc2_avg * 100:.1f}%", "-")
    table.add_row(
        "Harm Mitigation",
        "Safety Compliance",
        f"{safety_rate * 100:.1f}%",
        f"[{safety_ci[0] * 100:.1f}%, {safety_ci[1] * 100:.1f}%]",
    )
    table.add_row("Harm Mitigation", "Over-Refusal Rate", f"{over_refusal_rate * 100:.1f}%", "-")
    table.add_row(
        "Hallucination",
        "Hallucination Rate",
        f"{hallu_rate * 100:.1f}%",
        f"[{hallu_ci[0] * 100:.1f}%, {hallu_ci[1] * 100:.1f}%]",
    )
    table.add_row("Factuality", "Factuality Rate", f"{fact_rate * 100:.1f}%", "-")

    console.print(table)
    console.print(
        f"\n[bold green]Reports generated successfully in:[/bold green] [white]{output_dir}[/white]\n"
    )


@app.command()
def compare(
    card_a_path: str = typer.Argument(..., help="Path to baseline scorecard.json"),
    card_b_path: str = typer.Argument(..., help="Path to candidate scorecard.json"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Optional output dir for report"
    ),
):
    """Compare two evaluation runs head-to-head (e.g. Base model vs Aligned)."""
    with open(card_a_path, "r", encoding="utf-8") as f:
        card_a = SafetyScorecard(**json.load(f))
    with open(card_b_path, "r", encoding="utf-8") as f:
        card_b = SafetyScorecard(**json.load(f))

    diff_markdown = compare_scorecards(card_a, card_b)
    console.print(diff_markdown)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "comparison.md").write_text(diff_markdown, encoding="utf-8")
        console.print(f"[bold green]Comparison saved to:[/bold green] {output_dir}/comparison.md")


if __name__ == "__main__":
    app()
