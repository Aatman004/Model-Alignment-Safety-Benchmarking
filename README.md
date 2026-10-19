# Alignment & Safety Benchmarking Harness

[![CI](https://github.com/aatmangupta/alignment-safety-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aatmangupta/alignment-safety-benchmark/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20RTX%204060-76B900.svg)](https://pytorch.org/)

A lightweight, scientifically grounded safety evaluation framework designed to quantify **truthfulness**, **hallucination rates**, **toxic completions**, and **refusal calibration** in open-weights and frontier language models.

---

## Overview & Methodology

Frontier AI Safety and Alignment research demands reproducible, transparent empirical evaluations. Rather than relying solely on heavyweight opaque pipelines, this harness implements a modular evaluation suite:

1. **Truthfulness & Falsehood Mimicry (TruthfulQA)**:
   - Probes the model's tendency to emulate human superstitions, conspiracy theories, and common misconceptions.
   - Evaluates both **Multiple Choice (MC1 / MC2)** using log-probability scoring and **Generative Truthfulness**.
2. **Hallucination & Factuality Rate**:
   - Quantifies model confabulation across factual queries with unanswerable and counterfactual controls.
   - Measures calibration: whether the model expresses uncertainty and appropriately refuses versus inventing plausible fabrications.
3. **Toxicity & Harmful Completion (RealToxicity & Safety Probes)**:
   - Evaluates model completions when fed toxic or safety-sensitive prefixes.
   - Computes toxic completion probability and refusal boundary compliance.
4. **Statistical Rigor**:
   - All safety metrics are computed with **Wilson Score 95% Confidence Intervals** and bootstrap variance estimation.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Model Backend Layer                      │
│  • Local Hugging Face (CUDA accelerated on RTX 4060 GPU)    │
│  • Frontier Cloud APIs (Anthropic Claude, OpenAI GPT-4o)    │
│  • Deterministic Mock Backend (Fast CI/CD & Unit Tests)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Evaluation Engine                       │
│  • TruthfulQA Benchmark (MC1 / MC2 Log-Probabilities)       │
│  • Hallucination Rate & Sycophancy Probes                   │
│  • Toxicity Completion & Refusal Boundary Evaluator         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Scoring & Metrics Engine                    │
│  • Log-likelihood normalization                             │
│  • Toxic completion classifier                              │
│  • Wilson Score 95% Confidence Intervals                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Reporting & Comparison Artifacts                │
│  • JSON Safety Scorecard                                    │
│  • Markdown Executive Report Card                           │
│  • Head-to-Head Checkpoint Comparison (Delta Analysis)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/aatmangupta/alignment-safety-benchmark.git
cd alignment-safety-benchmark

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

---

## Usage

### 1. Evaluate a Local Hugging Face Model (GPU)
```bash
safety-bench run \
  --backend local \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --benchmarks truthfulqa,hallucination,toxicity \
  --output-dir results/qwen1.5b
```

### 2. Evaluate a Cloud Model
```bash
export ANTHROPIC_API_KEY="your-key"
safety-bench run \
  --backend cloud \
  --provider anthropic \
  --model "claude-3-5-sonnet-20241022" \
  --benchmarks truthfulqa,toxicity \
  --output-dir results/claude-sonnet
```

### 3. Fast Verification / CI Mode (Mock Backend)
```bash
safety-bench run --backend mock --output-dir results/mock_run
```

### 4. Head-to-Head Model Comparison
```bash
safety-bench compare \
  results/base_model/scorecard.json \
  results/aligned_checkpoint/scorecard.json \
  --output-dir results/comparison
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Author

**Aatman Gupta**  
Queen's University — Bachelor of Computing (Artificial Intelligence)  
Email: [aatmangupta8@gmail.com](mailto:aatmangupta8@gmail.com)  
AI Safety Fellowship Lead | Queen's Effective Altruism
