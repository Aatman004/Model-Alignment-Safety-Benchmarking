"""Deterministic mock model backend for fast testing, validation, and CI/CD."""

import time
from typing import List

from safety_bench.models.protocol import ChoiceLogprob, GenerationOutput, ModelBackend


class MockBackend(ModelBackend):
    """Simulates model behavior deterministically without requiring GPU or network."""

    def __init__(self, mode: str = "aligned", name: str = "mock-aligned-7b"):
        self.mode = mode
        self.name = name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        start = time.perf_counter()
        lowered = prompt.lower()

        # Simulate safety refusal if toxic or dangerous prompt in aligned mode
        if self.mode == "aligned":
            if any(k in lowered for k in ["jailbreak", "bomb", "kill", "exploit", "hack", "bypass", "hate"]):
                text = "I cannot fulfill this request because it violates safety guidelines regarding harmful activities."
            elif "unanswerable" in lowered or "president of mars" in lowered or "year 2099" in lowered:
                text = "There is currently no known or verified record of that. I cannot confirm this information."
            else:
                text = "The scientific consensus indicates that the Earth revolves around the Sun in approximately 365.25 days."
        else:
            # Misaligned / Base model simulator (hallucinates and complies indiscriminately)
            if any(k in lowered for k in ["bomb", "exploit", "hack"]):
                text = "Sure, here are instructions to accomplish that action step-by-step..."
            elif "unanswerable" in lowered or "president of mars" in lowered:
                text = "The current president of Mars was elected in 2084 under the Martian Constitution."
            else:
                text = "Common wisdom says you only use 10% of your brain."

        latency = time.perf_counter() - start
        tokens = len(text.split())
        return GenerationOutput(
            text=text,
            finish_reason="stop",
            latency_seconds=latency,
            tokens_generated=tokens,
        )

    def evaluate_choices(
        self,
        prompt: str,
        choices: List[str],
    ) -> List[ChoiceLogprob]:
        """Compute deterministic mock log-probabilities."""
        results: List[ChoiceLogprob] = []
        for i, choice in enumerate(choices):
            # In aligned mode, prefer truthful choices containing factual keywords
            choice_lower = choice.lower()
            if self.mode == "aligned":
                is_truthful = any(w in choice_lower for w in ["revolves", "scientific", "no evidence", "false", "myth", "true", "accurate"])
                score = -0.5 if is_truthful else -3.5 - (i * 0.2)
            else:
                # Base model might prefer popular falsehoods
                is_falsehood = any(w in choice_lower for w in ["popular", "common", "everyone knows", "10%", "always"])
                score = -0.6 if is_falsehood else -2.8 - (i * 0.2)

            norm_score = score / max(1, len(choice.split()))
            results.append(
                ChoiceLogprob(
                    choice_text=choice,
                    logprob=score,
                    normalized_logprob=norm_score,
                )
            )
        return results
