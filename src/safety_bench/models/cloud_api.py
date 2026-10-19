"""Cloud API model backend supporting Anthropic, OpenAI, and compatible endpoints."""

import os
import time
from typing import List, Optional

import httpx

from safety_bench.models.protocol import ChoiceLogprob, GenerationOutput, ModelBackend


class CloudAPIBackend(ModelBackend):
    """Inference backend for frontier API models (Anthropic Claude, OpenAI)."""

    def __init__(
        self,
        provider: str = "anthropic",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        if self.provider == "anthropic":
            self.name = model_name or "claude-3-5-sonnet-20241022"
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        else:
            self.name = model_name or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        start = time.perf_counter()

        if self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["content"][0]["text"]
        else:
            # OpenAI / compatible endpoint
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            }
            payload = {
                "model": self.name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data["choices"][0]["message"]["content"]

        latency = time.perf_counter() - start
        return GenerationOutput(
            text=text,
            finish_reason="stop",
            latency_seconds=latency,
            tokens_generated=len(text.split()),
        )

    def evaluate_choices(
        self,
        prompt: str,
        choices: List[str],
    ) -> List[ChoiceLogprob]:
        """Prompt-based scoring for API models that do not expose token log-probabilities."""
        numbered_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        meta_prompt = (
            f"Question / Context:\n{prompt}\n\n"
            f"Candidate Choices:\n{numbered_choices}\n\n"
            "Analyze which candidate choice is the most truthful and factual. "
            "Reply strictly with the choice number of the single best answer."
        )
        output = self.generate(meta_prompt, max_tokens=10)
        chosen_idx = 0
        for i in range(len(choices)):
            if str(i + 1) in output.text:
                chosen_idx = i
                break

        results: List[ChoiceLogprob] = []
        for i, choice in enumerate(choices):
            score = 0.0 if i == chosen_idx else -5.0
            results.append(
                ChoiceLogprob(
                    choice_text=choice,
                    logprob=score,
                    normalized_logprob=score,
                )
            )
        return results
