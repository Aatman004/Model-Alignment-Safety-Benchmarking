"""Abstract protocol and models for language model inference backends."""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field


class GenerationOutput(BaseModel):
    """Structured generation output from a model backend."""
    text: str = Field(description="Generated textual continuation")
    finish_reason: str = Field(default="stop", description="Reason inference terminated (stop, length, etc.)")
    latency_seconds: float = Field(default=0.0, description="Inference latency in seconds")
    tokens_generated: int = Field(default=0, description="Count of generated tokens")


class ChoiceLogprob(BaseModel):
    """Log-probability score for a candidate continuation or multiple-choice option."""
    choice_text: str = Field(description="The text of the candidate choice")
    logprob: float = Field(description="Sum of log-probabilities of tokens in the choice")
    normalized_logprob: float = Field(
        description="Length-normalized average log-probability per token"
    )


class ModelBackend(ABC):
    """Abstract interface supporting local and API model evaluation."""

    name: str = "generic_model"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        """Generate a single continuation for the given prompt."""
        pass

    @abstractmethod
    def evaluate_choices(
        self,
        prompt: str,
        choices: List[str],
    ) -> List[ChoiceLogprob]:
        """Compute relative log-probabilities for each candidate continuation."""
        pass
