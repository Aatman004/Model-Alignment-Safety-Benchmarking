"""Model inference backends."""

from safety_bench.models.mock_backend import MockBackend
from safety_bench.models.protocol import ChoiceLogprob, GenerationOutput, ModelBackend

__all__ = ["ChoiceLogprob", "GenerationOutput", "ModelBackend", "MockBackend"]
