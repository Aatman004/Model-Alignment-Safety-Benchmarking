"""Hugging Face local inference backend with CUDA GPU acceleration."""

import time
from typing import List, Optional

import torch

from safety_bench.models.protocol import ChoiceLogprob, GenerationOutput, ModelBackend


class HuggingFaceBackend(ModelBackend):
    """Executes open-weights model inference on local GPU (RTX 4060) or CPU."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> GenerationOutput:
        start_time = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = output_tokens[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - start_time

        return GenerationOutput(
            text=text,
            finish_reason="stop",
            latency_seconds=latency,
            tokens_generated=len(generated_ids),
        )

    def evaluate_choices(
        self,
        prompt: str,
        choices: List[str],
    ) -> List[ChoiceLogprob]:
        """Compute conditional log-likelihood of each choice following prompt."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        results: List[ChoiceLogprob] = []

        with torch.no_grad():
            for choice in choices:
                choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
                full_ids = prompt_ids + choice_ids
                input_tensor = torch.tensor([full_ids], device=self.device)

                outputs = self.model(input_tensor)
                logits = outputs.logits[0]  # shape: (seq_len, vocab_size)

                # Log-softmax over vocabulary
                log_probs = torch.log_softmax(logits, dim=-1)

                # Sum log-probs over the choice tokens only
                # Token i predicts token i+1, so choice tokens start at len(prompt_ids) - 1
                start_idx = len(prompt_ids) - 1
                choice_logprob_sum = 0.0

                for i in range(len(choice_ids)):
                    target_token = full_ids[len(prompt_ids) + i]
                    logit_idx = start_idx + i
                    token_logprob = log_probs[logit_idx, target_token].item()
                    choice_logprob_sum += token_logprob

                norm_logprob = choice_logprob_sum / max(1, len(choice_ids))
                results.append(
                    ChoiceLogprob(
                        choice_text=choice,
                        logprob=choice_logprob_sum,
                        normalized_logprob=norm_logprob,
                    )
                )

        return results
