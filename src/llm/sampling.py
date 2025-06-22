from dataclasses import dataclass

from llama_cpp import StoppingCriteriaList


@dataclass
class SamplingParameters:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.0
    max_tokens: int = 16

    # stop: str | list[str] | None = field(default_factory=list)
    stopping_criteria: StoppingCriteriaList | None = None
    # n_sigma: float = 0.0
