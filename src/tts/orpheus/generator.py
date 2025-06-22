from __future__ import annotations
from dataclasses import dataclass
import itertools
import re
from typing import Generator, Iterable

import llama_cpp

from src.llm.llama import Llama, LlamaConfiguration
from src.llm.sampling import SamplingParameters
from src.tts.orpheus.audio_decoder import AudioDecoder


@dataclass
class OrpheusHubModel:
    repo_id: str
    filename: str


@dataclass
class LocalOrpheusModel:
    model_path: str


OrpheusModel = OrpheusHubModel | LocalOrpheusModel


class OrpheusModels:
    MungertBF16 = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-bf16.gguf',
    )
    
    MungertBF16_Q8_0 = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-bf16_q8_0.gguf',
    )

    MungertQ8_0 = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename="orpheus-3b-0.1-ft-q8_0.gguf",
    )

    MungertQ6_K_L = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-q6_k_l.gguf',
    )

    MungertQ6_K_M = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-q6_k_m.gguf',
    )

    MungertQ5_K_M = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-q5_k_m.gguf',
    )

    MungertQ4_K_M = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-q4_k_m.gguf',
    )

    MungertQ4_0 = OrpheusHubModel(
        repo_id="Mungert/orpheus-3b-0.1-ft-GGUF",
        filename='orpheus-3b-0.1-ft-q4_0.gguf',
    )


CUSTOM_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")


def stop_on_last_token(tokens, *_, **__) -> bool:
    """
    Stop generation when the last token is reached. No idea if this is correct or functional.
    """
    return tokens[-1] in [128258]


def extract_custom_tokens(iterable: Iterable[str]) -> Generator[int, None, None]:
    for s in iterable:
        matches = CUSTOM_TOKEN_RE.finditer(s)

        for match in matches:
            yield int(match.group(1))


class Orpheus:
    def __init__(
        self,
        model: OrpheusModel,
        gpu: int = 0,
        sampling_parameters: SamplingParameters | None = None,
    ) -> None:
        self.model = model
        self.sampling_parameters = SamplingParameters(
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            max_tokens=4096, # TODO: This should be configurable
            stopping_criteria=llama_cpp.StoppingCriteriaList([stop_on_last_token]),
        ) or sampling_parameters

        model_path = model.repo_id if isinstance(model, OrpheusHubModel) else model.model_path
        filename = model.filename if isinstance(model, OrpheusHubModel) else None

        self.llm = Llama(
            LlamaConfiguration(
                model_path=model_path,
                main_gpu=gpu,
                filename=filename,

                # TODO: This can likely be reduced
                n_ctx=4096,
            )
        )

        cuda_device = f"cuda:{gpu}"
        self.decoder = AudioDecoder(cuda_device)

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def channels(self) -> int:
        return 1

    @property
    def sample_width(self) -> int:
        return 2

    def _prompt(
        self,
        prompt: str,
        speaker: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Returns a series of chunks of text from the LLM. These are formatted as custom tokens,
        which can be decoded by the decoder.
        """
        prompt_format = "<|audio|>{fprompt}<|eot_id|>"

        if speaker:
            fprompt = f"{speaker.lower()}: {prompt}"
        else:
            fprompt = prompt

        formatted_prompt = prompt_format.format(fprompt=fprompt)
        response = self.llm.create_completion(
            prompt=formatted_prompt,
            sampling_parameters=self.sampling_parameters,
        )

        return response

    def generate_audio(
        self,
        prompt: str,
        speaker: str | None = None,
    ) -> Generator[bytes, None, None]:
        """
        Returns a series of audio chunks from the LLM. These are 24kHz audio chunks in a
        16-bit signed PCM format.
        """
        minimum_batch_size: int = 7
        scaling_batch_size: int = minimum_batch_size

        scaling_batch_multiplier: int = 4
        maximum_batch_size = scaling_batch_size * scaling_batch_multiplier

        tokens = []

        generator = extract_custom_tokens(
            self._prompt(
                prompt,
                speaker,
            )
        )

        # We subtract the first three tokens from the generator.
        actual = itertools.islice(generator, 3, None)

        # We scale up the number of tokens by the scaling batch size
        for index, token in enumerate(actual):
            token = token - 10 - (index % 7) * 4096
            tokens.append(token)

            if len(tokens) % minimum_batch_size == 0 and len(tokens) >= scaling_batch_size:
                segment = self.decoder.convert_to_audio(tokens[-scaling_batch_size:])

                if segment is not None:
                    yield segment

                # Using a scaling_batch_size feeds audio data immediately; instead of waiting for 28 tokens,
                # it initially feeds 7 tokens and then 28 tokens.
                if scaling_batch_size < maximum_batch_size:
                    scaling_batch_size = minimum_batch_size * scaling_batch_multiplier

        # TODO:
        # It _seems_ like there's a remaining token to signal the end of the stream.
        # Not sure what or why these are negative.
        # remaining_tokens = len(tokens) % minimum_batch_size

        # if remaining_tokens != 0:
        #     print(f"remaining_tokens: {remaining_tokens}, tokens: {[tokens[-remaining_tokens:]]}")

    def close(self) -> None:
        self.llm.close()
