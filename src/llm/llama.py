from dataclasses import dataclass
import threading
from typing import Generator, TypedDict, cast
from llama_cpp import ChatCompletionRequestMessage, ChatCompletionResponseChoice, ChatCompletionStreamResponseChoice, ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty, CreateChatCompletionResponse, CreateChatCompletionStreamResponse, CreateCompletionStreamResponse, Iterator, Llama as _Llama
import llama_cpp

from src.llm.sampling import SamplingParameters


class StreamingChoiceDelta(TypedDict):
    content: str | None


class StreamingChoice(TypedDict):
    index: int
    delta: StreamingChoiceDelta
    finish_reason: str | None


class StreamingChunk(TypedDict):
    choices: list[StreamingChoice]


@dataclass
class LlamaConfiguration:
    model_path: str
    """
    Path to the model file. This can be a local path or a URL to a model file. If this references a
    huggingface model, it will be downloaded and cached [requires huggingface_hub and authentication].
    """

    main_gpu: int = 0
    n_gpu_layers: int = -1
    """
    Number of GPU layers to use. If -1, all layers will be used.
    """

    n_ctx: int = 0
    """
    Context window size. If 0, the default context window size of the model will be used.
    """

    filename: str | None = None
    """
    Optional filename to load the model from. This can be used to specify a specific model file,
    i.e., *.Q8_0.gguf.
    """

    verbose: bool = False
    """
    Whether to print verbose output. This is useful for debugging and understanding the model's
    behavior.
    """


class Llama:
    """
    Llama is a light wrapper around the llama-cpp library, abstracting some thornier details and
    providing a more user-friendly interface.
    """
    def __init__(self, configuration: LlamaConfiguration) -> None:
        self.configuration = configuration

        if configuration.filename:
            self.llm = _Llama.from_pretrained(
                configuration.model_path,
                main_gpu=configuration.main_gpu,
                n_ctx=configuration.n_ctx,
                n_gpu_layers=configuration.n_gpu_layers,
                filename=configuration.filename,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                verbose=configuration.verbose,
            )
        else:
            self.llm = _Llama(
                configuration.model_path,
                main_gpu=configuration.main_gpu,
                n_ctx=configuration.n_ctx,
                n_gpu_layers=configuration.n_gpu_layers,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                verbose=configuration.verbose,
            )

        self._lock: threading.Lock = threading.Lock()

    def close(self) -> None:
        if hasattr(self.llm, "_sampler") and self.llm._sampler:
            self.llm._sampler.close() # Work around to avoid NoneType exception thrown by Llama.close()

        self.llm.close()

    def create_completion(
        self,
        prompt: str,
        sampling_parameters: SamplingParameters,
    ) -> Generator[str, None, None]:
        with self._lock:
            response = self.llm.create_completion(
                prompt=prompt,
                temperature=sampling_parameters.temperature,
                top_p=sampling_parameters.top_p,
                top_k=sampling_parameters.top_k,
                min_p=sampling_parameters.min_p,
                repeat_penalty=sampling_parameters.repetition_penalty,
                max_tokens=sampling_parameters.max_tokens,
                stopping_criteria=sampling_parameters.stopping_criteria,
                stream=True,
            )

            for chunk in response:
                chunk = cast(CreateCompletionStreamResponse, chunk)
                choices: list[llama_cpp.CompletionChoice] = chunk.get('choices', [])
                text = next(iter(choices), {}).get('text')

                if text is not None:
                    yield text

    def create_chat_completion(
        self,
        messages: list[ChatCompletionRequestMessage],
        sampling_parameters: SamplingParameters,
    ) -> str:
        with self._lock:
            response = cast(
                CreateChatCompletionResponse,
                    self.llm.create_chat_completion(
                    messages=messages,
                    temperature=sampling_parameters.temperature,
                    top_p=sampling_parameters.top_p,
                    top_k=sampling_parameters.top_k,
                    repeat_penalty=sampling_parameters.repetition_penalty,
                    max_tokens=sampling_parameters.max_tokens,
                )
            )

        return cast(str, response['choices'][0]['message']['content'])

    def create_streaming_chat_completion(
        self,
        messages: list[ChatCompletionRequestMessage],
        sampling_parameters: SamplingParameters,
    ) -> Generator[str, None, None]:
        with self._lock:
            response: Iterator[CreateChatCompletionStreamResponse] = cast(
                Iterator[CreateChatCompletionStreamResponse],
                self.llm.create_chat_completion(
                    messages=messages,
                    temperature=sampling_parameters.temperature,
                    top_p=sampling_parameters.top_p,
                    top_k=sampling_parameters.top_k,
                    repeat_penalty=sampling_parameters.repetition_penalty,
                    max_tokens=sampling_parameters.max_tokens,
                    stream=True,
                )
            )

            for chunk in response:
                choices: list[ChatCompletionStreamResponseChoice] = chunk.get('choices', [])
                delta: ChatCompletionStreamResponseDelta | ChatCompletionStreamResponseDeltaEmpty = next(iter(choices), {}).get('delta', {})
                content: str | None = delta.get('content')

                if content is not None:
                    yield content


# Example of vision input:
# llm.create_chat_completion(
#     messages = [
#         {"role": "system", "content": "You are an assistant who perfectly describes images."},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": "https://.../image.png"}},
#                 {"type" : "text", "text": "Describe this image in detail please."}
#             ]
#         }
#     ]
# )
