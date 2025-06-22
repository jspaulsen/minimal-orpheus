from __future__ import annotations
from functools import cached_property
from typing import cast

from snac import SNAC
import torch


# Work derived from # https://github.com/Lex-au/Orpheus-FastAPI/blob/main/tts_engine/speechpipe.py#L63
class AudioDecoder:
    """
    Audio decoder for SNAC model. This class is responsible for decoding audio data
    using the SNAC model. It handles CUDA streams and device management for efficient
    processing.

    This class is implemented as a singleton to ensure that only one instance of the
    decoder is created and used throughout the application.
    """
    def __init__(self, device: str) -> None:
        self.model: SNAC = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.device = device

        self.model = self.model.to(self.device)

        # Prepare CUDA streams for parallel processing if available
        self.cuda_stream: torch.cuda.Stream | None = None

        if self.cuda_device:
            self.cuda_stream = cast(torch.cuda.Stream, torch.cuda.Stream())

    @cached_property
    def cuda_device(self) -> bool:
        return 'cuda' in self.device
    
    def convert_to_audio(self, multiframe: list[int]) -> bytes | None:
        if len(multiframe) < 7:
            return None

        num_frames = len(multiframe) // 7
        frame_len = num_frames * 7
        
        # Create tensor directly from the relevant slice of the list
        frame_tensor = torch.tensor(
            multiframe[:frame_len], 
            dtype=torch.int32, 
            device=self.device,
        )

        # Reshape to (num_frames, 7) to easily access columns
        reshaped_frames = frame_tensor.view(num_frames, 7)

        # Select columns using advanced indexing and flatten where necessary
        codes_0 = reshaped_frames[:, 0]
        codes_1 = reshaped_frames[:, [1, 4]].flatten()
        codes_2 = reshaped_frames[:, [2, 3, 5, 6]].flatten()

        # Reshape codes into the expected [1, N] format for the model
        codes = [
            codes_0.unsqueeze(0),
            codes_1.unsqueeze(0),
            codes_2.unsqueeze(0)
        ]

        # OPTIMIZATION 2: More efficient range check
        if torch.any((codes[0] < 0) | (codes[0] > 4096)) or \
           torch.any((codes[1] < 0) | (codes[1] > 4096)) or \
           torch.any((codes[2] < 0) | (codes[2] > 4096)):
            return None

        with torch.inference_mode():
            context = torch.cuda.stream(self.cuda_stream) if self.cuda_device else torch.no_grad()

            with context:
                audio_hat = self.model.decode(codes)

                # Slice the relevant audio segment (this is a fast, no-copy view)
                audio_slice = audio_hat[:, :, 2048:4096]

                # Scale and convert to int16 on the GPU to minimize data transfer size
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16)

                # Transfer final, small tensor to CPU and convert to bytes
                audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()

        return audio_bytes
    
    def _convert_to_audio(self, multiframe: list[int]) -> bytes | None:
        if len(multiframe) < 7:
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        # Pre-allocate tensors instead of incrementally building them
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=self.device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=self.device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=self.device)

        # Use vectorized operations where possible
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=self.device)

        # Direct indexing is much faster than concatenation in a loop
        for j in range(num_frames):
            idx = j * 7

            # Code 0 - single value per frame
            codes_0[j] = frame_tensor[idx]

            # Code 1 - two values per frame
            codes_1[j*2] = frame_tensor[idx+1]
            codes_1[j*2+1] = frame_tensor[idx+4]

            # Code 2 - four values per frame
            codes_2[j*4] = frame_tensor[idx+2]
            codes_2[j*4+1] = frame_tensor[idx+3]
            codes_2[j*4+2] = frame_tensor[idx+5]
            codes_2[j*4+3] = frame_tensor[idx+6]

        # Reshape codes into expected format
        codes = [
            codes_0.unsqueeze(0),
            codes_1.unsqueeze(0),
            codes_2.unsqueeze(0)
        ]

        if (
            torch.any((codes[0] < 0) | (codes[0] > 4096)) or 
            torch.any((codes[1] < 0) | (codes[1] > 4096)) or 
            torch.any((codes[2] < 0) | (codes[2] > 4096))
        ):
            return None
        
        with torch.inference_mode():
            context = torch.cuda.stream(self.cuda_stream) if self.cuda_stream is not None else torch.no_grad()

            with context:
                audio_hat = self.model.decode(codes) # Decode the audio

                # Extract the relevant slice and efficiently convert to bytes
                # Keep data on GPU as long as possible
                audio_slice = audio_hat[:, :, 2048:4096]
                audio_int16_tensor = (audio_slice * 32767).to(torch.int16) # Scale directly on GPU

                # Only transfer the final result to CPU
                audio_bytes = (
                    audio_int16_tensor
                        .cpu()
                        .numpy()
                        .tobytes()
                )

        return audio_bytes
