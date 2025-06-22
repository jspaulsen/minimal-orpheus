from typing import Generator
import pydub

from src.stream.constants import AUDIO_PTIME
from src.stream.frame import Frame


def pydub_frame_generator(segment: pydub.AudioSegment, frame_size: int) -> Generator[bytes, None, None]:
    raw_data = segment.raw_data

    if raw_data is None:
        return

    for i in range(0, len(raw_data), frame_size):
        chunk = raw_data[i:i+frame_size]

        # Pad the last chunk if it's shorter
        if len(chunk) < frame_size:
            chunk = chunk.ljust(frame_size, b'\0')

        yield chunk


def frame_generator(
    audio_generator: Generator[bytes, None, None],
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> Generator[Frame, None, None]:
    buffer = bytearray()
    frame_width: int = int(sample_rate * sample_width * channels * AUDIO_PTIME)

    for frame in audio_generator:
        if not frame:
            continue

        buffer.extend(frame)

        if len(buffer) < frame_width:
            continue

        # Yield frames of the specified size
        while len(buffer) >= frame_width:
            yield Frame(
                data=bytes(buffer[:frame_width]),
                channels=channels,
                sample_width=sample_width,
                sample_rate=sample_rate,
            )

            buffer = buffer[frame_width:]

    # If we have any remaining data in the buffer, yield it as a final frame;
    # fill the difference with zeroes to match the frame size.
    if len(buffer) > 0:
        remaining_size = frame_width - len(buffer)

        if remaining_size > 0:
            buffer.extend(b'\0' * remaining_size)

        yield Frame(
            data=buffer,
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
        )
