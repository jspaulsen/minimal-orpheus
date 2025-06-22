import io
from typing import Generator
import httpx
import pydub

from src.stream.constants import AUDIO_PTIME
from src.stream.frame import Frame
from src.tts.streamelements.speakers import StreamElementsSpeaker


def frame_generator(segment: pydub.AudioSegment, frame_size: int) -> Generator[bytes, None, None]:
    raw_data = segment.raw_data

    if raw_data is None:
        return

    for i in range(0, len(raw_data), frame_size):
        chunk = raw_data[i:i+frame_size]

        # Pad the last chunk if it's shorter
        if len(chunk) < frame_size:
            chunk = chunk.ljust(frame_size, b'\0')

        yield chunk


class StreamElements:
    def __init__(self) -> None:
        self.url = "https://api.streamelements.com/kappa/v2/speech"

    def generate_audio(
        self,
        speaker: StreamElementsSpeaker,
        text: str,
    ) -> Generator[Frame, None, None]:
        response = httpx.get(
            self.url, params={
                "voice": speaker.value,
                "text": text,
            }
        )

        response.raise_for_status()
        audio = io.BytesIO(response.content)
        segment: pydub.AudioSegment = pydub.AudioSegment.from_mp3(audio)
        sample_width = 2

        # TODO: Maybe make this configurable
        segment.set_channels(1)
        segment.set_sample_width(sample_width)
        segment = segment.set_frame_rate(24000)

        frame_size = int(segment.frame_rate * AUDIO_PTIME) * sample_width
        generator = frame_generator(
            segment,
            frame_size,
        )

        for frame in generator:
            yield Frame(
                data=frame,
                channels=1,
                sample_width=sample_width,
                sample_rate=segment.frame_rate,
            )
