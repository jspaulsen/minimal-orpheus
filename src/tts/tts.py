import abc
from typing import Generator


class TTSModel(abc.ABC):
    @abc.abstractmethod
    def speak(self, text: str) -> Generator[bytes, None, None]:
        """
        Convert text to speech.

        Args:
            text (str): The text to convert to speech.

        Yields:
            bytes: The audio data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
