import abc
from abc import abstractmethod

from audio_transformers.core.model import Signal


class Transform(abc.ABC):
    """Abstract base for audio signal transformers."""

    # The transformation is uniform if applying it to chunks
    # and then concatenating results is equivalent to applying
    # the transformation to the whole signal.
    uniform: bool = True

    @abstractmethod
    def __call__(self, signal: Signal) -> Signal:
        """Apply transformation to the given signal samples.

        :param signal: Input signal
        :return: Transformed signal
        """
