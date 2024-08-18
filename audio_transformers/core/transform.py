import abc
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Transform(abc.ABC):
    """Abstract base for audio signal transformers."""

    # The transformation is uniform if applying it to chunks
    # and then concatenating results is equivalent to applying
    # the transformation to the whole signal.
    uniform: bool = True

    @abstractmethod
    def __call__(self, signal: NDArray[np.float32], rate: int) -> NDArray[np.float32]:
        """Apply transformation to the given signal samples.

        :param signal: Signal data with shape=(n_channels, n_samples), dtype=float32
        :param rate: Sampling rate (hz)
        :return: Transformed signal

        As for some transformations we may want to specify frequency-valued parameters
        it is required to pass sampling rate in order to use them correctly.
        """
