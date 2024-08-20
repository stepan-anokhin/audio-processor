import numpy as np

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform


class GaussianNoise(Transform):
    """Add gaussian noise to the signal."""

    def __init__(self, amplitude: float):
        """
        :param amplitude: Noise amplitude.
        """
        self.amplitude: float = amplitude

    def __call__(self, signal: Signal) -> Signal:
        noise = self.amplitude * np.random.randn(*signal.data.shape)
        return Signal(signal.data + noise, signal.rate)
