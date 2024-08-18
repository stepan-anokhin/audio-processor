import numpy as np
from numpy.typing import NDArray

from audio_transformers.core.transform import Transform


class GaussianNoise(Transform):
    """Add gaussian noise to the signal."""

    def __init__(self, amplitude: float):
        self.amplitude: float = amplitude

    def __call__(self, signal: NDArray[np.float32], rate: int) -> NDArray[np.float32]:
        noise = self.amplitude * np.random.randn(*signal.shape)
        return signal + noise
