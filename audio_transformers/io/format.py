import numpy as np
from numpy.typing import NDArray

from audio_transformers.core.model import Signal


def to_signal(raw_data, rate) -> Signal:
    """Convert ffmpegio raw data to Signal."""
    samples, channels = raw_data.shape
    return Signal(raw_data.reshape((channels, samples), order="F"), rate)


def from_signal(signal: Signal) -> NDArray[np.float32]:
    """Convert signal to raw data ready to be written by ffmpegio."""
    channels, samples = signal.data.shape
    return signal.data.reshape((samples, channels), order="F")
