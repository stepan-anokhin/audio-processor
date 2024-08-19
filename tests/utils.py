from typing import Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from audio_transformers.core.model import Signal


def pulse(freq: float, rate: int, time_start: float = -1, time_stop: float = 1, channels: int = 1) -> Signal:
    """Generate gauss pulse signal."""
    duration = time_stop - time_start
    time = np.linspace(time_start, time_stop, int(duration * rate), endpoint=False)
    data = scipy.signal.gausspulse(time, fc=freq, retquad=False, retenv=False)
    return Signal(np.array([data] * channels), rate)


def sinusoid(freq: float, rate: int, time_start=0, time_stop=1, phase: float = 0, channels: int = 1) -> Signal:
    """Generate sinusoid signal."""
    duration = time_stop - time_start
    time = np.linspace(time_start, time_stop, int(duration * rate))
    angular_freq = 2 * np.pi * freq
    data = np.sin(time * angular_freq + phase)

    return Signal(np.array([data] * channels), rate)


def get_spectre(signal: Signal) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate spectre of the signal."""
    window_size_samples = signal.rate // 10  # Window size is 100 ms
    hop_samples = window_size_samples // 4
    window = gaussian(window_size_samples, std=window_size_samples // 2, sym=True)
    stft = ShortTimeFFT(window, hop=hop_samples, fs=signal.rate, scale_to="magnitude")
    return stft.stft(signal.data), stft.f


def fundamental_freq(signal: Signal) -> float:
    """Calculate the loudest frequency in the signal."""
    spectre, frequencies = get_spectre(signal)
    max_index = spectre.mean(axis=-1).argmax(axis=-1)
    return frequencies[max_index]
