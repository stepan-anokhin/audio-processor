from typing import Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian


def pulse(freq: float, rate: int, time_start: float = -1, time_stop: float = 1) -> NDArray[np.float32]:
    """Generate gauss pulse signal."""
    duration = time_stop - time_start
    time = np.linspace(time_start, time_stop, int(duration * rate), endpoint=False)
    signal = scipy.signal.gausspulse(time, fc=freq, retquad=False, retenv=False)
    return signal.reshape((1, len(signal)))


def sinusoid(freq: float, rate: int, time_start=0, time_stop=1, phase: float = 0) -> NDArray[np.float32]:
    """Generate sinusoid signal."""
    duration = time_stop - time_start
    time = np.linspace(time_start, time_stop, int(duration * rate))
    angular_freq = 2 * np.pi * freq
    signal = np.sin(time * angular_freq + phase)
    return signal.reshape((1, len(signal)))


def get_spectre(signal: NDArray[np.float32], rate: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate spectre of the signal."""
    window_size_samples = rate // 10  # Window size is 100 ms
    hop_samples = window_size_samples // 4
    window = gaussian(window_size_samples, std=window_size_samples // 2, sym=True)
    stft = ShortTimeFFT(window, hop=hop_samples, fs=rate, scale_to="magnitude")
    return stft.stft(signal), stft.f


def fundamental_freq(signal: NDArray[np.float32], rate: int) -> float:
    """Calculate the loudest frequency in the signal."""
    spectre, frequencies = get_spectre(signal, rate)
    max_index = spectre.mean(axis=-1).argmax(axis=-1)
    return frequencies[max_index]
