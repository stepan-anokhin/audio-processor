import math

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform


class SpeedPerturbation(Transform):
    """Speed perturbation transformer."""

    def __init__(self, speed_factor: float, window_size: float = 0.1):
        """
        :param speed_factor: Speed perturbation factor.
        :param window_size: Short Time FFT window size in seconds.
        """
        self.speed_factor: float = speed_factor
        self.window_size: float = window_size

    def __call__(self, signal: Signal) -> Signal:
        window_size_samples = int(self.window_size * signal.rate)  # Window size in samples
        hop_samples = window_size_samples // 2
        window = gaussian(window_size_samples, std=window_size_samples // 2, sym=True)
        stft = ShortTimeFFT(window, hop=hop_samples, fs=signal.rate, scale_to="magnitude")
        spectre = stft.stft(signal.data)

        # Now we need to stretch or squeeze spectre depending on the speed factor
        spectre_samples = spectre.shape[-1]
        target_samples = math.floor(spectre_samples / self.speed_factor)  # Number of samples in target spectre
        target_shape = list(spectre.shape)
        target_shape[-1] = target_samples
        target_spectre = np.zeros(target_shape, dtype=spectre.dtype)
        per_sample = spectre_samples / target_samples
        for i in range(target_samples):
            start_index = math.floor(per_sample * i)
            end_index = math.floor(per_sample * (i + 1)) + 1
            target_spectre[:, :, i] = spectre[:, :, start_index:end_index].mean(axis=-1)

        output = stft.istft(target_spectre)
        return Signal(output, signal.rate)
