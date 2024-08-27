import math

import numpy as np
from scipy.interpolate import interp1d
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
        # Calculate spectrogram or original signal
        window_size_samples = int(self.window_size * signal.rate)  # Window size in samples
        hop_samples = window_size_samples // 2
        window = gaussian(window_size_samples, std=window_size_samples // 2, sym=True)
        stft = ShortTimeFFT(window, hop=hop_samples, fs=signal.rate, scale_to="magnitude")
        spectre = stft.stft(signal.data)

        # Now we need to stretch or squeeze spectrogram depending on the speed factor

        # Initialize empty result array
        orig_spectre_samples = spectre.shape[-1]
        target_spectre_samples = max(math.floor(orig_spectre_samples / self.speed_factor), 1)
        target_shape = list(spectre.shape)
        target_shape[-1] = target_spectre_samples
        target_spectre = np.zeros(target_shape, dtype=spectre.dtype)

        # Represent spectre as a function of time (measured in
        # sample index) by linear interpolation along the time axis.
        time_values = range(orig_spectre_samples)
        spectre_func = interp1d(time_values, spectre, axis=-1, bounds_error=False, fill_value=0.0)

        # Fill the target spectre
        for time in range(target_spectre_samples):
            orig_time = orig_spectre_samples / target_spectre_samples * time
            target_spectre[:, :, time] = spectre_func(orig_time)

        # Reverse Short-Time Fourier Transform
        output = stft.istft(target_spectre)
        return Signal(output, signal.rate)
