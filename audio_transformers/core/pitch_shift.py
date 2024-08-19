import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform


class PitchShift(Transform):
    """Pitch shift transformation."""

    def __init__(self, shift: float, fft_window_size: float = 0.1):
        """
        :param shift: Pitch shift in octaves.
        :param fft_window_size: Short Time FFT window size in seconds.
        """
        self.shift: float = shift
        self.window_size: float = fft_window_size

    def __call__(self, signal: Signal) -> Signal:
        window_size_samples = int(self.window_size * signal.rate)  # Window size in samples
        hop_samples = window_size_samples // 2
        window = gaussian(window_size_samples, std=window_size_samples // 2, sym=True)
        stft = ShortTimeFFT(window, hop=hop_samples, fs=signal.rate, scale_to="magnitude")
        spectre = stft.stft(signal.data)

        # Now we need to scale frequencies of the spectre
        spectre_func = interp1d(stft.f, spectre, axis=-2, bounds_error=False, fill_value=0.0)
        factor = 2 ** -self.shift
        result_spectre = np.zeros_like(spectre)
        for i, freq in enumerate(stft.f):
            result_spectre[:, i, :] = spectre_func(freq * factor)
        output = stft.istft(result_spectre)

        # truncate extra window-size in output signal
        output = output[:, : signal.data.shape[-1]]
        return Signal(output, signal.rate)
