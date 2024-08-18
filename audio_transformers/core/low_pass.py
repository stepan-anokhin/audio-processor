import numpy as np
import scipy
from numpy.typing import NDArray

from audio_transformers.core.transform import Transform


class LowPass(Transform):
    """Implements low-pass filter."""

    def __init__(self, cutoff_freq: float, roll_off: int = 6):
        """
        :param cutoff_freq: Cutoff frequency at which attenuation reaches -3dB (Hz)
        :param roll_off: Signal attenuation slope (dB/octave)
        """
        self.cutoff_freq: float = cutoff_freq
        self.roll_off: int = roll_off

    def __call__(self, signal: NDArray[np.float32], rate: int) -> NDArray[np.float32]:
        nyquist_freq = rate // 2
        if self.cutoff_freq >= nyquist_freq:
            return signal
        sos_coefficients = scipy.signal.butter(
            self.roll_off // 6,
            self.cutoff_freq,
            btype="lowpass",
            analog=False,
            fs=rate,
            output="sos",
        )
        processed = np.zeros_like(signal, dtype=np.float32)
        for channel in range(signal.shape[0]):
            sos_start = scipy.signal.sosfilt_zi(sos_coefficients) * signal[channel, 0]
            processed_channel, _ = scipy.signal.sosfilt(sos_coefficients, signal[channel, :], zi=sos_start)
            processed[channel, :] = processed_channel
        return processed
