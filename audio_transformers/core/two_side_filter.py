from typing import TypeAlias, Literal

import numpy as np
import scipy

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform

TwoSideType: TypeAlias = Literal["bandpass", "bandstop"]


class TwoSideFilter(Transform):
    """Implements two-side filter (with ∩ or ∪-shaped transfer function)."""

    def __init__(self, filter_type: TwoSideType, low_cutoff: float, high_cutoff: float, roll_off: int = 6):
        """
        :param filter_type: Two-side filter type ("bandpass" or "bandstop")
        :param low_cutoff: Start of the pass- or stop-band at which attenuation is approx. -3dB (Hz)
        :param high_cutoff: End of the pass- or stop-band at which attenuation is approx -3dB (Hz)
        :param roll_off: Signal attenuation slope (dB/octave)
        """
        self.type: TwoSideType = filter_type
        self.low_cutoff: float = low_cutoff
        self.high_cutoff: float = high_cutoff
        self.roll_off: int = roll_off

    def __call__(self, signal: Signal) -> Signal:
        nyquist_freq = signal.rate // 2
        low_cutoff = self.low_cutoff
        high_cutoff = min(self.high_cutoff, nyquist_freq * 0.999)

        # We cannot initialize the second-order sections coefficients
        # in advance because we need to know sampling rate for that.
        sos_coefficients = scipy.signal.butter(
            self.roll_off // 6,
            [low_cutoff, high_cutoff],
            btype=self.type,
            analog=False,
            fs=signal.rate,
            output="sos",
        )

        processed = np.zeros_like(signal.data, dtype=np.float32)
        for channel in range(signal.data.shape[0]):
            sos_start = scipy.signal.sosfilt_zi(sos_coefficients) * signal.data[channel, 0]
            processed_channel, _ = scipy.signal.sosfilt(sos_coefficients, signal.data[channel, :], zi=sos_start)
            processed[channel, :] = processed_channel
        return Signal(processed, signal.rate)
