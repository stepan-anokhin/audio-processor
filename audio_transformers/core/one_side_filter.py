from typing import TypeAlias, Literal

import numpy as np
import scipy

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform

OneSideType: TypeAlias = Literal["highpass", "lowpass"]


class OneSideFilter(Transform):
    """Implements one-sided filter (with ‾\\ or /‾-shaped transfer function)."""

    def __init__(self, filter_type: OneSideType, cutoff_freq: float, roll_off: int = 6):
        """
        :param filter_type: Filter type ("lowpass" or "highpass").
        :param cutoff_freq: Cutoff frequency at which attenuation reaches -3dB (Hz)
        :param roll_off: Signal attenuation slope (dB/octave)
        """
        self.type: OneSideType = filter_type
        self.cutoff_freq: float = cutoff_freq
        self.roll_off: int = roll_off

    def __call__(self, signal: Signal) -> Signal:
        nyquist_freq = signal.rate // 2
        if self.cutoff_freq >= nyquist_freq and self.type == "lowpass":
            return signal

        # We cannot initialize the second-order sections coefficients
        # in advance because we need to know sampling rate for that.
        sos_coefficients = scipy.signal.butter(
            self.roll_off // 6,
            self.cutoff_freq,
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
