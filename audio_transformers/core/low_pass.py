from audio_transformers.core.one_side_filter import OneSideFilter


class LowPass(OneSideFilter):
    """Apply low-pass filter."""

    def __init__(self, cutoff_freq: float, roll_off: int = 6):
        """
        :param cutoff_freq: Cutoff frequency at which attenuation reaches -3dB (Hz)
        :param roll_off: Signal attenuation slope (dB/octave)
        """
        super().__init__("lowpass", cutoff_freq, roll_off)
