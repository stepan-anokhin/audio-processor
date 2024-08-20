from audio_transformers.core.two_side_filter import TwoSideFilter


class BandPass(TwoSideFilter):
    """Apply band-pass filter."""

    def __init__(self, low_cutoff: float, high_cutoff: float, roll_off: int = 6):
        """
        :param low_cutoff: Start of the pass-band at which attenuation is approx. -3dB (Hz)
        :param high_cutoff: End of the pass-band at which attenuation is approx -3dB (Hz)
        :param roll_off: Signal attenuation slope (dB/octave)
        """
        super().__init__("bandpass", low_cutoff, high_cutoff, roll_off)
