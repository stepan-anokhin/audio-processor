from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform


class Inversion(Transform):
    """Inverse waveform polarity by multiplying it by -1."""

    def __call__(self, signal: Signal) -> Signal:
        return Signal(-signal.data, signal.rate)
