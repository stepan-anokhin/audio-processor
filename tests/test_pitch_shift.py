import pytest

from audio_transformers.core.pitch_shift import PitchShift
from tests.utils import sinusoid, fundamental_freq


@pytest.mark.parametrize("channels", (1, 2))
def test_pitch_shift_sin(channels):
    rate = 16000
    probe_freq = 2000.0  # Hz
    shift = 1.0  # 1 octave

    probe_signal = sinusoid(probe_freq, rate, channels=channels)

    aug = PitchShift(shift)
    output_signal = aug(probe_signal)
    output_freq = fundamental_freq(output_signal)

    assert output_freq == pytest.approx(probe_freq * 2**shift)
    assert output_signal.data.shape == probe_signal.data.shape
