import pytest

from audio_transformers.core.speed_perturbation import SpeedPerturbation
from tests.utils import sinusoid, fundamental_freq


@pytest.mark.parametrize("speed_factor", (0.5, 2.0))
@pytest.mark.parametrize("ch", (1, 2))
def test_speed_perturbation(speed_factor, ch):
    rate = 16000
    freq_first = 1000
    freq_second = 4000

    # Two sinusoid with different frequencies concatenated
    probe_signal = sinusoid(freq_first, rate, channels=ch) + sinusoid(freq_second, rate, channels=ch)

    aug = SpeedPerturbation(speed_factor)
    output_signal = aug(probe_signal)

    input_length = probe_signal.samples
    output_length = output_signal.samples
    length_half = output_length // 2
    first_half = output_signal[:length_half]
    second_half = output_signal[length_half:]

    assert output_length / input_length == pytest.approx(1 / speed_factor, rel=0.05)
    assert fundamental_freq(first_half) == pytest.approx(freq_first)
    assert fundamental_freq(second_half) == pytest.approx(freq_second)
