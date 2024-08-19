import numpy as np
import pytest

from audio_transformers.core.speed_perturbation import SpeedPerturbation
from tests.utils import sinusoid, fundamental_freq


def test_speed_perturbation_slow_down():
    rate = 16000
    freq_first = 1000
    freq_second = 4000
    speed_factor = 0.5

    # Two sinusoid with different frequencies concatenated
    probe_signal = np.concatenate([sinusoid(freq_first, rate), sinusoid(freq_second, rate)], axis=1)

    aug = SpeedPerturbation(speed_factor)
    output_signal = aug(probe_signal, rate)

    input_length = probe_signal.shape[-1]
    output_length = output_signal.shape[-1]
    first_half = output_signal[:, : output_length // 2]
    second_half = output_signal[:, output_length // 2 :]

    assert output_length / input_length == pytest.approx(1 / speed_factor, rel=0.05)
    assert fundamental_freq(first_half, rate) == pytest.approx(freq_first)
    assert fundamental_freq(second_half, rate) == pytest.approx(freq_second)
