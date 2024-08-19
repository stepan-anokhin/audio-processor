import math

from audio_transformers.core.high_pass import HighPass
from tests.utils import pulse, sinusoid


def test_high_pass_pulse():
    cutoff = 1000  # 10 kHz
    rate = 16000
    low_probe = pulse(cutoff / 2, rate)  # Cutoff - octave
    high_probe = pulse(cutoff * 2, rate)  # Cutoff + octave

    aug = HighPass(cutoff_freq=cutoff)
    passed = aug(high_probe, rate)
    stopped = aug(low_probe, rate)

    assert passed.max() / stopped.max() > math.sqrt(2)


def test_high_pass_sin():
    cutoff = 1000  # 10 kHz
    rate = 16000
    low_probe = sinusoid(cutoff / 2, rate)  # Cutoff - octave
    high_probe = sinusoid(cutoff * 2, rate)  # Cutoff + octave

    aug = HighPass(cutoff_freq=cutoff)
    passed = aug(high_probe, rate)
    stopped = aug(low_probe, rate)

    assert passed.max() / stopped.max() > math.sqrt(2)
