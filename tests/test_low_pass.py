import math

from audio_transformers.core.low_pass import LowPass
from tests.utils import pulse, sinusoid


def test_low_pass_pulse():
    cutoff = 1000  # 10 kHz
    rate = 16000
    low_freq_signal = pulse(cutoff / 2, rate)  # Cutoff - octave
    high_freq_signal = pulse(cutoff * 2, rate)  # Cutoff + octave

    low_pass = LowPass(cutoff_freq=cutoff)
    passed = low_pass(low_freq_signal, rate)
    stopped = low_pass(high_freq_signal, rate)

    assert passed.max() / stopped.max() > math.sqrt(2)


def test_low_pass_sin():
    cutoff = 1000  # 10 kHz
    rate = 16000
    low_freq_signal = sinusoid(cutoff / 2, rate)  # cutoff - octave
    high_freq_signal = sinusoid(cutoff * 2, rate)  # cutoff + octave

    low_pass = LowPass(cutoff_freq=cutoff)
    passed = low_pass(low_freq_signal, rate)
    stopped = low_pass(high_freq_signal, rate)
    print(passed.max())
    print(stopped.max())
    assert passed.max() / stopped.max() > math.sqrt(2)


def test_low_pass_high_cutoff():
    rate = 16000
    nyquist = rate // 2
    cutoff = nyquist + 1000
    probe_signal = sinusoid(cutoff / 2, rate)

    aug = LowPass(cutoff_freq=cutoff)
    assert aug(probe_signal, rate).max() / probe_signal.max() > 0.9
