from audio_transformers.core.low_pass import LowPass
from tests.utils import pulse


def test_low_pass():
    cutoff = 1000  # 10 kHz
    rate = 16000
    low_freq_signal = pulse(cutoff / 2, rate)
    high_freq_signal = pulse(cutoff * 2, rate)

    low_pass = LowPass(cutoff_freq=cutoff)
    passed = low_pass(low_freq_signal, rate)
    stopped = low_pass(high_freq_signal, rate)

    assert passed.max() / stopped.max() > 2
