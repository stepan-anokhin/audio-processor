from audio_transformers.core.inversion import Inversion
from tests.utils import sinusoid


def test_inversion_basic():
    rate = 100
    probe_signal = sinusoid(freq=rate // 2, rate=rate)
    aug = Inversion()
    output = aug(probe_signal, rate)
    assert output.shape == probe_signal.shape
    assert abs(probe_signal.max()) == abs(probe_signal.max())
