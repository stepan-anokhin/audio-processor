from audio_transformers.core.inversion import Inversion
from tests.utils import sinusoid


def test_inversion_basic():
    rate = 100
    probe_signal = sinusoid(freq=rate // 2, rate=rate)
    aug = Inversion()
    output = aug(probe_signal)
    assert output.data.shape == probe_signal.data.shape
    assert abs(probe_signal.data.max()) == abs(probe_signal.data.max())
    assert output.rate == probe_signal.rate
