import numpy as np
import pytest

from audio_transformers.core.gaussian_noise import GaussianNoise
from audio_transformers.core.model import Signal


def test_gaussian_noise_stats():
    rate = 16000
    amplitude = 0.8
    probe_signal = Signal(np.zeros((2, rate * 10), dtype=np.float32), rate)
    aug = GaussianNoise(amplitude=amplitude)
    output = aug(probe_signal)

    assert output.data.mean() == pytest.approx(0.0, abs=0.01)
    assert output.data.std().mean() == pytest.approx(1.0, abs=0.5)
    assert output.data.shape == probe_signal.data.shape
    assert output.rate == probe_signal.rate
