import numpy as np
import pytest

from audio_transformers.core.gaussian_noise import GaussianNoise


def test_gaussian_noise_stats():
    rate = 16000
    amplitude = 0.8
    probe_signal = np.zeros((2, rate * 10), dtype=np.float32)
    aug = GaussianNoise(amplitude=amplitude)
    output = aug(probe_signal, rate)

    assert output.mean() == pytest.approx(0.0, abs=0.01)
    assert output.std().mean() == pytest.approx(1.0, abs=0.5)
