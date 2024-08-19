import math

import pytest

from audio_transformers.core.band_pass import BandPass
from audio_transformers.core.band_stop import BandStop
from tests.utils import sinusoid


@pytest.mark.parametrize("channels", (1, 2))
def test_band_pass_sin(channels: int):
    rate = 16000
    low_cutoff = 2000
    high_cutoff = 4000
    probe_low = sinusoid(low_cutoff // 2, rate, channels=channels)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate, channels=channels)  # Geometric center
    probe_high = sinusoid(high_cutoff * 2, rate, channels=channels)  # High cutoff + octave

    aug = BandPass(low_cutoff, high_cutoff)
    output_low = aug(probe_low)
    output_center = aug(probe_center)
    output_high = aug(probe_high)

    assert output_low.data.max() / probe_low.data.max() < 1 / math.sqrt(2)
    assert output_center.data.max() / probe_center.data.max() == pytest.approx(1.0, 0.1)
    assert output_high.data.max() / probe_high.data.max() < 1 / math.sqrt(2)


@pytest.mark.parametrize("channels", (1, 2))
def test_band_stop_sin(channels: int):
    rate = 32000
    low_cutoff = 1000
    high_cutoff = 4000
    probe_low = sinusoid(low_cutoff // 2, rate, channels=channels)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate, channels=channels)  # Geometric center
    probe_high = sinusoid(high_cutoff * 2, rate, channels=channels)  # High cutoff + octave

    aug = BandStop(low_cutoff, high_cutoff)
    output_low = aug(probe_low)
    output_center = aug(probe_center)
    output_high = aug(probe_high)

    assert output_low.data.max() / probe_low.data.max() == pytest.approx(1.0, 0.1)
    assert output_center.data.max() / probe_center.data.max() < 1 / 2
    assert output_high.data.max() / probe_high.data.max() == pytest.approx(1.0, 0.1)


@pytest.mark.parametrize("channels", (1, 2))
def test_band_pass_high(channels: int):
    rate = 16000
    nyquist = rate // 2
    low_cutoff = 2000
    high_cutoff = nyquist * 1.5
    probe_low = sinusoid(low_cutoff // 2, rate, channels=channels)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate, channels=channels)  # Geometric center

    aug = BandPass(low_cutoff, high_cutoff)
    output_low = aug(probe_low)
    output_center = aug(probe_center)

    assert output_low.data.max() / probe_low.data.max() < 1 / math.sqrt(2)
    assert output_center.data.max() / probe_center.data.max() == pytest.approx(1.0, 0.1)
