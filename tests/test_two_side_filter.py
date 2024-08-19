import math

import pytest

from audio_transformers.core.band_pass import BandPass
from audio_transformers.core.band_stop import BandStop
from tests.utils import sinusoid


def test_band_pass_sin():
    rate = 16000
    low_cutoff = 2000
    high_cutoff = 4000
    probe_low = sinusoid(low_cutoff // 2, rate)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate)  # Geometric center of the pass-band
    probe_high = sinusoid(high_cutoff * 2, rate)  # High cutoff + octave

    aug = BandPass(low_cutoff, high_cutoff)
    output_low = aug(probe_low, rate)
    output_center = aug(probe_center, rate)
    output_high = aug(probe_high, rate)

    assert output_low.max() / probe_low.max() < 1 / math.sqrt(2)
    assert output_center.max() / probe_center.max() == pytest.approx(1.0, 0.1)
    assert output_high.max() / probe_high.max() < 1 / math.sqrt(2)


def test_band_stop_sin():
    rate = 32000
    low_cutoff = 1000
    high_cutoff = 4000
    probe_low = sinusoid(low_cutoff // 2, rate)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate)  # Geometric center of the stop-band
    probe_high = sinusoid(high_cutoff * 2, rate)  # High cutoff + octave

    aug = BandStop(low_cutoff, high_cutoff)
    output_low = aug(probe_low, rate)
    output_center = aug(probe_center, rate)
    output_high = aug(probe_high, rate)

    assert output_low.max() / probe_low.max() == pytest.approx(1.0, 0.1)
    assert output_center.max() / probe_center.max() < 1 / 2
    assert output_high.max() / probe_high.max() == pytest.approx(1.0, 0.1)


def test_band_pass_high():
    rate = 16000
    nyquist = rate // 2
    low_cutoff = 2000
    high_cutoff = nyquist * 1.5
    probe_low = sinusoid(low_cutoff // 2, rate)  # Low cutoff - octave
    probe_center = sinusoid(math.sqrt(low_cutoff * high_cutoff), rate)  # Geometric center of the pass-band

    aug = BandPass(low_cutoff, high_cutoff)
    output_low = aug(probe_low, rate)
    output_center = aug(probe_center, rate)

    assert output_low.max() / probe_low.max() < 1 / math.sqrt(2)
    assert output_center.max() / probe_center.max() == pytest.approx(1.0, 0.1)
