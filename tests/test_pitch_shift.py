import numpy as np
import pytest

from audio_transformers.core.pitch_shift import PitchShift
from tests.utils import sinusoid, fundamental_freq


def test_pitch_shift_sin():
    rate = 16000
    probe_freq = 2000.0  # Hz
    shift = 1.0  # 1 octave

    probe_signal = sinusoid(probe_freq, rate)

    aug = PitchShift(shift)
    output_signal = aug(probe_signal, rate)
    output_freq = fundamental_freq(output_signal, rate)

    assert output_freq == pytest.approx(probe_freq * 2**shift)
    assert output_signal.shape == probe_signal.shape


def test_pitch_shift_stereo():
    rate = 16000
    probe_freq = 2000.0  # Hz
    shift = 1.0  # 1 octave

    probe_signal = sinusoid(probe_freq, rate)
    probe_signal = np.concatenate([probe_signal, probe_signal], axis=0)

    aug = PitchShift(shift)
    output_signal = aug(probe_signal, rate)
    output_freq = fundamental_freq(output_signal, rate)

    assert output_freq == pytest.approx(probe_freq * 2**shift)
    assert output_signal.shape == probe_signal.shape
