import json
import os
import tempfile
from io import StringIO

import pytest

from audio_transformers.cli.handlers.transform import TransformHandler
from audio_transformers.cli.task.executor import DEFAULT_TRANSFORMS
from audio_transformers.cli.task.model import TaskSpec, TransformSpec
from audio_transformers.core.pitch_shift import PitchShift
from audio_transformers.io.file import AudioFile
from audio_transformers.utils.console import Console
from audio_transformers.utils.docs import Docs
from tests.utils import sinusoid, fundamental_freq


def make_task(speed_factor: float = 0.5, pitch_shift: float = 1.0) -> TaskSpec:
    """Create example task."""
    return TaskSpec(
        transforms=[
            TransformSpec(
                type="PitchShift",
                params=dict(
                    shift=pitch_shift,
                ),
            ),
            TransformSpec(
                type="SpeedPerturbation",
                params=dict(
                    speed_factor=speed_factor,
                ),
            ),
        ]
    )


@pytest.fixture
def tempdir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory(prefix="audio-tests-") as directory:
        yield directory


def test_transform_list():
    output = StringIO()
    console = Console(output_file=output, errors_file=StringIO())

    handler = TransformHandler(console, DEFAULT_TRANSFORMS)
    handler.list(format="json")

    listed = json.loads(output.getvalue())
    assert len(listed) == len(DEFAULT_TRANSFORMS)
    assert {item["name"] for item in listed} == DEFAULT_TRANSFORMS.keys()


def test_transform_params():
    output = StringIO()
    console = Console(output_file=output, errors_file=StringIO())

    handler = TransformHandler(console, DEFAULT_TRANSFORMS)
    handler.params(name="PitchShift", format="json")

    listed = json.loads(output.getvalue())
    docs = Docs.from_func(PitchShift)

    assert len(docs.params) > 0
    assert len(docs.params) == len(listed)
    assert {param["name"] for param in listed} == {param.name for param in docs.params}


def test_transform_file(tempdir):
    output = StringIO()
    console = Console(output_file=output, errors_file=StringIO())

    handler = TransformHandler(console, DEFAULT_TRANSFORMS)

    freq = 1000
    rate = 16000
    input_signal = sinusoid(freq, rate, time_stop=120.0, channels=2)  # 2 minutes sinusoid
    input_path = os.path.join(tempdir, "input.wav")
    output_path = os.path.join(tempdir, "output.mp3")
    task_path = os.path.join(tempdir, "task.yaml")

    with AudioFile(input_path, "w", rate=input_signal.rate) as file:
        file.write(input_signal)

    pitch_shift = 1.0
    speed_factor = 0.5
    task = make_task(pitch_shift=pitch_shift, speed_factor=speed_factor)
    task.save(task_path)

    handler.file(input=input_path, output=output_path, config=task_path)

    with AudioFile(output_path, "r") as file:
        output_signal = file.read()

    assert output_signal.duration == pytest.approx(input_signal.duration / speed_factor, rel=0.1)
    assert fundamental_freq(output_signal) == pytest.approx(fundamental_freq(input_signal) * 2, rel=0.1)


def test_transform_files(tempdir):
    output = StringIO()
    console = Console(output_file=output, errors_file=StringIO())

    handler = TransformHandler(console, DEFAULT_TRANSFORMS)

    freq = 1000
    rate = 16000
    input_signal = sinusoid(freq, rate, time_stop=120.0, channels=2)  # 2 minutes sinusoid
    input_path = os.path.join(tempdir, "nested/file.mp3")
    output_path = os.path.join(tempdir, "nested/file.wav")
    task_path = os.path.join(tempdir, "task.yaml")
    os.makedirs(os.path.join(tempdir, "nested"))

    with AudioFile(input_path, "w", rate=input_signal.rate) as file:
        file.write(input_signal)

    pitch_shift = 1.0
    speed_factor = 0.5
    task = make_task(pitch_shift=pitch_shift, speed_factor=speed_factor)
    task.input_root = tempdir
    task.input_pattern = "**/*.mp3"
    task.output_root = tempdir
    task.output_pattern = "{reldir}/{name}.wav"
    task.save(task_path)

    handler.files(config=task_path)

    with AudioFile(output_path, "r") as file:
        output_signal = file.read()

    assert output_signal.duration == pytest.approx(input_signal.duration / speed_factor, rel=0.1)
    assert fundamental_freq(output_signal) == pytest.approx(fundamental_freq(input_signal) * 2, rel=0.1)
