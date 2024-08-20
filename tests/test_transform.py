import json
import tempfile
from io import StringIO

import pytest

from audio_transformers.cli.handlers.transform import TransformHandler
from audio_transformers.cli.task.executor import DEFAULT_TRANSFORMS
from audio_transformers.core.pitch_shift import PitchShift
from audio_transformers.utils.console import Console
from audio_transformers.utils.docs import Docs


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
