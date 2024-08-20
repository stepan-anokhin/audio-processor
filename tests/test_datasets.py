import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import Sequence

import pytest

from audio_transformers.cli.datasets.public import DatasetSource
from audio_transformers.cli.handlers.datasets import DatasetsHandler
from audio_transformers.utils.console import Console


@dataclass
class DummyDataset:
    """Dummy dataset info."""
    paths: Sequence[str] = (
        "./file.opus",
        "./nested/file.opus",
    )
    source: DatasetSource = DatasetSource(
        name="Dummy",
        url="https://raw.githubusercontent.com/stepan-anokhin/audio-transformers/master/tests/dummy_dataset.tar.gz",
        format="opus",
        size_archive=16295,
        size=39030,
    )


@pytest.fixture
def tempdir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory(prefix="audio-tests-") as directory:
        yield directory


@pytest.fixture
def dummy():
    return DummyDataset()


def test_datasets_download(dummy: DummyDataset, tempdir: str):
    output_file = StringIO("")
    console = Console(output_file=output_file, errors_file=StringIO())

    datasets = DatasetsHandler(console, [dummy.source])
    datasets.download(dummy.source.name, tempdir)

    assert "Downloading" in output_file.getvalue()
