import json
import os
import tempfile
from dataclasses import dataclass, asdict
from io import StringIO
from typing import Sequence

import pytest

from audio_transformers.cli.datasets.public import DatasetSource, PublicDataset
from audio_transformers.cli.handlers.datasets import DatasetsHandler
from audio_transformers.utils.console import Console


@dataclass
class DummyDataset:
    """Dummy dataset info."""

    expected_paths: Sequence[str] = (
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

    dataset_path = os.path.join(tempdir, dummy.source.name)
    assert os.path.isdir(dataset_path)
    for file_path in dummy.expected_paths:
        content_path = os.path.join(dataset_path, file_path)
        assert os.path.isfile(content_path)
    assert PublicDataset(dataset_path).exists()

    assert "up to date" not in output_file.getvalue()
    datasets.download(dummy.source.name, dataset_path)
    assert "up to date" in output_file.getvalue()


def test_datasets_list(dummy: DummyDataset):
    output_file = StringIO("")
    console = Console(output_file=output_file, errors_file=StringIO())

    datasets = DatasetsHandler(console, [dummy.source])
    datasets.list(format="json")

    assert json.loads(output_file.getvalue()) == [asdict(dummy.source)]
