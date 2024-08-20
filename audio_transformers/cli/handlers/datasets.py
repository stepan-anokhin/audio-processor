import os
from typing import Sequence, Dict

from tqdm import tqdm

from audio_transformers.cli.datasets.public import DatasetSource, PublicDataset
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.utils.console import Format, Console

DEFAULT_DOWNLOAD_DIR: str = "~/.audio-processor/datasets"


class DatasetsHandler:
    """Download and prepare for processing public datasets."""

    def __init__(self, console: Console, public_datasets: Sequence[DatasetSource]):
        self._console = console
        self._sources: Sequence[DatasetSource] = public_datasets
        self._index: Dict[str, DatasetSource] = {dataset.name: dataset for dataset in public_datasets}

    def list(self, format: Format = "table"):
        """List datasets."""
        self._console.output(self._sources, format)

    def download(self, name: str, path: str | None = None):
        """Download dataset.

        Default download directory is '~/.audio-processor/datasets/<dataset.name>/'
        """
        if name not in self._index:
            raise CliUsageError(f"Unknown dataset name: {name}")
        source = self._index[name]
        path = path or os.path.join(DEFAULT_DOWNLOAD_DIR)
        if os.path.exists(path) and not PublicDataset(path).exists():
            path = os.path.join(path, source.name)

        if not source.should_pool(path):
            self._console.ok("Dataset is up to date!")
            return

        total_bytes = source.download_bytes()
        with tqdm(total=total_bytes, unit="bytes", unit_scale=True) as progress:
            source.pull(path, progress=progress.update)
