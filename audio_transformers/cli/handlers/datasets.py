import os
from typing import Sequence, Dict

from audio_transformers.cli.config import CliConfig
from audio_transformers.cli.datasets.public import DatasetSource, PublicDataset
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.cli.handlers.base import BaseHandler
from audio_transformers.utils.console import Format

DEFAULT_DOWNLOAD_DIR: str = "~/.audio-processor/datasets"


class DatasetsHandler(BaseHandler):
    """Download and prepare for processing public datasets."""

    def __init__(self, config: CliConfig):
        super().__init__(config)
        self._sources: Sequence[DatasetSource] = config.public_datasets
        self._index: Dict[str, DatasetSource] = {dataset.name: dataset for dataset in config.public_datasets}

    def list(self, format: Format = "table"):
        """List datasets."""
        self._output(self._sources, format)

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
        source.pull(path)
