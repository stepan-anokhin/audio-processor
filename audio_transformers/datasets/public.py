import logging
import os
import os.path
from dataclasses import asdict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import humanize
import requests
import yaml
from dacite import from_dict
from humanize import naturalsize
from tqdm import tqdm

import audio_transformers.utils.archives as archives
import audio_transformers.utils.urls as urls
from audio_transformers.utils.console import Tabular

logger = logging.getLogger(__name__)


@dataclass
class Metadata:
    """Public dataset metadata."""

    name: str
    url: str
    etag: str

    DEFAULT_PATH = ".meta"


class PublicDataset:
    """Represents a local copy of a public dataset."""

    path: str
    metadata: Metadata | None = None

    def __init__(self, path: str):
        self.path = path
        self._load()

    def _load(self):
        """Load metadata."""
        if self.exists():
            with open(self.metadata_path) as metadata_file:
                metadata_dict = yaml.safe_load(metadata_file)
            metadata: Metadata = from_dict(Metadata, metadata_dict)
            self.metadata = metadata

    def exists(self) -> bool:
        """Check if dataset exists."""
        return os.path.isfile(self.metadata_path)

    def update(self, metadata: Metadata):
        """Update metadata."""
        self.metadata = metadata
        os.makedirs(self.path, exist_ok=True)
        with open(self.metadata_path, "w") as metadata_file:
            yaml.dump(asdict(metadata), metadata_file)

    @cached_property
    def metadata_path(self) -> str:
        """Get the dataset metadata file location."""
        return os.path.join(self.path, Metadata.DEFAULT_PATH)

    @property
    def etag(self) -> str | None:
        """Get public dataset ETag."""
        if self.metadata is not None:
            return self.metadata.etag

    @property
    def name(self) -> str | None:
        """Get dataset name."""
        if self.metadata is not None:
            return self.metadata.name

    @property
    def source(self) -> str | None:
        """Get source URL."""
        if self.metadata is not None:
            return self.metadata.url


@dataclass(frozen=True)
class DownloadConfig:
    """Download config."""

    chunk_size: int = 10 * 1024  # 10 KiB
    temp_folder: str = "{dataset_path}/.."
    remove_archive: bool = True


@dataclass
class DatasetSource(Tabular):
    """Represents a remote URL source of a public dataset."""

    name: str
    url: str
    format: str
    size: int
    size_archive: int

    @classmethod
    def headers(cls) -> Sequence[str]:
        """Headers in table representation."""
        return ["Name", "Format", "Size", "Archive Size"]

    def table_row(self) -> Sequence[str]:
        """Represent as a table row."""
        return [self.name, self.format, naturalsize(self.size), naturalsize(self.size_archive)]

    def pull(self, path: str, config: DownloadConfig = DownloadConfig()) -> PublicDataset:
        """Download remote dataset to the local directory."""
        dataset = PublicDataset(path)

        with requests.head(self.url, stream=True) as resp:
            etag = resp.headers["ETag"]

        if etag == dataset.etag:
            logger.info(f"Dataset '{dataset.name}' is up to date!")
            return dataset

        with requests.get(self.url, stream=True) as resp:
            logger.info(f"Downloading '{self.name}' ({humanize.naturalsize(self.size)}) to {path}")
            archive_name = urls.filename(self.url)
            archive_dir = os.path.abspath(config.temp_folder.format(dataset_path=dataset.path))
            archive_path = os.path.join(archive_dir, archive_name)
            os.makedirs(archive_dir, exist_ok=True)
            total_size = int(resp.headers["Content-Length"])
            with open(archive_path, "wb") as archive:
                with tqdm(total=total_size, unit="bytes", unit_scale=True) as progress:
                    for chunk in resp.iter_content(chunk_size=config.chunk_size):
                        archive.write(chunk)
                        progress.update(len(chunk))

        archives.extract_all(archive_path, dataset.path)
        metadata = Metadata(name=self.name, url=self.url, etag=etag)
        dataset.update(metadata)

        if config.remove_archive:
            os.remove(archive_path)

        return dataset
