import logging
import os
import os.path
from dataclasses import asdict
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Tuple

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


DEFAULT_DATASETS: Tuple[DatasetSource, ...] = (
    DatasetSource(
        name="radio_v4_and_public_speech_5percent",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/radio_pspeech_sample_manifest.tar.gz",  # noqa: E501
        format="opus",
        size_archive=11400000000,
        size=65800000000,
    ),
    DatasetSource(
        name="audiobook_2",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/private_buriy_audiobooks_2.tar.gz",  # noqa: E501
        format="opus",
        size=162000000000,
        size_archive=25800000000,
    ),
    DatasetSource(
        name="radio_2",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/radio_2.tar.gz",
        format="opus",
        size=154000000000,
        size_archive=24600000000,
    ),
    DatasetSource(
        name="public_youtube1120",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube1120.tar.gz",  # noqa: E501
        format="opus",
        size=237000000000,
        size_archive=19000000000,
    ),
    DatasetSource(
        name="asr_public_phone_calls_2",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_public_phone_calls_2.tar.gz",  # noqa: E501
        format="opus",
        size=66000000000,
        size_archive=9400000000,
    ),
    DatasetSource(
        name="public_youtube1120_hq",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube1120_hq.tar.gz",  # noqa: E501
        format="opus",
        size=31000000000,
        size_archive=4900000000,
    ),
    DatasetSource(
        name="asr_public_stories_2",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_public_stories_2.tar.gz",  # noqa: E501
        format="opus",
        size=9000000000,
        size_archive=1400000000,
    ),
    DatasetSource(
        name="tts_russian_addresses_rhvoice_4voices",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/tts_russian_addresses_rhvoice_4voices.tar.gz",  # noqa: E501
        format="opus",
        size=80900000000,
        size_archive=12900000000,
    ),
    DatasetSource(
        name="public_youtube700",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700.tar.gz",  # noqa: E501
        format="opus",
        size=75000000000,
        size_archive=12200000000,
    ),
    DatasetSource(
        name="asr_public_phone_calls_1",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_public_phone_calls_1.tar.gz",  # noqa: E501
        format="opus",
        size=22700000000,
        size_archive=3200000000,
    ),
    DatasetSource(
        name="asr_public_stories_1",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_public_stories_1.tar.gz",  # noqa: E501
        format="opus",
        size=4100000000,
        size_archive=700000000,
    ),
    DatasetSource(
        name="public_series_1",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_series_1.tar.gz",  # noqa: E501
        format="opus",
        size=1900000000,
        size_archive=300000000,
    ),
    DatasetSource(
        name="public_lecture_1",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_lecture_1.tar.gz",  # noqa: E501
        format="opus",
        size=700000000,
        size_archive=100000000,
    ),
    DatasetSource(
        name="asr_calls_2_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_calls_2_val.tar.gz",  # noqa: E501
        format="wav",
        size=2000000000,
        size_archive=800000000,
    ),
    DatasetSource(
        name="buriy_audiobooks_2_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/buriy_audiobooks_2_val.tar.gz",  # noqa: E501
        format="wav",
        size=1000000000,
        size_archive=500000000,
    ),
    DatasetSource(
        name="public_youtube700_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700_val.tar.gz",  # noqa: E501
        format="wav",
        size=2000000000,
        size_archive=130000000,
    ),
)
