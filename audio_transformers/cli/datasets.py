import os
from typing import Tuple, Sequence, Dict

from audio_transformers.cli.errors import CliUsageError
from audio_transformers.datasets.public import DatasetSource, PublicDataset
from audio_transformers.utils.console import Console, Format

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

DEFAULT_DOWNLOAD_DIR: str = "~/.audio-processor/datasets"


class DatasetsHandler:
    """Download and prepare for processing public datasets."""

    def __init__(self, sources: Sequence[DatasetSource] = DEFAULT_DATASETS):
        self._sources: Sequence[DatasetSource] = tuple(sources)
        self._index: Dict[str, DatasetSource] = {dataset.name: dataset for dataset in sources}

    def list(self, format: Format = "table"):
        """List datasets."""
        Console.output(self._sources, format)

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
