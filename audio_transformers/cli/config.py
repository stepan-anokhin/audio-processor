import sys
from dataclasses import dataclass
from typing import Mapping, Sequence, TextIO, TypeAlias, Literal

from audio_transformers.cli.datasets.public import DEFAULT_DATASETS, DatasetSource
from audio_transformers.cli.task.executor import DEFAULT_TRANSFORMS
from audio_transformers.cli.task.initializers import Initializer

LogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARN", "ERROR"]


@dataclass(frozen=True)
class LogConfig:
    """CLI tool logging config."""

    console_format: str = "%(asctime)s %(levelname)-8s %(message)s"
    file_format: str = "%(asctime)s %(levelname)-8s %(name)-15s %(message)s"
    console_level: LogLevel = "INFO"
    file_level: LogLevel = "INFO"
    file: str | None = None


@dataclass(frozen=True)
class CliConfig:
    """CLI tool runtime configuration."""

    transforms: Mapping[str, Initializer] = DEFAULT_TRANSFORMS
    input_block_duration: float = 60.0  # 10m blocks
    public_datasets: Sequence[DatasetSource] = DEFAULT_DATASETS
    output_file: TextIO = sys.stdout
    errors_file: TextIO = sys.stderr
    log: LogConfig = LogConfig()
