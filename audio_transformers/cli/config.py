import sys
from dataclasses import dataclass
from typing import Mapping, Sequence, TextIO

from audio_transformers.cli.datasets import DEFAULT_DATASETS, DatasetSource
from audio_transformers.cli.task.executor import DEFAULT_TRANSFORMS
from audio_transformers.cli.task.initializers import Initializer


@dataclass(frozen=True)
class CliConfig:
    """CLI tool runtime configuration."""

    transforms: Mapping[str, Initializer] = DEFAULT_TRANSFORMS
    input_block_duration: float = 60.0  # 10m blocks
    public_datasets: Sequence[DatasetSource] = DEFAULT_DATASETS
    file: TextIO = sys.stdout
