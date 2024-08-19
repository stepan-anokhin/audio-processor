from dataclasses import dataclass
from typing import Mapping

from audio_transformers.config.initializers import Initializer
from audio_transformers.config.reader import DEFAULT_TRANSFORMS


@dataclass(frozen=True)
class CliConfig:
    """CLI tool configuration."""

    transforms: Mapping[str, Initializer] = DEFAULT_TRANSFORMS
    input_block_duration: float = 60.0  # 10m blocks
