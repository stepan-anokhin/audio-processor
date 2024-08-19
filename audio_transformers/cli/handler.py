import sys

import fire

from audio_transformers.cli.config import CliConfig
from audio_transformers.cli.datasets import DatasetsHandler
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.cli.transform import TransformHandler
from audio_transformers.utils.console import Console


class CommandHandler:
    """Audio transformation and augmentation tool."""

    _config: CliConfig

    def __init__(self, config: CliConfig = CliConfig()):
        self._config = config
        self.datasets = DatasetsHandler()
        self.transform = TransformHandler(config)


def run(name: str = "audio"):
    """CLI entry point."""
    try:
        fire.Fire(CommandHandler(), name=name)
    except CliUsageError as usage_error:
        Console.error(f"Usage error: {usage_error}")
        sys.exit(2)


if __name__ == "__main__":
    run()
