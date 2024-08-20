import sys

import fire

from audio_transformers.cli.config import CliConfig
from audio_transformers.cli.datasets.handler import DatasetsHandler
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.cli.handler import Handler
from audio_transformers.cli.transform_handler import TransformHandler
from audio_transformers.utils.console import Console


class RootHandler(Handler):
    """Audio transformation and augmentation tool."""

    def __init__(self, config: CliConfig = CliConfig()):
        super().__init__(config)
        self.datasets = DatasetsHandler(config)
        self.transform = TransformHandler(config)


def run(name: str = "audio"):
    """CLI entry point."""
    try:
        fire.Fire(RootHandler(), name=name)
    except CliUsageError as usage_error:
        Console.error(str(usage_error))
        sys.exit(2)


if __name__ == "__main__":
    run()
