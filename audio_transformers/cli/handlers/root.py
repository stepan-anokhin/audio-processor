import sys

import fire

from audio_transformers.cli.config import CliConfig
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.cli.handlers.datasets import DatasetsHandler
from audio_transformers.cli.handlers.transform import TransformHandler
from audio_transformers.utils.console import Console


class RootHandler:
    """Audio transformation and augmentation tool."""

    def __init__(self, datasets: DatasetsHandler, transform: TransformHandler):
        self.datasets: DatasetsHandler = datasets
        self.transform: TransformHandler = transform

    @staticmethod
    def make(config: CliConfig = CliConfig()) -> "RootHandler":
        """Initialize root handler based on CLI config."""
        console = RootHandler.make_console(config)
        datasets_handler = DatasetsHandler(
            console=console,
            public_datasets=config.public_datasets,
        )
        transform_handler = TransformHandler(
            console=console,
            transforms=config.transforms,
            input_block_duration=config.input_block_duration,
        )
        root_handler = RootHandler(
            datasets=datasets_handler,
            transform=transform_handler,
        )
        return root_handler

    @staticmethod
    def make_console(config: CliConfig = CliConfig()) -> Console:
        """Initialize console."""
        return Console(output_file=config.output_file, errors_file=config.errors_file)


def run(name: str = "audio", config: CliConfig = CliConfig()):
    """CLI entry point."""
    try:
        fire.Fire(RootHandler.make(config), name=name)
    except CliUsageError as usage_error:
        console = RootHandler.make_console(config)
        console.error(str(usage_error))
        sys.exit(2)


if __name__ == "__main__":
    run()
