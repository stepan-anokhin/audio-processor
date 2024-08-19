import sys

import fire

from audio_transformers.cli.errors import CliUsageError
from audio_transformers.utils.console import Console


class CommandHandler:
    """Audio transformation and augmentation tool."""

    def __init__(self):
        pass


def run(name: str = "audio"):
    """CLI entry point."""
    try:
        fire.Fire(CommandHandler(), name=name)
    except CliUsageError as usage_error:
        Console.error(f"Usage error: {usage_error}")
        sys.exit(2)
