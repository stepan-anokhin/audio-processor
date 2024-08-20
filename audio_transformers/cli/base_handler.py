from typing import Sequence, Any

from audio_transformers.cli.config import CliConfig
from audio_transformers.utils.console import Format, Console


class BaseHandler:
    """Base class for CLI handlers to provide shared functionality."""

    _config: CliConfig

    def __init__(self, config: CliConfig = CliConfig()):
        self._config = config

    def _output(self, data: Sequence[Any], format: Format = "table"):
        """Output data collection."""
        Console.output(data, format, self._config.file)

    def _error(self, message: str, prefix: str = "ERROR:", end: str = "\n"):
        """Print error message."""
        Console.error(message, prefix, end, self._config.file)

    def _fatal(self, message: str, end: str = "\n"):
        """Print fatal error message."""
        Console.error(message, prefix="FATAL:", end=end, file=self._config.file)

    def _warning(self, message: str, prefix: str = "WARNING:", end: str = "\n"):
        """Print warning message."""
        Console.warning(message, prefix, end, file=self._config.file)

    def _ok(self, message: str, prefix: str = "OK:", end: str = "\n"):
        """Print OK message."""
        Console.ok(message, prefix, end, file=self._config.file)
