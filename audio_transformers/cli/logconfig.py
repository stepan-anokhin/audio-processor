import logging
import logging.config
import logging.handlers
import os

from audio_transformers.cli.config import CliConfig


def configure_logging(config: CliConfig = CliConfig()):
    """Apply logging configuration."""
    formatters = {
        "file": {
            "format": config.log.file_format,
        },
        "console": {
            "format": config.log.console_format,
        },
    }
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": config.log.console_level,
            "stream": config.output_file,
        }
    }
    if config.log.file is not None:
        os.makedirs(os.path.dirname(config.log.file), exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file",
            "level": config.log.file_level,
            "filename": config.log.file,
            "backupCount": 3,
        }
    loggers = {"audio_transformers.cli": {"handlers": list(handlers.keys()), "level": "INFO"}}
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": loggers,
        }
    )
