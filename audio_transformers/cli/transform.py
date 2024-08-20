import logging
import multiprocessing
import os.path
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Pool
from typing import List, Sequence

from tqdm import tqdm

from audio_transformers.cli.config import CliConfig
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.config.initializers import Initializer
from audio_transformers.config.model import TransformSpec, ConfigFile
from audio_transformers.config.reader import ConfigReader
from audio_transformers.core.transform import Transform
from audio_transformers.io.file import AudioFile
from audio_transformers.utils.console import Console, Tabular, Format

logger = logging.getLogger(__name__)


@dataclass
class TransformPreview(Tabular):
    """Transform preview."""
    name: str
    description: str

    @classmethod
    def headers(cls) -> Sequence[str]:
        return "Name", "Description"

    def table_row(self) -> Sequence[str]:
        return self.name, self.description


class TransformHandler:
    """Transform audio files."""

    _config: CliConfig

    def __init__(self, config: CliConfig):
        self._config = config

    def list(self, format: Format = "table"):
        """List available transformations"""
        previews = [TransformPreview(name, init.docs.brief) for name, init in self._config.transforms.items()]
        Console.output(previews, format)

    def params(self, name: str, format: Format = "table"):
        """List transformation parameters."""
        if name not in self._config.transforms:
            known = ", ".join(self._config.transforms.keys())
            raise CliUsageError(f"Unknown transformation: '{name}'. Must be one of: {known}")
        init: Initializer = self._config.transforms[name]
        Console.output(init.docs.params, format)

    def file(self, input: str, output: str, type: str | None = None, config: str | None = None, **options):
        """Process a single file."""
        if type is None and config is None:
            raise CliUsageError("Either transformation type or a config file must be specified.")
        if type is not None and config is not None:
            raise CliUsageError("Ambiguous usage: transformation type and config cannot be specified simultaneously.")

        specs: List[TransformSpec] = []
        if type is not None:
            specs = [TransformSpec(type=type, params=options)]
        elif config is not None:  # config file is specified
            transform_config = ConfigFile.read(config)
            specs = transform_config.transforms
        reader = ConfigReader(self._config.transforms)
        transform: Transform = reader.build(specs)

        logger.info(f"Processing file {input} -> {output}")

        cpu_count = multiprocessing.cpu_count()
        pool: Pool = multiprocessing.Pool(processes=cpu_count)

        if os.path.exists(output):
            os.remove(output)

        start_time = time.time()
        block_duration = self._config.input_block_duration
        with AudioFile(input, "r", block_duration=block_duration) as input_file:
            with AudioFile(output, "w", rate=input_file.rate) as output_file:
                with tqdm(total=input_file.samples, unit="samples", unit_scale=True) as progress:
                    for result_block in pool.imap(transform, input_file, chunksize=1):
                        output_file.write(result_block)
                        progress.update(len(result_block))
        elapsed = timedelta(seconds=time.time() - start_time)
        logger.info(f"Processing done: {input} -> {output}")
        Console.ok(f"Done! Elapsed time: {elapsed}")
