import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Sequence, Mapping

from tqdm import tqdm

import audio_transformers.io.probe as probe
from audio_transformers.cli.errors import CliUsageError
from audio_transformers.cli.task.errors import InitError
from audio_transformers.cli.task.executor import TaskExecutor, FileTask, TaskStats
from audio_transformers.cli.task.initializers import Initializer
from audio_transformers.cli.task.model import TransformSpec, TaskSpec
from audio_transformers.core.transform import Transform
from audio_transformers.utils.console import Tabular, Format, Console

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

    def __init__(self, console: Console, transforms: Mapping[str, Initializer], input_block_duration: float = 60.0):
        self._console: Console = console
        self._transforms: Mapping[str, Initializer] = transforms
        self._input_block_duration: float = input_block_duration

    def list(self, format: Format = "table"):
        """List available transformations"""
        previews = [TransformPreview(name, init.docs.brief) for name, init in self._transforms.items()]
        self._console.output(previews, format)

    def params(self, name: str, format: Format = "table"):
        """List transformation parameters."""
        if name not in self._transforms:
            known = ", ".join(self._transforms.keys())
            raise CliUsageError(f"Unknown transformation: '{name}'. Must be one of: {known}")
        init: Initializer = self._transforms[name]
        self._console.output(init.docs.params, format)

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
            transform_config = TaskSpec.from_file(config)
            specs = transform_config.transforms
        executor = TaskExecutor(self._transforms, self._input_block_duration)

        try:
            transform: Transform = executor.build_transform(specs)
        except InitError as error:
            raise CliUsageError(f"Cannot initialize {error.name} transformation: {error}")

        logger.info(f"Processing file {input} -> {output}")
        task = FileTask(
            input_path=input,
            output_path=output,
            transform=transform,
            block_duration=executor.block_duration,
        )

        start_time = time.time()
        with tqdm(total=probe.samples(input), unit="samples", unit_scale=True) as progress:
            TaskExecutor.execute_subtask_parallel(task, progress.update)
        elapsed = timedelta(seconds=time.time() - start_time)
        logger.info(f"Processing done: {input} -> {output}")
        self._console.ok(f"Done! Elapsed time: {elapsed}")

    def files(
        self,
        input_root: str | None = None,
        input_pattern: str | None = None,
        output_root: str | None = None,
        output_pattern: str | None = None,
        config: str | None = None,
        name: str | None = None,
        **options,
    ):
        """Process multiple files."""
        if name is None and config is None:
            raise CliUsageError("Either transformation name or config file must be provided.")
        task: TaskSpec = TaskSpec.from_cli(
            name=name,
            input_root=input_root,
            input_pattern=input_pattern,
            output_root=output_root,
            output_pattern=output_pattern,
            config=config,
            **options,
        )
        if task.input_root is None:
            task.input_root = "."
        if task.input_pattern is None:
            raise CliUsageError("Input files pattern must be specified either via CLI arguments or config file.")
        if len(task.transforms) == 0:
            raise CliUsageError("At least one transformation must be specified via CLI arguments or config file.")
        executor: TaskExecutor = TaskExecutor(self._transforms)

        start_time = time.time()
        stats: TaskStats = TaskExecutor.stats(task)
        with tqdm(total=stats.total_files, unit="files", unit_scale=True) as progress:
            try:
                executor.execute(task, progress.update)
            except InitError as error:
                raise CliUsageError(f"Cannot initialize {error.name} transformation: {error}")
        elapsed = timedelta(seconds=time.time() - start_time)
        self._console.ok(f"Done! Elapsed time: {elapsed}")
