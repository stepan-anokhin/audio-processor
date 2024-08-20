import logging
import multiprocessing
import os
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from types import MappingProxyType
from typing import Sequence, List, Mapping, Iterator, Callable, Any, Type

from audio_transformers.cli.task.errors import InitError, TaskExecutionError
from audio_transformers.cli.task.initializers import Initializer, BasicInit
from audio_transformers.cli.task.model import TransformSpec, TaskSpec
from audio_transformers.core.band_pass import BandPass
from audio_transformers.core.band_stop import BandStop
from audio_transformers.core.composite import Composite
from audio_transformers.core.gaussian_noise import GaussianNoise
from audio_transformers.core.high_pass import HighPass
from audio_transformers.core.inversion import Inversion
from audio_transformers.core.low_pass import LowPass
from audio_transformers.core.pitch_shift import PitchShift
from audio_transformers.core.speed_perturbation import SpeedPerturbation
from audio_transformers.core.transform import Transform
from audio_transformers.io.file import AudioFile

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORMS: Mapping[str, Initializer] = MappingProxyType(
    {
        "BandPass": BasicInit(BandPass),
        "BandStop": BasicInit(BandStop),
        "GaussianNoise": BasicInit(GaussianNoise),
        "HighPass": BasicInit(HighPass),
        "Inversion": BasicInit(Inversion),
        "LowPass": BasicInit(LowPass),
        "PitchShift": BasicInit(PitchShift),
        "SpeedPerturbation": BasicInit(SpeedPerturbation),
    }
)


@dataclass
class FileTask:
    """Single file processing task."""

    input_path: str
    output_path: str
    transform: Transform
    block_duration: float = 60.0


@dataclass
class ErrorDetails:
    """Subtask failure details."""

    type: Type[Exception]
    message: str
    subtask: FileTask


@dataclass
class TaskStats:
    """Task statistics."""

    total_files: int = 0
    total_size: int = 0


class TaskExecutor:
    """Reads and executes task config."""

    transforms: Mapping[str, Initializer]
    block_duration: float
    tolerate_errors: int = 10

    def __init__(
        self,
        transforms: Mapping[str, Initializer] | None,
        block_duration: float = 60.0,
        tolerate_errors: int = 10,
    ):
        """
        :param transforms: Available transformations.
        :param block_duration: Block duration in streamed IO
        """
        self.transforms = transforms or DEFAULT_TRANSFORMS
        self.block_duration = block_duration

    def build_transform(self, specs: Sequence[TransformSpec]) -> Transform:
        """Build transformation from the spec list."""
        transforms: List[Transform] = []
        for spec in specs:
            if spec.type not in self.transforms:
                known = ", ".join(self.transforms.keys())
                raise InitError(f"Unknown transformation: {spec.type}. Must be one of: {known}", spec.type, None)
            initializer = self.transforms[spec.type]
            try:
                transforms.append(initializer.init(spec, self.transforms))
            except TypeError as error:
                raise InitError(str(error), spec.type, initializer.docs)
        return Composite(transforms)

    @staticmethod
    def resolve_output(input_rel: str, output_root: str, output_pattern: str) -> str:
        """Resolve output path."""
        rel_dir = os.path.dirname(input_rel)
        basename = os.path.basename(input_rel)
        name, ext = os.path.splitext(basename)
        ext = ext.lstrip(".")
        output_rel = output_pattern.format(
            relpath=input_rel,
            reldir=rel_dir,
            name=name,
            ext=ext,
        )
        return os.path.join(output_root, output_rel)

    @staticmethod
    def _input_rel_paths(task: TaskSpec) -> Iterator[str]:
        """Iterate over relative input paths."""
        for path in TaskExecutor._input_paths(task):
            yield fspath(path.relative_to(task.input_root))

    @staticmethod
    def _input_paths(task: TaskSpec) -> Iterator[Path]:
        """Iterate over relative input paths."""
        for path in Path(task.input_root).rglob(task.input_pattern):
            if path.is_file():
                yield path

    @staticmethod
    def _count_input_files(task: TaskSpec) -> int:
        """Get count of input files."""
        total_count: int = 0
        for _ in TaskExecutor._input_rel_paths(task):
            total_count += 1
        return total_count

    def subtasks(self, task: TaskSpec) -> Iterator[FileTask]:
        """List file tasks."""
        transform: Transform = self.build_transform(task.transforms)
        for rel_path in TaskExecutor._input_rel_paths(task):
            input_path = os.path.join(task.input_root, rel_path)
            output_path = TaskExecutor.resolve_output(rel_path, task.output_root, task.output_pattern)
            yield FileTask(input_path, output_path, transform, self.block_duration)

    def execute(self, task: TaskSpec, progress: Callable[[int], Any] | None = None):
        """Execute task."""
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        error: ErrorDetails
        failed_subtasks: int = 0
        for error in pool.imap_unordered(self.execute_subtask, self.subtasks(task), chunksize=1):
            if error is not None:
                logger.exception(
                    "Subtask failed while processing "
                    f"{error.subtask.input_path} -> {error.subtask.output_path}: "
                    f"{error.type.__name__}: {error.message}"
                )
                failed_subtasks += 1
                if failed_subtasks > self.tolerate_errors:
                    raise TaskExecutionError(f"{failed_subtasks} subtasks failed. See log for more details.")
            if progress is not None:
                progress(1)

    @staticmethod
    def execute_subtask(subtask: FileTask) -> ErrorDetails | None:
        """Execute single file processing."""
        try:
            if os.path.exists(subtask.output_path):
                os.remove(subtask.output_path)
            os.makedirs(os.path.dirname(subtask.output_path), exist_ok=True)
            with AudioFile(subtask.input_path, "r", block_duration=subtask.block_duration) as input_file:
                with AudioFile(subtask.output_path, "w", rate=input_file.rate) as output_file:
                    for block in input_file:
                        output_block = subtask.transform(block)
                        output_file.write(output_block)
        except Exception as error:
            return ErrorDetails(
                type=type(error),
                message=str(error),
                subtask=subtask,
            )

    @staticmethod
    def execute_subtask_parallel(subtask: FileTask, progress: Callable[[int], Any] | None = None):
        """Execute single file in parallel processes."""
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpu_count)

        if os.path.exists(subtask.output_path):
            os.remove(subtask.output_path)
        os.makedirs(os.path.dirname(subtask.output_path), exist_ok=True)

        block_duration = subtask.block_duration
        with AudioFile(subtask.input_path, "r", block_duration=block_duration) as input_file:
            with AudioFile(subtask.output_path, "w", rate=input_file.rate) as output_file:
                for result_block in pool.imap(subtask.transform, input_file, chunksize=1):
                    output_file.write(result_block)
                    if progress is not None:
                        progress(len(result_block))

    @staticmethod
    def stats(task: TaskSpec) -> TaskStats:
        """Collect task stats."""
        stats = TaskStats()
        for path in TaskExecutor._input_paths(task):
            stats.total_files += 1
            stats.total_size += path.stat().st_size
        return stats
