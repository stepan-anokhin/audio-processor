from contextlib import AbstractContextManager
from functools import cached_property
from os import PathLike, fspath
from typing import Iterator, TypeAlias, Literal

import ffmpegio
from ffmpegio.streams import SimpleAudioWriter, SimpleAudioReader

import audio_transformers.io.format as format
import audio_transformers.io.probe as probe
from audio_transformers.core.model import Signal

Mode: TypeAlias = Literal["r", "w"]


class AudioFile(AbstractContextManager):
    """Audio file io."""

    path: str
    mode: Mode
    rate: int
    block_duration: float | None
    block_size: int | None
    _file: SimpleAudioReader | SimpleAudioWriter

    def __init__(
        self,
        path: PathLike | str,
        mode: Mode = "r",
        rate: int | None = None,
        block_duration: float | None = None,
        block_size: int | None = None,
    ):
        """
        :param path: audio file path.
        """

        self.path: str = fspath(path)
        self.mode: Mode = mode
        self._init_rate(rate)
        self._init_block(block_duration, block_size)
        self._init_file()

    def __enter__(self) -> "AudioFile":
        return self

    def __exit__(self, __exc_type, __exc_value, __traceback):
        self._file.close()

    def _init_rate(self, rate: int | None):
        if self.mode == "r":
            if rate is not None:
                raise ValueError("Cannot explicitly specify sampling rate in read mode.")
            self.rate = probe.rate(self.path)
        else:
            if rate is None:
                raise ValueError("Sampling rate must be specified in write mode.")
            self.rate = rate

    def _init_block(self, block_duration: float | None = None, block_size: int | None = None):
        """Resolve block duration and block size."""
        if self.mode == "w":
            if block_duration is not None:
                raise ValueError("Block duration cannot be specified in write mode.")
            if block_size is not None:
                raise ValueError("Block size cannot be specified in write mode.")
            self.block_duration = None
            self.block_size = None
        else:  # read mode
            if block_duration is not None and block_size is not None:
                raise ValueError("Either block_duration xor block_size could be specified.")
            if block_duration is None and block_size is None:
                self.block_duration = 60.0  # 1 minute
                self.block_size = int(self.block_duration * self.rate)
            elif block_duration is not None:
                self.block_duration = block_duration
                self.block_size = int(self.rate * block_duration)
            else:  # block_size is not None:
                self.block_duration = block_size / self.rate
                self.block_size = block_size

    def _init_file(self):
        """Initialize file."""
        if self.mode == "r":
            self._file = ffmpegio.open(self.path, "ra", blocksize=self.block_size, sample_fmt="flt")
        else:  # write mode
            self._file = ffmpegio.open(self.path, "wa", rate_in=self.rate, sample_fmt="flt")

    @cached_property
    def duration(self) -> float:
        """Get file duration."""
        if self.mode == "w":
            raise NotImplementedError("Duration is not implemented in write mode.")
        return probe.duration(self.path)

    @cached_property
    def samples(self) -> int:
        """Get total approximate samples."""
        if self.mode == "w":
            raise NotImplementedError("Samples count is not available in write mode.")
        return int(self.duration * self.rate)

    def read(self, n: int = -1) -> Signal:
        """Read entire file."""
        data = self._file.read(n)
        return format.to_signal(data, self.rate)

    def __iter__(self) -> Iterator[Signal]:
        """Iterate over blocks as signals."""
        for block in self._file:
            yield format.to_signal(block, self.rate)

    def write(self, signal: Signal) -> int:
        """Write signal object to the file."""
        return self._file.write(format.from_signal(signal))
