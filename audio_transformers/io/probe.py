import os
import re
from os import PathLike, fspath
from subprocess import PIPE

import ffmpegio
from ffmpegio import ffprobe

DURATION_PATTERN = re.compile(r"Duration:\s+(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)")


def duration(path: PathLike | str) -> float:
    """Get file duration."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    probe = ffprobe(f"-i {fspath(path)}", stderr=PIPE, universal_newlines=True)
    if probe.returncode != 0:
        raise IOError(f"Cannot determine file duration: {path}. FFProbe stderr:\n{probe.stderr}")
    match = DURATION_PATTERN.search(probe.stderr)
    if not match:
        raise IOError(f"Cannot determine file duration: {path}")
    return 3600 * float(match.group("hours")) + 60 * float(match.group("minutes")) + float(match.group("seconds"))


def rate(path: PathLike | str) -> int:
    """Get file sampling rate."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with ffmpegio.open(fspath(path), "ra", blocksize=1) as file:
        return file.rate
