from typing import Sequence

from audio_transformers.core.model import Signal
from audio_transformers.core.transform import Transform


class Composite(Transform):
    """Composite transformation which applies underlying sequence of transformations to the signal one by one."""

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms: Sequence[Transform] = tuple(transforms)

    @property
    def uniform(self) -> bool:
        """Check if composite transformation is uniform."""
        return all(t.uniform for t in self.transforms)

    def __call__(self, signal: Signal) -> Signal:
        """Apply transformations in sequence."""
        for transform in self.transforms:
            signal = transform(signal)
        return signal
