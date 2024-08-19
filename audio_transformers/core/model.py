from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Signal:
    """Sample data combined with a sampling rate.

    For many transformations we have to specify sampling rate along with the
    sampled data. In practice, we have to pass around the rate every time
    we pass sampled data. So sampled data alone is incomplete and this is
    a good idea to combine the two.

    Signal data must have shape=(n_channels, n_samples) and dtype=float32.
    """

    data: NDArray[np.float32]
    rate: int

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self.data.shape[0]

    @property
    def samples(self) -> int:
        """Get number of samples in signal data."""
        return self.data.shape[-1]

    @property
    def duration(self) -> float:
        """Get signal duration in seconds."""
        return self.samples / self.rate

    def concatenate(self, other: "Signal") -> "Signal":
        """Concatenate two signals."""
        if other.rate != self.rate:
            raise ValueError(f"Incompatible sampling rate: {other.rate} != {self.rate}")
        if other.channels != self.channels:
            raise ValueError(f"Incompatible number of channels: {other.channels} != {self.channels}")
        return Signal(np.concatenate([self.data, other.data], axis=1), self.rate)

    def stack(self, other: "Signal") -> "Signal":
        """Combine signal channels."""
        if other.rate != self.rate:
            raise ValueError(f"Incompatible sampling rate: {other.rate} != {self.rate}")
        if other.samples != self.samples:
            raise ValueError(f"Non equal number of samples: {other.samples} != {self.samples}")
        return Signal(np.concatenate([self.data, other.data], axis=0), self.rate)

    def __add__(self, other: "Signal") -> "Signal":
        """Concatenate two signals."""
        return self.concatenate(other)

    def __len__(self) -> int:
        """Get signal samples."""
        return self.samples

    def __getitem__(self, slice_spec):
        """Shortcut to slice channel."""
        return Signal(self.data[:, slice_spec], self.rate)
