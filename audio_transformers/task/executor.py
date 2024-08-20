from types import MappingProxyType
from typing import Sequence, List, Mapping

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
from audio_transformers.task.initializers import Initializer, BasicInit
from audio_transformers.task.model import TransformSpec

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


class TaskExecutor:
    """Reads and executes task config."""

    transforms: Mapping[str, Initializer]

    def __init__(self, transforms: Mapping[str, Initializer] | None):
        self.transforms = transforms or DEFAULT_TRANSFORMS

    def build_transform(self, specs: Sequence[TransformSpec]) -> Transform:
        """Build transformation from the spec list."""
        transforms: List[Transform] = []
        for spec in specs:
            if spec.type not in self.transforms:
                known = ", ".join(self.transforms.keys())
                raise ValueError(f"Unknown transformation: {spec.type}. Must be one of: {known}")
            initializer = self.transforms[spec.type]
            transforms.append(initializer.init(spec, self.transforms))
        return Composite(transforms)
