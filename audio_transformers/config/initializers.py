import abc
from abc import abstractmethod
from typing import Callable, TypeAlias, Type, Mapping

from audio_transformers.config.model import TransformSpec
from audio_transformers.core.transform import Transform


class Initializer(abc.ABC):
    """Abstract base class for transformation initializers.

    Transformation initializer takes TransformSpec and builds
    the corresponding transformation object.
    """

    @abstractmethod
    def init(self, spec: TransformSpec, transformations: Mapping[str, "Initializer"]) -> Transform:
        """Initialize transformation from spec."""


TransformFactory: TypeAlias = Callable[[...], Transform] | Type[Transform]


class BasicInit(Initializer):
    """Basic transformation initializer for the case when all parameters are basic values."""

    factory: TransformFactory

    def __init__(self, factory: TransformFactory):
        self.factory = factory

    def init(self, spec: TransformSpec, transformations: Mapping[str, "Initializer"]) -> Transform:
        """Create the transformation from spec."""
        return self.factory(**spec.params)
