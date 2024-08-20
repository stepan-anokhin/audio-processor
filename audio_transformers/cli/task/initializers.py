import abc
from abc import abstractmethod
from functools import cached_property
from typing import Callable, TypeAlias, Type, Mapping

from audio_transformers.core.transform import Transform
from audio_transformers.cli.task.model import TransformSpec
from audio_transformers.utils.docs import Docs


class Initializer(abc.ABC):
    """Abstract base class for transformation initializers.

    Transformation initializer takes TransformSpec and builds
    the corresponding transformation object.
    """

    @property
    @abstractmethod
    def docs(self) -> Docs:
        """Get transformation docs."""

    @abstractmethod
    def init(self, spec: TransformSpec, transformations: Mapping[str, "Initializer"]) -> Transform:
        """Initialize transformation from spec."""


TransformFactory: TypeAlias = Callable[[...], Transform] | Type[Transform]


class BasicInit(Initializer):
    """Basic transformation initializer for the case when all parameters are basic values."""

    factory: TransformFactory

    def __init__(self, factory: TransformFactory):
        self.factory = factory

    @cached_property
    def docs(self) -> Docs:
        return Docs.from_func(self.factory)

    def init(self, spec: TransformSpec, transformations: Mapping[str, "Initializer"]) -> Transform:
        """Create the transformation from spec."""
        params = {}
        for param in self.docs.params:
            if param.name in spec.params:
                params[param.name] = spec.params[param.name]
        return self.factory(**params)
