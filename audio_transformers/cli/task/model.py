from dataclasses import dataclass, field, asdict
from typing import Dict, List

import yaml
from dacite import from_dict

from audio_transformers.utils.types import BasicValue


@dataclass
class TransformSpec:
    """Transformation spec as present in config file."""

    type: str
    params: Dict[str, BasicValue]


@dataclass
class TaskSpec:
    """Transformation task specification."""

    input_root: str | None = None
    input_pattern: str | None = None

    output_root: str | None = None
    output_pattern: str | None = "{reldir}/{name}_aug.{ext}"

    transforms: List[TransformSpec] = field(default_factory=list)

    def __post_init__(self):
        if self.output_root is None:
            self.output_root = self.input_root

    @staticmethod
    def from_file(path: str) -> "TaskSpec":
        """Read transformation config from YAML file."""
        with open(path, "r") as file:
            dict_data = yaml.safe_load(file)
            return from_dict(TaskSpec, dict_data)

    @staticmethod
    def from_cli(
            name: str | None,
            input_root: str | None,
            input_pattern: str | None,
            output_root: str | None,
            output_pattern: str | None,
            config: str | None,
            **options,
    ) -> "TaskSpec":
        """Create task spec from CLI arguments."""
        spec: TaskSpec = TaskSpec()
        if config is not None:
            spec = TaskSpec.from_file(config)
        if name is not None:
            transform = TransformSpec(type=name, params=options)
            spec.transforms = [transform]
        if input_root is not None:
            spec.input_root = input_root
        if input_pattern is not None:
            spec.input_pattern = input_pattern
        if output_root is not None:
            spec.output_root = output_root
        if output_pattern is not None:
            spec.output_pattern = output_pattern
        if spec.output_root is None:
            spec.output_root = spec.input_root
        return spec

    def save(self, path: str):
        """Save to file."""
        with open(path, "w") as file:
            yaml.safe_dump(asdict(self), file)
