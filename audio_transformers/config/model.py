from dataclasses import dataclass, field
from typing import Dict, TypeAlias, List

import yaml
from dacite import from_dict

BasicValue: TypeAlias = int | str | float | bool | None


@dataclass
class TransformSpec:
    """Transformation spec as present in config file."""
    type: str
    params: Dict[str, BasicValue]


@dataclass
class ConfigFile:
    """Config file."""
    transforms: List[TransformSpec] = field(default_factory=list)

    @staticmethod
    def read(path: str) -> "ConfigFile":
        """Read transformation config from YAML file."""
        with open(path, 'r') as file:
            dict_data = yaml.safe_load(file)
            return from_dict(ConfigFile, dict_data)
