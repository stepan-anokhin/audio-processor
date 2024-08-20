import inspect
import re
from dataclasses import dataclass
from typing import Sequence, List

from audio_transformers.utils.console import Tabular
from audio_transformers.utils.types import BasicValue

DEFAULT_MAX_LENGTH: int = 60


@dataclass
class Param(Tabular):
    """Parameter descriptor."""

    @classmethod
    def headers(cls) -> Sequence[str]:
        return "Name", "Type", "Default", "Description"

    def table_row(self) -> Sequence[str]:
        return self.name, self.type, self.default, Docs.ellipsis(self.description)

    name: str
    type: str
    default: BasicValue
    description: str


@dataclass
class Docs:
    """Func docs reader."""

    brief: str
    full_docs: str
    params: Sequence[Param]

    @staticmethod
    def from_func(func) -> "Docs":
        """Get docs from function."""
        full_docs: str = func.__doc__ or ""
        if isinstance(func, type):
            full_docs += "\n" + (func.__init__.__doc__ or "")

        brief: str = full_docs.split("\n", maxsplit=1)[0]

        signature = inspect.signature(func)
        params: List[Param] = []
        for arg in signature.parameters.values():
            arg_type = "Any"
            if arg.annotation is not None:
                arg_type = arg.annotation.__name__
            default = ""
            if arg.default is not inspect.Parameter.empty:
                default = arg.default
                if default is None or isinstance(default, str):
                    default = repr(default)
            param = Param(
                name=arg.name,
                type=arg_type,
                default=default,
                description=Docs.param_doc(arg.name, full_docs),
            )
            params.append(param)
        return Docs(brief=brief, full_docs=full_docs, params=tuple(params))

    @staticmethod
    def param_doc(name: str, func_docs: str) -> str:
        """Get parameter docstring."""
        match = re.search(rf"^\s*:\s*param\s+{name}\s*:\s*(.*)\s*$", func_docs, re.MULTILINE)
        if match is None:
            return ""
        return match.group(1)

    @staticmethod
    def ellipsis(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
        """Get shortened text."""
        if len(text) <= max_length:
            return text
        return f"{text[:max_length - 3]}..."
