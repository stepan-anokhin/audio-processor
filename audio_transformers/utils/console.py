import abc
import json
import sys
from abc import abstractmethod
from dataclasses import asdict
from io import StringIO
from types import MappingProxyType
from typing import Any, Sequence, Type, Mapping, TypeAlias, Literal, TextIO

import yaml
from tabulate import tabulate
from termcolor import colored


class Formatter(abc.ABC):
    """Base class for data formatters."""

    @abstractmethod
    def dumps(self, data: Sequence[Any]) -> str:
        """Convert stream of data items into string representation."""


class YamlFormatter(Formatter):
    """YAML output formatter."""

    def dumps(self, data: Sequence[Any]) -> str:
        dict_data = [asdict(item) for item in data]
        stream = StringIO()
        yaml.safe_dump(dict_data, stream)
        return stream.getvalue()


class JsonFormatter(Formatter):
    """JSON output formatter."""

    def dumps(self, data: Sequence[Any]) -> str:
        dict_data = [asdict(item) for item in data]
        return json.dumps(dict_data, indent=2, default=str)


class Tabular(abc.ABC):
    """Abstract parent for table-representable data item.

    Should be inherited by classes that want to integrate with Console tabular format.
    """

    @classmethod
    @abstractmethod
    def headers(cls) -> Sequence[str]:
        """Table headers."""

    @abstractmethod
    def table_row(self) -> Sequence[str]:
        """Get a table row."""


class TableFormatter(Formatter):
    """Table output formatter."""

    def dumps(self, data: Sequence[Tabular]) -> str:
        """Tabulate data items."""
        if len(data) == 0:
            return ""
        item_type: Type[Tabular] = type(data[0])
        if not issubclass(item_type, Tabular):
            raise ValueError(f"Cannot tabulate data type: {item_type.__name__}")
        headers = item_type.headers()
        rows = (item.table_row() for item in data)
        return tabulate(rows, headers, tablefmt="simple")


Format: TypeAlias = Literal["table", "json", "yaml"]


class Console:
    """Console output utils."""

    _formatters: Mapping[str, Formatter]
    _output_file: TextIO
    _errors_file: TextIO

    DEFAULT_FORMATTERS: Mapping[Format, Formatter] = MappingProxyType(
        {
            "json": JsonFormatter(),
            "yaml": YamlFormatter(),
            "table": TableFormatter(),
        }
    )

    def __init__(
        self,
        formatters: Mapping[str, Formatter] | None = None,
        output_file: TextIO | None = None,
        errors_file: TextIO | None = None,
    ):
        self._formatters = formatters or self.DEFAULT_FORMATTERS
        self._output_file = output_file or sys.stdout
        self._errors_file = errors_file or sys.stderr

    def dumps(self, data: Sequence[Any], format: Format) -> str:
        """Convert to string ready for console output."""
        formatter = self._formatters.get(format)
        if formatter is None:
            raise ValueError(f"Unknown formatter: {format}")
        return formatter.dumps(data)

    def output(self, data: Sequence[Any], format: Format = "table"):
        """Output data collection."""
        return print(self.dumps(data, format), file=self._output_file)

    def error(self, message: str, prefix: str = "ERROR:", end: str = "\n"):
        """Print error message."""
        print(colored(prefix, "red", attrs=["bold"]), message, end=end, file=self._errors_file)

    def fatal(self, message: str, end: str = "\n"):
        """Print fatal error message."""
        self.error(message, prefix="FATAL:", end=end)

    def warning(self, message: str, prefix: str = "WARNING:", end: str = "\n"):
        """Print warning message."""
        print(colored(prefix, "yellow", attrs=["bold"]), message, end=end, file=self._errors_file)

    def ok(self, message: str, prefix: str = "OK:", end: str = "\n"):
        """Print OK message."""
        print(colored(prefix, "green", attrs=["bold"]), message, end=end, file=self._output_file)
