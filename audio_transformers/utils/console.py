import abc
import json
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

    FORMATTERS: Mapping[Format, Formatter] = MappingProxyType(
        {
            "json": JsonFormatter(),
            "yaml": YamlFormatter(),
            "table": TableFormatter(),
        }
    )

    @staticmethod
    def dumps(data: Sequence[Any], format: Format) -> str:
        """Convert to string ready for console output."""
        formatter = Console.FORMATTERS.get(format)
        if formatter is None:
            raise ValueError(f"Unknown formatter: {format}")
        return formatter.dumps(data)

    @staticmethod
    def output(data: Sequence[Any], format: Format = "table", file: TextIO | None = None):
        """Output data collection."""
        return print(Console.dumps(data, format), file=file)

    @staticmethod
    def error(message: str, prefix: str = "ERROR:", end: str = "\n", file: TextIO | None = None):
        """Print error message."""
        print(colored(prefix, "red", attrs=["bold"]), message, end=end, file=file)

    @staticmethod
    def fatal(message: str, end: str = "\n", file: TextIO | None = None):
        """Print fatal error message."""
        Console.error(message, prefix="FATAL:", end=end, file=file)

    @staticmethod
    def warning(message: str, prefix: str = "WARNING:", end: str = "\n", file: TextIO | None = None):
        """Print warning message."""
        print(colored(prefix, "yellow", attrs=["bold"]), message, end=end, file=file)

    @staticmethod
    def ok(message: str, prefix: str = "OK:", end: str = "\n", file: TextIO | None = None):
        """Print OK message."""
        print(colored(prefix, "green", attrs=["bold"]), message, end=end, file=file)
