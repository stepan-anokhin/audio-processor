from audio_transformers.utils.docs import Docs


class InitError(Exception):
    """Indicates Transform initialization error."""

    name: str
    docs: Docs | None

    def __init__(self, message: str, name: str, docs: Docs | None):
        super().__init__(message)
        self.name = name
        self.docs = docs


class TaskExecutionError(Exception):
    """Indicates total number of failed subtasks exceeds threshold."""
