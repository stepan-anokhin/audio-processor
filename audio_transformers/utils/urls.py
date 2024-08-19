import os
from urllib.parse import urlparse


def filename(url: str) -> str:
    """Get default filename from URL."""
    parsed = urlparse(url)
    return os.path.basename(parsed.path)
