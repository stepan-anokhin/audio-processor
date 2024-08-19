import logging
import os
import tarfile

logger = logging.getLogger(__name__)


def extract_all(archive_path: str, destination: str):
    logger.info(f"Extracting '{archive_path}' to '{destination}'")
    filename = os.path.basename(archive_path)
    if filename.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(path=destination)
    elif filename.endswith(".tar"):
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(path=destination)
