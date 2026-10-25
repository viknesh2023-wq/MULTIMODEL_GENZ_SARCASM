"""Hash and media verification utilities."""
import hashlib
import os
from typing import Optional, Tuple
from PIL import Image


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Calculates the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_image_file(image_path: str) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """
    Verifies that an image file can be opened and decoded without corruption.
    Returns: (is_valid, error_msg, (width, height), format)
    """
    if not os.path.isfile(image_path):
        return False, "File does not exist", None, None
    try:
        with Image.open(image_path) as img:
            img.verify()
        # verify() closes image, reopen to get format and dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            fmt = img.format
            return True, None, (width, height), fmt
    except Exception as e:
        return False, str(e), None, None
