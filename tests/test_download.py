import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.download import ensure_directories


class TestEnsureDirectories:
    def test_creates_directories(self, tmp_path):
        import utils.download as dl
        original_raw = dl.RAW_DIR
        original_processed = dl.PROCESSED_DIR
        try:
            dl.RAW_DIR = tmp_path / "data" / "raw"
            dl.PROCESSED_DIR = tmp_path / "data" / "processed"
            ensure_directories()
            assert dl.RAW_DIR.exists()
            assert dl.PROCESSED_DIR.exists()
        finally:
            dl.RAW_DIR = original_raw
            dl.PROCESSED_DIR = original_processed
