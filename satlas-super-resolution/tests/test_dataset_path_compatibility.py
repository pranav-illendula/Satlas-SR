"""
Focused unit tests for Windows vs POSIX path compatibility in S2NAIPDataset logic.
Tests both forward-slash and backslash path conventions using Python's standard unittest.
"""
import os
import unittest
from pathlib import Path


def extract_chip_name_from_naip_path_legacy(path_str: str) -> str:
    """Legacy code assumption: assumes POSIX forward slash."""
    split_path = path_str.split('/')
    return split_path[-2]


def extract_chip_name_from_naip_path_robust(path_str: str) -> str:
    """Cross-platform robust implementation using pathlib / os.path."""
    norm = os.path.normpath(path_str)
    return Path(norm).parent.name


def extract_chip_stem_robust(path_str: str) -> str:
    """Cross-platform extraction of filename stem without extension."""
    norm = os.path.normpath(path_str)
    return Path(norm).stem


class TestDatasetPathCompatibility(unittest.TestCase):

    def test_posix_path_chip_extraction(self):
        posix_path = "/data/small_val_set/naip/20771_46525/rgb.png"
        self.assertEqual(extract_chip_name_from_naip_path_robust(posix_path), "20771_46525")
        self.assertEqual(extract_chip_stem_robust(posix_path), "rgb")

    def test_windows_path_chip_extraction(self):
        win_path = r"C:\Users\CSE-SDPL\Desktop\DL_image\data\small_val_set\naip\20771_46525\rgb.png"
        # Legacy fails on Windows path with IndexError
        with self.assertRaises(IndexError):
            extract_chip_name_from_naip_path_legacy(win_path)
        
        # Robust implementation passes
        self.assertEqual(extract_chip_name_from_naip_path_robust(win_path), "20771_46525")
        self.assertEqual(extract_chip_stem_robust(win_path), "rgb")

    def test_mixed_separator_path_extraction(self):
        mixed_path = "C:/Users/CSE-SDPL/Desktop\\DL_image/data/small_val_set\\naip/38649_48273/rgb.png"
        self.assertEqual(extract_chip_name_from_naip_path_robust(mixed_path), "38649_48273")


if __name__ == "__main__":
    unittest.main()
