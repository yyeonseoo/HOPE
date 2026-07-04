import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.analysis.table.crop import crop_table_image


class CropTableImageTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self._tmp_dir.name) / "page.png"
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100] = (255, 0, 0)
        image[:, 100:] = (0, 255, 0)
        cv2.imwrite(str(self.image_path), image)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_crops_within_bounds(self):
        crop = crop_table_image(self.image_path, [10, 10, 50, 50])
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (40, 40))

    def test_clamps_bbox_exceeding_image_bounds(self):
        crop = crop_table_image(self.image_path, [150, -20, 500, 500])
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (100, 50))

    def test_returns_none_for_empty_bbox(self):
        self.assertIsNone(crop_table_image(self.image_path, [10, 10, 10, 10]))
        self.assertIsNone(crop_table_image(self.image_path, [300, 300, 400, 400]))

    def test_returns_none_for_missing_image(self):
        missing = Path(self._tmp_dir.name) / "does_not_exist.png"
        self.assertIsNone(crop_table_image(missing, [0, 0, 10, 10]))


if __name__ == "__main__":
    unittest.main()
