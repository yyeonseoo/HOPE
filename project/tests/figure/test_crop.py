import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.analysis.figure.crop import crop_and_save_figure_block, crop_figure_image


class FigureCropTests(unittest.TestCase):
    def test_crop_is_clamped_to_page_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (100, 80), "white").save(image_path)

            crop = crop_figure_image(image_path, [-10, 10, 60, 100])

            self.assertIsNotNone(crop)
            self.assertEqual(crop.size, (60, 70))

    def test_crop_is_saved_with_deterministic_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            output_dir = Path(tmp) / "crops"
            Image.new("RGB", (100, 80), "white").save(image_path)
            block = {"block_id": "p4_b2", "bbox": [10, 20, 70, 60]}

            crop_path = crop_and_save_figure_block(image_path, block, 4, output_dir)

            self.assertEqual(Path(crop_path).name, "p4_p4_b2.png")
            with Image.open(crop_path) as crop:
                self.assertEqual(crop.size, (60, 40))

    def test_invalid_bbox_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            self.assertIsNone(crop_figure_image(image_path, [30, 20, 20, 50]))


if __name__ == "__main__":
    unittest.main()
