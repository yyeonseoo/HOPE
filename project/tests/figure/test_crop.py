import tempfile
import unittest
from pathlib import Path

import fitz
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

    def test_caption_crop_is_rerendered_without_raising_page_layout_dpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "page.pdf"
            document = fitz.open()
            document.new_page(width=72, height=72)
            document.save(pdf_path)
            document.close()
            image_path = Path(tmp) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            block = {"block_id": "p1_b1", "bbox": [0, 0, 100, 100]}

            crop_path = crop_and_save_figure_block(
                image_path,
                block,
                1,
                Path(tmp) / "crops",
                pdf_path=pdf_path,
                source_dpi=100,
                caption_dpi=200,
            )

            with Image.open(crop_path) as crop:
                self.assertEqual(crop.size, (200, 200))


if __name__ == "__main__":
    unittest.main()
