import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.analysis.formula.formula_analyzer import (
    analyze_formula_blocks,
    parse_formula_text,
    normalize_formula_text,
    crop_formula_block,
)

from src.analysis.formula.formula_recognizer import (
    recognize_formula_from_crop,
    convert_latex_to_mathml,
    generate_formula_description,
    is_reliable_model_latex,
    extract_formula_from_model_latex,
)

class TestFormulaAnalyzer(unittest.TestCase):
    def test_extract_formula_from_noisy_pix2tex_output(self):
        noisy_positive_fraction = (
            r"\begin{array}{c}{{\sim\uparrow}}\\ "
            r"{{(1)\:y=\frac{8}{x}}}\end{array}"
        )

        noisy_negative_fraction = (
            r"(\mathbf{\partial}_{(2)}\bigcup_{y=-{\frac{8}{x}}}"
        )

        noisy_other_fraction = r"\bigcup_{y=-{\frac{3}{t}}}"

        self.assertEqual(
            extract_formula_from_model_latex(noisy_other_fraction),
            r"y=-\frac{3}{t}",
        )

        self.assertEqual(
            extract_formula_from_model_latex(noisy_positive_fraction),
            r"y=\frac{8}{x}",
        )

        self.assertEqual(
            extract_formula_from_model_latex(noisy_negative_fraction),
            r"y=-\frac{8}{x}",
        )

    def test_analyze_formula_blocks_returns_only_formula_results(self):
        page = {
            "page_id": 9,
            "blocks": [
                {
                    "block_id": "p9_b1",
                    "type": "paragraph",
                    "bbox": [10, 10, 200, 50],
                    "text": "정비례 관계를 알아보자.",
                    "score": 0.95,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p9_b2",
                    "type": "formula",
                    "bbox": [20, 80, 180, 120],
                    "text": "y = ax",
                    "score": 0.91,
                    "detector": "doclayout_yolo",
                },
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["schema_version"], "1.0.0")
        self.assertEqual(result["page_id"], 9)
        self.assertEqual(result["block_id"], "p9_b2")
        self.assertEqual(result["type"], "formula")
        self.assertEqual(result["bbox"], [20, 80, 180, 120])

        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["analysis"]["result"]["kind"], "formula")
        self.assertEqual(result["analysis"]["result"]["latex"], "y=ax")
        self.assertEqual(result["analysis"]["result"]["plain_text"], "y = ax")

        self.assertEqual(result["context"]["previous_block_id"], "p9_b1")
        self.assertIsNone(result["context"]["next_block_id"])

    def test_formula_without_text_returns_partial(self):
        page = {
            "page_id": 10,
            "blocks": [
                {
                    "block_id": "p10_b1",
                    "type": "formula",
                    "bbox": [50, 100, 250, 140],
                    "text": "",
                    "score": 0.8,
                    "detector": "doclayout_yolo",
                }
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["analysis"]["status"], "partial")
        self.assertIsNone(results[0]["analysis"]["result"]["latex"])
        self.assertGreater(len(results[0]["warnings"]), 0)

    def test_parse_formula_with_condition_text(self):
        page = {
            "page_id": 9,
            "blocks": [
                {
                    "block_id": "p9_b2",
                    "type": "paragraph",
                    "bbox": [286, 168, 875, 195],
                    "text": "일반적으로 x와 y가 정비례할 때, x와 y 사이에는 다음과 같은 식이 성립한다.",
                    "score": 0.856,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p9_b3",
                    "type": "formula",
                    "bbox": [469, 213, 683, 240],
                    "text": "y=ax (단, a는 0이 아니다.)",
                    "score": 0.584,
                    "detector": "doclayout_yolo",
                },
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["page_id"], 9)
        self.assertEqual(result["block_id"], "p9_b3")
        self.assertEqual(result["type"], "formula")
        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["analysis"]["result"]["latex"], "y=ax")
        self.assertEqual(
            result["analysis"]["result"]["plain_text"],
            "y=ax (단, a는 0이 아니다.)",
        )

    def test_crop_formula_block_creates_image_file(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed.")

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            page_image_path = temp_dir_path / "page9.png"

            image = Image.new("RGB", (100, 100), "white")
            image.save(page_image_path)

            block = {
                "block_id": "p9_b3",
                "type": "formula",
                "bbox": [10, 20, 60, 50],
                "text": "y=ax",
            }

            crop_path = crop_formula_block(
                page_image_path=str(page_image_path),
                block=block,
                page_id=9,
            )

            self.assertIsNotNone(crop_path)

            crop_file = Path(crop_path)
            self.assertTrue(crop_file.exists())

            with Image.open(crop_file) as cropped:
                self.assertEqual(cropped.size, (90, 80))
            crop_file.unlink(missing_ok=True)

    def test_recognize_formula_from_crop_uses_fallback_text(self):
        result = recognize_formula_from_crop(
            crop_path=None,
            fallback_text="y=ax (단, a는 0이 아니다.)",
        )

        self.assertEqual(result["latex"], "y=ax")
        self.assertIsNotNone(result["mathml"])
        self.assertIn("<math>", result["mathml"])
        self.assertIn("</math>", result["mathml"])        
        self.assertEqual(result["plain_text"], "y=ax (단, a는 0이 아니다.)")
        self.assertEqual(result["model"]["name"], "formula-recognizer-fallback")
        self.assertGreater(len(result["warnings"]), 0)
    
    def test_recognize_formula_rejects_non_formula_list(self):
        result = recognize_formula_from_crop(
            crop_path=None,
            fallback_text="⑴ -2, -1, 0, 1, 2\n⑵ 수 전체",
        )

        self.assertIsNone(result["latex"])
        self.assertIsNone(result["mathml"])
        self.assertEqual(result["plain_text"], "⑴ -2, -1, 0, 1, 2 ⑵ 수 전체")
        self.assertGreater(len(result["warnings"]), 0)
        self.assertIn(
            "Detected formula block does not contain a formula-like expression.",
            result["warnings"],
        )

    def test_recognize_formula_rejects_axis_or_unit_labels(self):
        cases = [
            "x(m/s)",
            "y(원)",
            "y(km)",
            "시간(초)",
            "거리(km)",
            "(m/s)",
            "(km)",
        ]

        for input_text in cases:
            with self.subTest(input_text=input_text):
                result = recognize_formula_from_crop(
                    crop_path=None,
                    fallback_text=input_text,
                )

                self.assertEqual(result["latex"], None)
                self.assertEqual(result["mathml"], None)
                self.assertIn(
                    "Detected formula block does not contain a formula-like expression.",
                    result["warnings"],
                )

    def test_recognize_formula_rejects_table_like_noise(self):
        cases = [
            "x(m/s) 10 20 30 40 50 60 y(초)",
            "x 10 20 30 40 50 60 y 480 240 160 120 96 80",
            "0 1 2 3 4 5 6 x(톤) 0 y(만 원)",
        ]

        for input_text in cases:
            with self.subTest(input_text=input_text):
                result = recognize_formula_from_crop(
                    crop_path=None,
                    fallback_text=input_text,
                )

                self.assertEqual(result["latex"], None)
                self.assertEqual(result["mathml"], None)
                self.assertIn(
                    "Detected formula block does not contain a formula-like expression.",
                    result["warnings"],
                )

    def test_recognize_formula_common_textbook_patterns(self):
        cases = [
            ("y=2x", "y=2x"),
            ("y=-2x", "y=-2x"),
            ("y=ax (단, a는 0이 아니다.)", "y=ax"),
            ("a>0", "a>0"),
            ("a<0", "a<0"),
            ("y=6/x", r"y=\frac{6}{x}"),
            ("y = 4 / x", r"y=\frac{4}{x}"),
            ("y=;2!;x", r"y=\frac{1}{2}x"),
            ("y=;3@;x", r"y=\frac{2}{3}x"),
            ("(1, a)", "(1,a)"),
            ("(x, y)", "(x,y)"),
            ("(-2, -4)", "(-2,-4)"),
            ("⑴ y=4x", "y=4x"),
            ("⑵ y=-3x", "y=-3x"),
            ("식 y=800x", "y=800x"),
            ("⑴ y=4x ⑵ y=-3x", "y=4x;y=-3x"),
            ("(1) y=4x (2) y=-3x", "y=4x;y=-3x"),
            ("⑴ y=8/x ⑵ y=-8/x", r"y=\frac{8}{x};y=-\frac{8}{x}"),
            ("y=ax의 그래프", "y=ax"),
            ("y=ax`", "y=ax"),
            ("y=a/x", r"y=\frac{a}{x}"),
            ("y = a / x", r"y=\frac{a}{x}"),
            ("y=a÷x", r"y=\frac{a}{x}"),
            ("y=12/5", r"y=\frac{12}{5}"),
            ("y=-a/x", r"y=-\frac{a}{x}"),
            ("⑴ y=a/x ⑵ y=-a/x", r"y=\frac{a}{x};y=-\frac{a}{x}"),
            ("y=ax", "y=ax"),
            ("y= a\nx`(단, a는 0이 아니다.)", r"y=\frac{a}{x}"),
            ("y= 8\nx", r"y=\frac{8}{x}"),
            ("y= -8\nx", r"y=-\frac{8}{x}"),
            ("y=ax", "y=ax"),
            ("⑴ y= 8\nx ⑵ y= -8\nx", r"y=\frac{8}{x};y=-\frac{8}{x}"),
            ("(1) y= 8\nx (2) y= -8\nx", r"y=\frac{8}{x};y=-\frac{8}{x}"),
            ("⑴ y=4x\n⑵ y=- 2\nx", r"y=4x;y=-\frac{2}{x}"),
            ("(1) y=4x\n(2) y=- 2\nx", r"y=4x;y=-\frac{2}{x}"),
            ("⑴ y=4x ⑵ y=-2/x", r"y=4x;y=-\frac{2}{x}"),
        ]

        for input_text, expected_latex in cases:
            with self.subTest(input_text=input_text):
                result = recognize_formula_from_crop(
                    crop_path=None,
                    fallback_text=input_text,
                )

                self.assertEqual(result["latex"], expected_latex)
                self.assertTrue(
                    result["mathml"] is None or result["mathml"].startswith("<math>")
                )    
    def test_convert_latex_to_mathml_for_common_patterns(self):
        cases = [
            "y=ax",
            "y=4x",
            "y=-3x",
            "y=8/x",
            "(1,a)",
            "(-2,-4)",
            "y=4x;y=-3x",
            r"y=\frac{1}{2}x",
            r"y=\frac{2}{3}x",
            r"y=\frac{8}{x}",
            r"y=-\frac{8}{x}",
            r"y=\frac{12}{5}",
            r"y=\frac{a}{x}",
            r"y=-\frac{a}{x}",
            r"y=\frac{12}{5}",
        ]

        for latex in cases:
            with self.subTest(latex=latex):
                mathml = convert_latex_to_mathml(latex)

                self.assertIsNotNone(mathml)
                self.assertIn("<math>", mathml)
                self.assertIn("</math>", mathml)

    def test_normalize_formula_text(self):
        self.assertEqual(normalize_formula_text(" y = 2 × x "), r"y=2\timesx")
        self.assertEqual(normalize_formula_text("y ÷ x"), r"y\divx")
        self.assertEqual(normalize_formula_text(""), None)
        self.assertEqual(normalize_formula_text(None), None)
    
    def test_generate_formula_description(self):
        result = generate_formula_description("y=4x")

        self.assertEqual(result["status"], "generated")
        self.assertIn("y", result["short_text"])
        self.assertIn("x", result["short_text"])
        self.assertEqual(result["review_status"], "auto")

        empty_result = generate_formula_description(None)

        self.assertEqual(empty_result["status"], "not_started")
        self.assertIsNone(empty_result["short_text"])
        fraction_result = generate_formula_description(r"y=\frac{8}{x}")

        self.assertEqual(fraction_result["status"], "generated")
        self.assertIn("반비례", fraction_result["short_text"])
        self.assertIn("분자와 분모", fraction_result["transcription_notes"])

        negative_fraction_result = generate_formula_description(r"y=-\frac{8}{x}")

        self.assertEqual(negative_fraction_result["status"], "generated")
        self.assertIn("반비례", negative_fraction_result["short_text"])
        self.assertIn("-8", negative_fraction_result["short_text"])

    def test_rejects_unreliable_pix2tex_output(self):
        bad_latex = (
            r"y{\stackrel{\ldots}{=}}d{\bf{x}}"
            r"\operatorname{c}\mathsf{cl}\scriptscriptstyle"
        )

        self.assertFalse(is_reliable_model_latex(bad_latex))
        self.assertTrue(is_reliable_model_latex("y=ax"))
        self.assertTrue(is_reliable_model_latex(r"y=\frac{1}{2}x"))
        
    @patch("src.analysis.formula.formula_recognizer.recognize_with_optional_pix2tex")
    def test_fallback_used_when_text_has_more_formula_parts_than_pix2tex(self, mock_pix2tex):
        mock_pix2tex.return_value = r"y=\frac{8}{x}"

        result = recognize_formula_from_crop(
            "fake_crop.png",
            "⑴ y= 8\nx\n⑵ y=- 8\nx",
        )

        self.assertEqual(result["latex"], r"y=\frac{8}{x};y=-\frac{8}{x}")
        self.assertEqual(result["model"]["name"], "formula-recognizer-fallback")
        self.assertTrue(
            any("fewer formula parts" in warning for warning in result["warnings"])
        )

    @patch("src.analysis.formula.formula_recognizer.recognize_with_optional_pix2tex")
    def test_warning_when_pix2tex_output_is_rejected(self, mock_pix2tex):
        mock_pix2tex.return_value = r"\stackrel{\operatorname{noise}}{\Phi}"

        result = recognize_formula_from_crop("fake_crop.png", "y=4x")

        self.assertEqual(result["latex"], "y=4x")
        self.assertEqual(result["model"]["name"], "formula-recognizer-fallback")
        self.assertTrue(
            any("rejected as unreliable" in warning for warning in result["warnings"])
        )
    
    @patch("src.analysis.formula.formula_recognizer.recognize_with_optional_pix2tex")
    def test_warning_when_pix2tex_is_unavailable_with_crop(self, mock_pix2tex):
        mock_pix2tex.return_value = None

        result = recognize_formula_from_crop("fake_crop.png", "y=4x")

        self.assertEqual(result["latex"], "y=4x")
        self.assertEqual(result["model"]["name"], "formula-recognizer-fallback")
        self.assertTrue(
            any("unavailable or failed" in warning for warning in result["warnings"])
        )

if __name__ == "__main__":
    unittest.main()