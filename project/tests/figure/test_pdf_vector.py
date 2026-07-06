import tempfile
import unittest
from pathlib import Path

import fitz

from src.analysis.figure.pdf_vector import (
    analyze_pdf_vector_figure,
    infer_axis_labels,
    summarize_path_trend,
)


class PDFVectorFigureTests(unittest.TestCase):
    def _make_pdf(self, directory: str) -> Path:
        path = Path(directory) / "chart.pdf"
        document = fitz.open()
        page = document.new_page(width=300, height=220)
        page.insert_text((260, 205), "Time", fontsize=10)
        page.insert_text((10, 20), "Distance", fontsize=10)
        page.insert_text((200, 35), "A", fontsize=10)
        page.insert_text((245, 55), "B", fontsize=10)
        page.draw_polyline([(30, 190), (200, 30)], color=(0.0, 0.7, 0.2), width=1)
        page.draw_polyline([(30, 190), (80, 80), (190, 80), (245, 45)], color=(1.0, 0.5, 0.1), width=1)
        document.save(path)
        document.close()
        return path

    def test_axis_labels_are_selected_by_pdf_position(self):
        words = [
            {"text": "시간", "x": 0.9, "y": 0.9},
            {"text": "거리", "x": 0.1, "y": 0.1},
            {"text": "A", "x": 0.7, "y": 0.2},
        ]
        self.assertEqual(infer_axis_labels(words), ("시간", "거리"))

    def test_trend_summary_preserves_increase_and_flat_segments(self):
        points = [
            {"x": 0.0, "y": 0.0},
            {"x": 0.2, "y": 0.5},
            {"x": 0.7, "y": 0.5},
            {"x": 1.0, "y": 0.8},
        ]
        self.assertEqual(summarize_path_trend(points), ["증가", "일정", "증가"])

    def test_synthetic_vector_chart_generates_context_free_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._make_pdf(tmp)
            output = analyze_pdf_vector_figure(
                pdf_path=pdf_path,
                page_number=1,
                bbox=[0, 0, 300, 220],
                dpi=72,
                block_id="p1_b1",
                detection_score=0.9,
            )

        record = output["record"]
        self.assertEqual(record["analysis"]["result"]["figure_type"], "line_chart")
        self.assertEqual(record["analysis"]["result"]["x_axis"]["label"], "Time")
        self.assertEqual(record["analysis"]["result"]["y_axis"]["label"], "Distance")
        self.assertEqual(len(record["analysis"]["result"]["series"]), 2)
        self.assertFalse(record["description"]["context_used"])
        self.assertIn("증가", record["description"]["long_text"])
        self.assertIn("일정하게 유지", record["description"]["long_text"])


if __name__ == "__main__":
    unittest.main()
