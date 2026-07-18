import unittest
import tempfile
import math
from pathlib import Path

from PIL import Image, ImageDraw

from src.analysis.figure.graph_visual import GraphVisualCue, analyze_graph_visual


class GraphVisualCueTests(unittest.TestCase):
    def _coordinate_plane(self):
        image = Image.new("RGB", (220, 220), "white")
        draw = ImageDraw.Draw(image)
        for position in range(20, 201, 20):
            draw.line((position, 10, position, 210), fill=(205, 205, 205), width=1)
            draw.line((10, position, 210, position), fill=(205, 205, 205), width=1)
        draw.line((110, 10, 110, 210), fill=(60, 60, 60), width=2)
        draw.line((10, 110, 210, 110), fill=(60, 60, 60), width=2)
        return image

    def test_empty_coordinate_plane(self):
        self.assertEqual(analyze_graph_visual(self._coordinate_plane()).state, "empty")

    def test_increasing_colored_line(self):
        image = self._coordinate_plane()
        ImageDraw.Draw(image).line((25, 190, 195, 30), fill=(230, 120, 30), width=4)
        cue = analyze_graph_visual(image)
        self.assertEqual((cue.state, cue.trend), ("plotted", "increasing"))
        self.assertEqual(cue.mark_type, "line")
        self.assertEqual(cue.path_shape, "straight_segments")

    def test_connected_polyline_is_detected_as_straight_segments(self):
        image = self._coordinate_plane()
        ImageDraw.Draw(image).line(
            [(25, 190), (85, 125), (150, 105), (195, 30)],
            fill=(35, 170, 85),
            width=4,
        )
        cue = analyze_graph_visual(image)
        self.assertEqual(cue.path_shape, "straight_segments")
        self.assertGreaterEqual(cue.bend_count, 1)

    def test_two_disconnected_curves_are_marked_as_multiple(self):
        image = self._coordinate_plane()
        draw = ImageDraw.Draw(image)
        draw.arc((20, 20, 120, 180), 260, 355, fill=(40, 170, 90), width=4)
        draw.arc((100, 40, 210, 200), 80, 175, fill=(40, 170, 90), width=4)
        cue = analyze_graph_visual(image)
        self.assertEqual(cue.state, "plotted")
        self.assertEqual(cue.mark_type, "multiple")

    def test_decreasing_colored_line(self):
        image = self._coordinate_plane()
        ImageDraw.Draw(image).line((25, 30, 195, 190), fill=(110, 60, 190), width=4)
        cue = analyze_graph_visual(image)
        self.assertEqual((cue.state, cue.trend), ("plotted", "decreasing"))

    def test_disconnected_scatter_points_are_not_treated_as_empty(self):
        image = self._coordinate_plane()
        draw = ImageDraw.Draw(image)
        for x, y in ((45, 175), (80, 145), (120, 110), (165, 70)):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(230, 80, 100))
        cue = analyze_graph_visual(image)
        self.assertEqual((cue.state, cue.trend), ("plotted", "increasing"))
        self.assertEqual(cue.mark_type, "points")

    def test_empty_plane_bypasses_language_model(self):
        from src.analysis.figure.captioners import Qwen3VLCaptioner

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.png"
            self._coordinate_plane().save(path)
            caption = Qwen3VLCaptioner(device="cpu").caption(path, "graph")

        self.assertEqual(caption.text, "x축과 y축, 격자가 표시된 빈 좌표평면이다.")
        self.assertEqual(caption.model_name, "opencv-ocr-grounded-graph-captioner")

    def test_coordinate_graph_uses_grounded_caption_without_loading_qwen(self):
        from src.analysis.figure.captioners import Qwen3VLCaptioner

        image = self._coordinate_plane()
        ImageDraw.Draw(image).line((25, 190, 195, 30), fill=(230, 120, 30), width=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.png"
            image.save(path)
            caption = Qwen3VLCaptioner(device="cpu").caption(
                path,
                "graph",
                evidence=["y=ax", "(1, a)"],
            )

        self.assertEqual(caption.model_name, "opencv-ocr-grounded-graph-captioner")
        self.assertIn("우상향", caption.text)
        self.assertIn("y=ax", caption.text)
        self.assertIn("(1, a) 표기", caption.text)
        self.assertNotIn("시작점", caption.text)
        self.assertNotIn("교점", caption.text)

    def _oscillating_graph(self, drift):
        image = self._coordinate_plane().resize((320, 220))
        draw = ImageDraw.Draw(image)
        points = [
            (x, int(110 + 35 * math.sin((x - 20) / 22) + drift * (x - 20)))
            for x in range(20, 301)
        ]
        draw.line(points, fill=(30, 140, 210), width=3)
        return image

    def test_oscillation_and_overall_trend_are_independent(self):
        level = analyze_graph_visual(self._oscillating_graph(0.0))
        falling = analyze_graph_visual(self._oscillating_graph(0.22))
        self.assertEqual((level.variation, level.trend), ("oscillating", "horizontal"))
        self.assertEqual((falling.variation, falling.trend), ("oscillating", "decreasing"))
        self.assertEqual(level.path_shape, "smooth_curve")
        self.assertGreaterEqual(len(level.direction_sequence), 3)

    def test_black_oscillating_curve_is_also_profiled(self):
        image = self._oscillating_graph(0.0)
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                red, green, blue = pixels[x, y]
                if blue > red + 50 and blue > green + 20:
                    pixels[x, y] = (30, 30, 30)
        cue = analyze_graph_visual(image)
        self.assertEqual((cue.state, cue.variation), ("plotted", "oscillating"))


class GraphTrendGroundingTests(unittest.TestCase):
    def test_trend_cue_replaces_conflicting_model_trend(self):
        from src.analysis.figure.captioners import _apply_graph_trend_grounding

        text = "직선은 우하향하며 감소한다. 점 (1, a)가 표시되어 있다."
        result = _apply_graph_trend_grounding(
            text,
            GraphVisualCue("plotted", "increasing", 0.96),
        )
        self.assertIn("우상향", result)
        self.assertNotIn("우하향", result)
        self.assertIn("(1, a)", result)

    def test_falling_oscillation_is_described_with_both_features(self):
        from src.analysis.figure.captioners import _graph_trend_lead

        cue = GraphVisualCue("plotted", "decreasing", 0.96, "oscillating", 4, "increasing")
        result = _graph_trend_lead(cue)
        self.assertIn("반복해서 오르내리는", result)
        self.assertIn("전체적으로", result)
        self.assertIn("낮아진다", result)

    def test_multiple_reciprocal_branches_are_not_replaced_by_single_trend(self):
        from src.analysis.figure.captioners import _apply_graph_trend_grounding

        text = "제1사분면과 제3사분면에 서로 마주 보는 두 갈래의 곡선이 있다."
        cue = GraphVisualCue(
            "plotted", "decreasing", 0.96, "monotonic",
            coordinate_plane=True, mark_type="multiple", series_count=2,
        )
        self.assertEqual(_apply_graph_trend_grounding(text, cue), text)

    def test_structured_point_result_uses_distribution_wording(self):
        from src.analysis.figure.captioners import _structured_graph_lead

        cue = GraphVisualCue("plotted", "increasing", 0.84, coordinate_plane=True)
        structure = {
            "arrangement": "single",
            "plots": [{"shape": "points", "dirs": ["up"], "net": "up", "x": None, "y": None, "name": None}],
        }
        result = _structured_graph_lead(cue, structure)
        self.assertIn("여러 점", result)
        self.assertIn("높게 분포", result)
        self.assertNotIn("직선", result)

    def test_structured_multiple_oscillation_keeps_overall_rise(self):
        from src.analysis.figure.captioners import _structured_graph_lead

        cue = GraphVisualCue("plotted", "horizontal", 0.96, "oscillating", coordinate_plane=True)
        structure = {
            "arrangement": "overlaid",
            "plots": [
                {"shape": "smooth_curve", "dirs": ["up", "down", "up"], "net": "up", "x": None, "y": None, "name": None},
                {"shape": "smooth_curve", "dirs": ["up", "down", "up"], "net": "up", "x": None, "y": None, "name": None},
            ],
        }
        result = _structured_graph_lead(cue, structure)
        self.assertIn("첫 번째 계열", result)
        self.assertIn("상승, 하강, 상승", result)
        self.assertIn("끝의 높이가 높다", result)

    def test_visible_reciprocal_equation_is_included_with_two_branches(self):
        from src.analysis.figure.captioners import _grounded_coordinate_graph_caption

        cue = GraphVisualCue(
            "plotted",
            "decreasing",
            0.96,
            "monotonic",
            coordinate_plane=True,
            mark_type="multiple",
            series_count=2,
        )
        structure = {
            "arrangement": "overlaid",
            "quadrants": [1, 3],
            "plots": [
                {"shape": "smooth_curve", "dirs": ["down"], "net": "down", "x": None, "y": None, "name": None},
                {"shape": "smooth_curve", "dirs": ["down"], "net": "down", "x": None, "y": None, "name": None},
            ],
        }
        result = _grounded_coordinate_graph_caption(cue, ["y=a/x", "(1, a)"], structure)
        self.assertIn("두 갈래", result)
        self.assertIn("제1사분면과 제3사분면", result)
        self.assertIn("y=a/x", result)
        self.assertIn("(1, a) 표기", result)

    def test_reciprocal_quadrants_are_omitted_when_not_confirmed(self):
        from src.analysis.figure.captioners import _grounded_coordinate_graph_caption

        cue = GraphVisualCue(
            "plotted", "decreasing", 0.96, "monotonic",
            coordinate_plane=True, mark_type="multiple", series_count=2,
        )
        structure = {
            "arrangement": "overlaid",
            "quadrants": [],
            "plots": [
                {"shape": "smooth_curve", "dirs": ["down"], "net": "down"},
                {"shape": "smooth_curve", "dirs": ["down"], "net": "down"},
            ],
        }
        result = _grounded_coordinate_graph_caption(cue, ["y=a/x"], structure)
        self.assertIn("두 갈래", result)
        self.assertNotIn("사분면", result)

    def test_context_formula_is_attributed_to_surrounding_text(self):
        from src.analysis.figure.captioners import _grounded_coordinate_graph_caption

        cue = GraphVisualCue(
            "plotted", "decreasing", 0.96, "monotonic",
            coordinate_plane=True, mark_type="multiple", series_count=2,
        )
        context = [{"block_id": "p14_b3", "type": "paragraph", "text": "함수 y=a/x의 그래프"}]
        result = _grounded_coordinate_graph_caption(cue, [], context=context)
        self.assertIn("두 갈래", result)
        self.assertIn("주변 설명에서는 y=a/x", result)
        self.assertNotIn("그림에는 y=a/x", result)

    def test_axis_label_is_not_repeated_as_series_name(self):
        from src.analysis.figure.captioners import _grounded_coordinate_graph_caption

        cue = GraphVisualCue("plotted", "increasing", 0.96, mark_type="line")
        evidence = [{"id": "x", "text": "시간"}, {"id": "y", "text": "높이"}]
        structure = {
            "arrangement": "single",
            "plots": [{
                "shape": "straight_segments", "dirs": ["up"], "net": "up",
                "x": "x", "y": "y", "name": "y",
            }],
        }
        result = _grounded_coordinate_graph_caption(cue, evidence, structure)
        self.assertNotIn("높이 계열", result)

    def test_same_direction_segments_keep_visible_bend(self):
        from src.analysis.figure.captioners import _structured_graph_lead

        cue = GraphVisualCue("plotted", "increasing", 0.96, mark_type="line", path_shape="straight_segments")
        structure = {
            "arrangement": "single",
            "plots": [{"shape": "straight_segments", "dirs": ["up", "up"], "net": "up", "bends": 1}],
        }
        result = _structured_graph_lead(cue, structure)
        self.assertIn("한 번 꺾인 선", result)

    def test_axis_labels_and_flat_net_change_are_verbalized(self):
        from src.analysis.figure.captioners import _grounded_coordinate_graph_caption

        cue = GraphVisualCue("plotted", "horizontal", 0.84, coordinate_plane=True)
        evidence = [
            {"id": "t1", "text": "온도", "bbox": [0, 0, 20, 20]},
            {"id": "t2", "text": "시간", "bbox": [80, 80, 110, 100]},
        ]
        structure = {
            "arrangement": "single",
            "plots": [{
                "shape": "smooth_curve",
                "dirs": ["down", "up", "down"],
                "net": "flat",
                "x": "t2",
                "y": "t1",
                "name": None,
            }],
        }
        result = _grounded_coordinate_graph_caption(cue, evidence, structure)
        self.assertIn("x축은 시간이고 y축은 온도", result)
        self.assertIn("시간에 따른 온도의 변화", result)
        self.assertIn("하강, 상승, 하강", result)
        self.assertIn("시작과 끝의 높이가 비슷", result)

    def test_piecewise_panels_are_described_individually_as_segments(self):
        from src.analysis.figure.captioners import _structured_graph_lead

        cue = GraphVisualCue("plotted", "horizontal", 0.84, coordinate_plane=True)
        structure = {
            "arrangement": "panels",
            "plots": [
                {"shape": "straight_segments", "dirs": ["up"], "net": "up", "x": None, "y": None, "name": None},
                {"shape": "straight_segments", "dirs": ["flat"], "net": "flat", "x": None, "y": None, "name": None},
                {"shape": "straight_segments", "dirs": ["up", "flat", "down"], "net": "flat", "x": None, "y": None, "name": None},
                {"shape": "straight_segments", "dirs": ["down", "flat", "down"], "net": "down", "x": None, "y": None, "name": None},
            ],
        }
        result = _structured_graph_lead(cue, structure)
        self.assertIn("첫 번째 그래프", result)
        self.assertIn("네 번째 그래프", result)
        self.assertIn("직선 조각", result)
        self.assertNotIn("곡선", result)

    def test_composite_diagrams_and_missing_panels_are_not_collapsed(self):
        from src.analysis.figure.captioners import _structured_graph_lead

        cue = GraphVisualCue("plotted", "increasing", 0.96, mark_type="multiple", series_count=3)
        structure = {
            "arrangement": "panels",
            "panel_count": 3,
            "context": {
                "kind": "solid_diagrams",
                "count": 3,
                "position": "above",
                "paired": True,
                "items": [
                    "위가 넓고 아래가 좁은 입체도형",
                    "위아래 폭이 다른 입체도형",
                    "원기둥 세 부분이 이어진 입체도형",
                ],
            },
            "plots": [
                {"shape": "smooth_curve", "dirs": ["up"], "net": "up", "x": None, "y": None, "name": None},
            ],
        }
        result = _structured_graph_lead(cue, structure)
        self.assertIn("첫 번째 입체도형", result)
        self.assertIn("세 번째 입체도형", result)
        self.assertIn("그래프가 3개", result)
        self.assertIn("순서대로 대응", result)
