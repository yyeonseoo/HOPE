import unittest

from src.analysis.figure.graph_visual import GraphVisualCue
from src.analysis.figure.type_signals import extract_type_signals


class ExtractTypeSignalsTests(unittest.TestCase):
    def test_graph_extracts_axis_labels_and_trend(self):
        evidence = [
            {"text": "시간", "relative_bbox": [0.75, 0.8, 0.85, 0.86]},
            {"text": "거리", "relative_bbox": [0.1, 0.15, 0.2, 0.2]},
        ]
        cue = GraphVisualCue(state="plotted", trend="increasing", confidence=0.9)

        signals = extract_type_signals("graph", evidence, cue)

        self.assertEqual(signals.x_axis, "시간")
        self.assertEqual(signals.y_axis, "거리")
        self.assertEqual(signals.trend, "increasing")

    def test_graph_without_visual_cue_has_no_trend(self):
        signals = extract_type_signals("graph", [], None)
        self.assertIsNone(signals.trend)

    def test_graph_trend_only_used_when_state_is_plotted(self):
        cue = GraphVisualCue(state="uncertain", trend=None, confidence=0.0)
        signals = extract_type_signals("graph", [], cue)
        self.assertIsNone(signals.trend)

    def test_legend_excludes_axis_labels_and_pure_numbers(self):
        evidence = [
            {"text": "시간", "relative_bbox": [0.75, 0.8, 0.85, 0.86]},
            {"text": "5", "relative_bbox": [0.5, 0.5, 0.55, 0.55]},
            {"text": "A선", "relative_bbox": [0.5, 0.3, 0.55, 0.35]},
        ]
        signals = extract_type_signals("graph", evidence, None)

        self.assertNotIn("시간", signals.legend)
        self.assertNotIn("5", signals.legend)
        self.assertIn("A선", signals.legend)

    def test_diagram_extracts_components_not_relations(self):
        evidence = [{"text": "삼각형 ABC", "relative_bbox": [0.5, 0.5, 0.6, 0.55]}]
        signals = extract_type_signals("mathematical_diagram", evidence, None)

        self.assertIn("삼각형 ABC", signals.components)
        self.assertEqual(signals.relations, ())

    def test_illustration_extracts_objects_not_interactions(self):
        evidence = [{"text": "토끼", "relative_bbox": [0.1, 0.1, 0.2, 0.2]}]
        signals = extract_type_signals("illustration", evidence, None)

        self.assertIn("토끼", signals.objects)
        self.assertEqual(signals.interactions, ())

    def test_photo_never_fabricates_a_scene(self):
        signals = extract_type_signals("photo", [{"text": "표지판", "relative_bbox": [0.1, 0.1, 0.2, 0.2]}], None)

        self.assertIsNone(signals.scene)

    def test_unknown_figure_type_yields_empty_signals(self):
        signals = extract_type_signals("unknown", [{"text": "x"}], None)
        self.assertFalse(signals.has_any())

    def test_no_evidence_yields_empty_signals_not_fabricated_ones(self):
        signals = extract_type_signals("graph", None, None)
        self.assertFalse(signals.has_any())

    def test_to_dict_shape(self):
        signals = extract_type_signals("graph", [], None)
        self.assertEqual(
            set(signals.to_dict()),
            {"x_axis", "y_axis", "legend", "trend", "components", "relations", "objects", "interactions", "scene"},
        )


if __name__ == "__main__":
    unittest.main()
