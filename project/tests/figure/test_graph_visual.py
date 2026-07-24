import unittest
import math

from PIL import Image, ImageDraw

from src.analysis.figure.graph_visual import analyze_graph_visual


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


if __name__ == "__main__":
    unittest.main()
