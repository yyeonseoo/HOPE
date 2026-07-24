import unittest

from src.analysis.figure.context_builder import FigureContext
from src.analysis.figure.grounding import (
    LOW_TOPIC_SIMILARITY_THRESHOLD,
    TopicGroundingScorer,
    compute_grounding_scores,
    find_topic_mismatch_warning,
)


class FakeScorer:
    def __init__(self, similarity):
        self.similarity = similarity
        self.calls = []

    def score(self, caption, context):
        self.calls.append((caption, context))
        return self.similarity


class FindTopicMismatchWarningTests(unittest.TestCase):
    def test_low_similarity_produces_a_warning(self):
        scorer = FakeScorer(LOW_TOPIC_SIMILARITY_THRESHOLD - 0.01)
        context = [{"block_id": "p1_b1", "type": "paragraph", "text": "x와 y가 정비례하는 관계"}]

        similarity, warnings = find_topic_mismatch_warning("도로 위의 자동차 사진이다.", context, scorer)

        self.assertEqual(similarity, LOW_TOPIC_SIMILARITY_THRESHOLD - 0.01)
        self.assertEqual(len(warnings), 1)
        self.assertIn("topical similarity", warnings[0])

    def test_similarity_at_or_above_threshold_produces_no_warning(self):
        scorer = FakeScorer(LOW_TOPIC_SIMILARITY_THRESHOLD)
        context = [{"block_id": "p1_b1", "type": "paragraph", "text": "x와 y가 정비례하는 관계"}]

        similarity, warnings = find_topic_mismatch_warning("정비례 관계를 나타낸 표이다.", context, scorer)

        self.assertEqual(similarity, LOW_TOPIC_SIMILARITY_THRESHOLD)
        self.assertEqual(warnings, [])

class TopicGroundingScorerTests(unittest.TestCase):
    def test_empty_caption_or_context_is_not_scored_without_loading_a_model(self):
        # Both early-outs happen before any model/torch is touched, so this
        # stays a fast, real (non-mocked) test of TopicGroundingScorer itself.
        scorer = TopicGroundingScorer()

        self.assertIsNone(scorer.score("", [{"block_id": "p1_b1", "text": "text"}]))
        self.assertIsNone(scorer.score("caption", []))
        self.assertIsNone(scorer.score("caption", [{"block_id": "p1_b1", "text": "   "}]))
        self.assertIsNone(scorer._model)


class RecordingScorer:
    """Fake TopicGroundingScorer that reports which text list it was scored
    against, so tests can tell caption_score and context_score apart."""

    def __init__(self, score_by_text):
        self.score_by_text = score_by_text
        self.calls = []

    def score(self, caption, context):
        self.calls.append((caption, context))
        texts = tuple(item.get("text") for item in context or [])
        return self.score_by_text.get(texts)


class ComputeGroundingScoresTests(unittest.TestCase):
    def test_uses_caption_and_context_separately(self):
        scorer = RecordingScorer({("반비례 관계 그래프",): 0.95, ("앞 문단", "뒤 문단"): 0.7})
        context = FigureContext(caption="반비례 관계 그래프", previous_paragraph="앞 문단", next_paragraph="뒤 문단")

        scores = compute_grounding_scores("설명", context, scorer)

        self.assertEqual(scores.caption_score, 0.95)
        self.assertEqual(scores.context_score, 0.7)
        self.assertEqual(scores.overall_score, round(0.6 * 0.95 + 0.4 * 0.7, 3))

    def test_overall_score_falls_back_to_whichever_side_is_available(self):
        scorer = RecordingScorer({("반비례 관계 그래프",): 0.8})
        context = FigureContext(caption="반비례 관계 그래프")

        scores = compute_grounding_scores("설명", context, scorer)

        self.assertEqual(scores.caption_score, 0.8)
        self.assertIsNone(scores.context_score)
        self.assertEqual(scores.overall_score, 0.8)

    def test_no_caption_or_paragraphs_yields_all_none(self):
        scorer = RecordingScorer({})
        context = FigureContext()

        scores = compute_grounding_scores("설명", context, scorer)

        self.assertIsNone(scores.caption_score)
        self.assertIsNone(scores.context_score)
        self.assertIsNone(scores.overall_score)
        self.assertEqual(scorer.calls, [])

    def test_to_dict_matches_the_three_named_scores(self):
        scores = compute_grounding_scores(
            "설명", FigureContext(caption="c"), RecordingScorer({("c",): 0.5})
        )

        self.assertEqual(scores.to_dict(), {"caption_score": 0.5, "context_score": None, "overall_score": 0.5})


if __name__ == "__main__":
    unittest.main()
