import unittest
from unittest.mock import patch

from src.page_confidence import build_page_confidence


class PageConfidenceTests(unittest.TestCase):
    @patch("src.page_confidence._get_embedding_model")
    def test_high_confidence_when_description_matches_context(self, mock_model):
        mock_model.return_value = None

        page_result = {
            "page_id": 1,
            "blocks": [
                {
                    "block_id": "p1_b1",
                    "type": "paragraph",
                    "text": "오른쪽 그래프는 시간에 따른 높이의 변화를 나타낸 것이다.",
                    "reading_order": 1,
                },
                {
                    "block_id": "p1_b2",
                    "type": "figure",
                    "text": None,
                    "score": 0.9,
                    "reading_order": 2,
                },
            ],
        }

        semantic_analyses = [
            {
                "block_id": "p1_b2",
                "type": "figure",
                "detection": {"confidence": 0.9},
                "description": {
                    "status": "generated",
                    "long_text": "이 그래프는 시간에 따른 높이 변화를 나타냅니다.",
                },
                "context": {
                    "previous_block_id": "p1_b1",
                },
                "warnings": [],
            }
        ]

        page_description = {
            "text": "[paragraph] 오른쪽 그래프는 시간에 따른 높이의 변화를 나타낸 것이다.\n[figure] 이 그래프는 시간에 따른 높이 변화를 나타냅니다.",
            "block_ids": ["p1_b1", "p1_b2"],
            "warnings": [],
        }

        result = build_page_confidence(page_result, semantic_analyses, page_description)

        self.assertGreaterEqual(result["components"]["semantic_context_similarity"], 0.6)
        self.assertGreaterEqual(result["score"], 70)

    @patch("src.page_confidence._get_embedding_model")
    def test_claim_mismatch_penalty_lowers_similarity(self, mock_model):
        mock_model.return_value = None

        page_result = {
            "page_id": 2,
            "blocks": [
                {
                    "block_id": "p2_b1",
                    "type": "paragraph",
                    "text": "가장 높은 곳은 190 m이다.",
                    "reading_order": 1,
                },
                {
                    "block_id": "p2_b2",
                    "type": "figure",
                    "text": None,
                    "score": 0.9,
                    "reading_order": 2,
                },
            ],
        }

        semantic_analyses = [
            {
                "block_id": "p2_b2",
                "type": "figure",
                "detection": {"confidence": 0.9},
                "description": {
                    "status": "generated",
                    "long_text": "그래프에서 가장 높은 높이는 199 m입니다.",
                },
                "context": {
                    "previous_block_id": "p2_b1",
                },
                "warnings": [],
            }
        ]

        page_description = {
            "text": "[paragraph] 가장 높은 곳은 190 m이다.\n[figure] 그래프에서 가장 높은 높이는 199 m입니다.",
            "block_ids": ["p2_b1", "p2_b2"],
            "warnings": [],
        }

        result = build_page_confidence(page_result, semantic_analyses, page_description)

        block_score = result["block_scores"][0]
        self.assertGreater(block_score["claim_penalty"], 0)
        self.assertLess(block_score["semantic_context_similarity"], block_score["embedding_similarity"])

    @patch("src.page_confidence._get_embedding_model")
    def test_warnings_lower_confidence(self, mock_model):
        mock_model.return_value = None

        page_result = {
            "page_id": 3,
            "blocks": [
                {
                    "block_id": "p3_b1",
                    "type": "paragraph",
                    "text": "x와 y는 정비례 관계이다.",
                    "reading_order": 1,
                },
                {
                    "block_id": "p3_b2",
                    "type": "formula",
                    "text": "y=ax",
                    "score": 0.8,
                    "reading_order": 2,
                },
            ],
        }

        semantic_analyses = [
            {
                "block_id": "p3_b2",
                "type": "formula",
                "detection": {"confidence": 0.8},
                "description": {
                    "status": "generated",
                    "long_text": "수식 y=ax는 정비례 관계를 나타냅니다.",
                },
                "context": {
                    "previous_block_id": "p3_b1",
                },
                "warnings": [
                    "Pix2tex output was rejected as unreliable; fallback recognizer was used."
                ],
            }
        ]

        page_description = {
            "text": "[paragraph] x와 y는 정비례 관계이다.\n[formula] 수식 y=ax는 정비례 관계를 나타냅니다.",
            "block_ids": ["p3_b1", "p3_b2"],
            "warnings": [],
        }

        result = build_page_confidence(page_result, semantic_analyses, page_description)

        self.assertLess(result["components"]["warning"], 1.0)


if __name__ == "__main__":
    unittest.main()