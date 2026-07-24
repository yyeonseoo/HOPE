import unittest

from src.analysis.figure.summary import derive_summary


class DeriveSummaryTests(unittest.TestCase):
    def test_none_or_empty_description_yields_none(self):
        self.assertIsNone(derive_summary(None))
        self.assertIsNone(derive_summary("   "))

    def test_short_first_sentence_is_returned_as_is(self):
        self.assertEqual(derive_summary("토끼와 거북이의 거리 변화 그래프."), "토끼와 거북이의 거리 변화 그래프")

    def test_only_the_first_sentence_is_used(self):
        text = "짧은 문장. 두 번째 문장은 무시된다."
        self.assertEqual(derive_summary(text), "짧은 문장")

    def test_long_first_sentence_is_truncated_at_a_word_boundary(self):
        text = "이 그래프는 토끼와 거북이의 경주에서 시간에 따른 이동 거리를 자세히 나타낸다."
        summary = derive_summary(text, max_chars=30)

        self.assertLessEqual(len(summary), 31)  # 30 chars + ellipsis
        self.assertTrue(summary.endswith("…"))
        self.assertNotIn("  ", summary)

    def test_summary_never_exceeds_max_chars_plus_ellipsis(self):
        text = "가" * 100 + "."
        summary = derive_summary(text, max_chars=30)

        self.assertEqual(len(summary), 31)
        self.assertTrue(summary.endswith("…"))


if __name__ == "__main__":
    unittest.main()
