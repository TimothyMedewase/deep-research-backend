import unittest

from models.schemas import ResearchConstraints


class ResearchConstraintsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ResearchConstraints()
        self.assertEqual(cfg.research_model, "gpt-4o")
        self.assertEqual(cfg.compression_model, "gpt-4o-mini")
        self.assertGreaterEqual(cfg.max_tokens, 2000)

    def test_clamps_and_allowlist(self) -> None:
        cfg = ResearchConstraints(
            max_tokens=999999,
            max_cost_usd=50,
            max_sub_questions=100,
            max_tool_rounds=100,
            memory_top_k=100,
            search_max_results=0,
            compression_threshold=50000,
            research_model="claude-3.5",
            compression_model="nope",
        )
        self.assertEqual(cfg.max_tokens, 20000)
        self.assertEqual(cfg.max_cost_usd, 1.0)
        self.assertEqual(cfg.max_sub_questions, 8)
        self.assertEqual(cfg.max_tool_rounds, 12)
        self.assertEqual(cfg.memory_top_k, 10)
        self.assertEqual(cfg.search_max_results, 1)
        self.assertLess(cfg.compression_threshold, cfg.max_tokens)
        self.assertEqual(cfg.research_model, "gpt-4o")
        self.assertEqual(cfg.compression_model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
