import unittest

from agent.thread_store import ThreadStore
from models.schemas import ResearchConstraints


class ThreadStoreTests(unittest.TestCase):
    def test_get_or_create_and_append(self) -> None:
        store = ThreadStore()
        thread = store.get_or_create("s1", ResearchConstraints())
        self.assertEqual(thread.session_id, "s1")
        store.append_turn(
            thread,
            user_query="What is X?",
            assistant_answer="X is Y.",
        )
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.last_answer, "X is Y.")
        store.update_from_tracker(
            thread,
            token_count=100,
            cost_usd=0.01,
            context_chunks=["chunk"],
        )
        again = store.get("s1")
        assert again is not None
        self.assertEqual(again.token_count, 100)
        self.assertEqual(again.context_chunks, ["chunk"])
        self.assertTrue(store.delete("s1"))
        self.assertIsNone(store.get("s1"))


if __name__ == "__main__":
    unittest.main()
