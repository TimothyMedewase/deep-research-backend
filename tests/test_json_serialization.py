from __future__ import annotations

import json
import unittest

import httpx

from agent.integrations.webhook_memory import _maybe_json
from agent.json_utils import make_json_safe
from agent.research_tool_loop import _json_dumps
from main import _event_json


class FakeSparseVector:
    __slots__ = ("values",)

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __str__(self) -> str:
        return f"FakeSparseVector({self.values!r})"


class JsonSerializationTests(unittest.TestCase):
    def test_make_json_safe_handles_non_serializable_nested_object(self) -> None:
        payload = {
            "hits": [
                {
                    "metadata": {
                        "sparse_embedding": FakeSparseVector([0.1, 0.2]),
                    }
                }
            ]
        }

        safe = make_json_safe(payload)

        self.assertEqual(
            safe["hits"][0]["metadata"]["sparse_embedding"],
            "FakeSparseVector([0.1, 0.2])",
        )

    def test_event_json_handles_non_serializable_nested_object(self) -> None:
        body = json.loads(
            _event_json(
                {
                    "type": "memory_hit",
                    "data": {
                        "chunks": [
                            {
                                "metadata": {
                                    "sparse_embedding": FakeSparseVector([0.3]),
                                }
                            }
                        ]
                    },
                }
            )
        )

        self.assertEqual(
            body["data"]["chunks"][0]["metadata"]["sparse_embedding"],
            "FakeSparseVector([0.3])",
        )

    def test_tool_json_dumps_handles_non_serializable_nested_object(self) -> None:
        body = json.loads(
            _json_dumps(
                {
                    "hits": [
                        {
                            "metadata": {
                                "sparse_embedding": FakeSparseVector([0.4]),
                            }
                        }
                    ]
                }
            )
        )

        self.assertEqual(
            body["hits"][0]["metadata"]["sparse_embedding"],
            "FakeSparseVector([0.4])",
        )

    def test_maybe_json_returns_empty_dict_for_empty_success_body(self) -> None:
        response = httpx.Response(200, text="")

        self.assertEqual(_maybe_json(response), {})


if __name__ == "__main__":
    unittest.main()
