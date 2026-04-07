from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())

    if isinstance(value, Mapping):
        return {str(key): make_json_safe(val) for key, val in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())

    if hasattr(value, "__dict__"):
        data = {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
        if data:
            return make_json_safe(data)

    return str(value)

