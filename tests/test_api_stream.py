import json
from unittest.mock import MagicMock, patch

import pytest

from src.api import stream_fact_check


class FakeGraph:
    async def astream(self, _payload):
        yield {"normalize": {"claim": "c"}}
        yield {"search": {"documents": []}}
        yield {"verdict": {"verdict": "unverified", "confidence": 0.0}}


@pytest.mark.asyncio
async def test_stream_emits_one_sse_event_per_node():
    with patch("src.api.build_graph", return_value=FakeGraph()):
        chunks = [chunk async for chunk in stream_fact_check("c")]

    assert all(chunk.startswith("data: ") for chunk in chunks)
    node_names = [json.loads(c[6:])["node"] for c in chunks if "node" in c]
    assert node_names[:3] == ["normalize", "search", "verdict"]


@pytest.mark.asyncio
async def test_stream_ends_with_a_done_event():
    with patch("src.api.build_graph", return_value=FakeGraph()):
        chunks = [chunk async for chunk in stream_fact_check("c")]

    assert json.loads(chunks[-1][6:]) == {"event": "done"}


class FailingGraph:
    """A node succeeds, then the next one raises mid-stream."""

    async def astream(self, _payload):
        yield {"normalize": {"claim": "c"}}
        raise ValueError("boom")


@pytest.mark.asyncio
async def test_stream_emits_error_event_and_terminates_on_node_failure():
    with patch("src.api.build_graph", return_value=FailingGraph()):
        chunks = [chunk async for chunk in stream_fact_check("c")]

    # The successful node before the failure still streamed.
    assert json.loads(chunks[0][6:])["node"] == "normalize"

    # The stream ends with exactly one error event, not a 'done' event,
    # and does not propagate the exception to the caller.
    assert len(chunks) == 2
    assert json.loads(chunks[-1][6:]) == {"event": "error", "detail": "boom"}
