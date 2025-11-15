import json
from pathlib import Path

from brain_as_llm.encoder import TextEncoder, EncoderConfig
from brain_as_llm.llm_clients.dummy_client import DummyLLMClient
from brain_as_llm.state.canvas_store import CanvasStore


def test_text_encoder_canvas_structure(tmp_path: Path):
    response = json.dumps(
        {
            "summary": "Point A",
            "facts": ["Fact 1"],
            "entities": [{"name": "Alice", "type": "person", "description": "analyst"}],
            "quotes": ['"Sample quote"'],
            "notes_for_reasoner": "Consider budget implications.",
        }
    )
    encoder = TextEncoder(
        DummyLLMClient([response]),
        model_name="dummy-small",
        config=EncoderConfig(chunk_size=200, max_chunks=1),
    )
    result = encoder.encode(raw_text="Short text for testing", question="What happened?")

    assert "Point A" in result.canvas.summaries
    assert result.canvas.entities[0]["name"] == "Alice"
    assert result.canvas.facts == ["Fact 1"]
    assert not result.from_cache


def test_canvas_store_persistence(tmp_path: Path):
    store = CanvasStore(tmp_path)
    response = json.dumps(
        {
            "summary": "Persistent summary",
            "facts": [],
            "entities": [],
            "quotes": [],
            "notes_for_reasoner": "",
        }
    )
    encoder = TextEncoder(
        DummyLLMClient([response]),
        model_name="dummy-small",
        config=EncoderConfig(chunk_size=200, max_chunks=1),
        canvas_store=store,
    )
    doc_id = "doc123"
    encoder.encode(raw_text="Persist me", question="Q?", doc_id=doc_id)

    cached = store.load_canvas(doc_id)
    assert cached is not None
    assert "Persist me"[:7] in (cached.raw_chunks[0])
