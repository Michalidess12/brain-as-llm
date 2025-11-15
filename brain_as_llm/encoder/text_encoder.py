"\"\"\"Text encoder that maintains a persistent, multi-resolution canvas.\"\"\""
from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, TYPE_CHECKING

from ..llm_clients.base import LLMClient

if TYPE_CHECKING:  # pragma: no cover
    from ..state.canvas_store import CanvasStore

logger = logging.getLogger(__name__)


@dataclass
class Canvas:
    """Multi-resolution representation shared across the brain pipeline."""

    raw_chunks: List[str] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    notes_for_reasoner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Canvas:
        return cls(
            raw_chunks=list(data.get("raw_chunks", [])),
            summaries=list(data.get("summaries", [])),
            facts=list(data.get("facts", [])),
            entities=list(data.get("entities", [])),
            quotes=list(data.get("quotes", [])),
            notes_for_reasoner=data.get("notes_for_reasoner"),
        )


class ChunkSummary(TypedDict):
    chunk_index: int
    usage: Dict[str, Any]
    summary: Dict[str, Any]


@dataclass
class EncoderConfig:
    """Runtime configuration for the text encoder."""

    chunk_size: int = 1800
    chunk_overlap: int = 300
    max_chunks: int = 8
    system_prompt: str = (
        "You are a meticulous research assistant. Given a chunk of a document and a question, "
        "produce a compact JSON summary with fields summary (string), facts (list of strings), "
        "entities (list of {name, type, description}), quotes (list of key quotes), and "
        "notes_for_reasoner (string). Keep entries short and relevant."
    )
    temperature: float = 0.0
    max_tokens: Optional[int] = 600


@dataclass
class TextEncoderResult:
    """Output returned by :class:`TextEncoder`."""

    canvas: Canvas
    chunk_summaries: List[ChunkSummary] = field(default_factory=list)
    from_cache: bool = False


class TextEncoder:
    """Encodes raw text into a structured canvas via an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        config: Optional[EncoderConfig] = None,
        *,
        canvas_store: Optional["CanvasStore"] = None,
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._config = config or EncoderConfig()
        self._canvas_store = canvas_store

    def encode(self, raw_text: str, question: str, *, doc_id: Optional[str] = None) -> TextEncoderResult:
        """Produce or load a persistent canvas describing the text w.r.t. the question."""

        if doc_id and self._canvas_store:
            cached = self._canvas_store.load_canvas(doc_id)
            if cached:
                logger.debug("Loaded cached canvas for doc_id=%s", doc_id)
                return TextEncoderResult(canvas=cached, chunk_summaries=[], from_cache=True)

        chunks = self._chunk_text(raw_text)
        canvas = Canvas(raw_chunks=list(chunks))
        chunk_summaries: List[ChunkSummary] = []

        for idx, chunk in enumerate(chunks, start=1):
            prompt = self._build_prompt(chunk=chunk, question=question, chunk_index=idx, total_chunks=len(chunks))
            logger.debug("Encoding chunk %s/%s (chars=%s)", idx, len(chunks), len(chunk))
            response = self._llm.chat(
                messages=[
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self._model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            summary_dict = self._parse_summary_response(response["text"])
            chunk_summaries.append(
                {"chunk_index": idx, "usage": response.get("usage", {}), "summary": summary_dict}
            )
            self._merge_summary(canvas, summary_dict)

        if doc_id and self._canvas_store:
            self._canvas_store.save_canvas(doc_id, canvas)

        return TextEncoderResult(canvas=canvas, chunk_summaries=chunk_summaries, from_cache=False)

    def _chunk_text(self, raw_text: str) -> List[str]:
        if not raw_text:
            return [""]
        max_chars = max(self._config.chunk_size, 1)
        if len(raw_text) <= max_chars:
            return [raw_text.strip()]

        chunks: List[str] = []
        stride = max_chars - self._config.chunk_overlap
        start = 0
        while start < len(raw_text) and len(chunks) < self._config.max_chunks:
            chunk = raw_text[start : start + max_chars]
            chunks.append(textwrap.dedent(chunk).strip())
            start += stride
        logger.debug("Chunked text into %s pieces (max=%s)", len(chunks), self._config.max_chunks)
        return chunks or [raw_text.strip()]

    def _build_prompt(self, *, chunk: str, question: str, chunk_index: int, total_chunks: int) -> str:
        return textwrap.dedent(
            f"""
            Chunk {chunk_index}/{total_chunks}
            QUESTION: {question.strip()}
            CHUNK:
            {chunk.strip()}

            Respond with valid JSON only.
            """
        ).strip()

    def _parse_summary_response(self, raw_text: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse encoder response as JSON; falling back to heuristic parsing")
            data = {
                "summary": raw_text.strip(),
                "facts": [line.strip("- ") for line in raw_text.splitlines() if line.strip()],
                "entities": [],
                "quotes": [],
                "notes_for_reasoner": raw_text.strip(),
            }
        data.setdefault("summary", "")
        data.setdefault("facts", [])
        data.setdefault("entities", [])
        data.setdefault("quotes", [])
        data.setdefault("notes_for_reasoner", "")
        return data

    def _merge_summary(self, canvas: Canvas, summary: Dict[str, Any]) -> None:
        if summary.get("summary"):
            canvas.summaries.append(summary["summary"])
        canvas.facts.extend(summary.get("facts", []))
        canvas.entities.extend(summary.get("entities", []))
        canvas.quotes.extend(summary.get("quotes", []))
        note = summary.get("notes_for_reasoner")
        combined = f"{canvas.notes_for_reasoner or ''}\n{note}".strip() if note else canvas.notes_for_reasoner
        canvas.notes_for_reasoner = combined or canvas.notes_for_reasoner


__all__ = ["TextEncoder", "EncoderConfig", "TextEncoderResult", "Canvas"]
