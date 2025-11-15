"""Persistence helpers for multi-resolution canvases."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..encoder.text_encoder import Canvas


class CanvasStore:
    """Very small JSON file based canvas store."""

    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, doc_id: str) -> Path:
        safe_id = doc_id.replace("/", "_")
        return self._root / f"{safe_id}.json"

    def load_canvas(self, doc_id: str) -> Optional[Canvas]:
        path = self._path_for(doc_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Canvas.from_dict(data)

    def save_canvas(self, doc_id: str, canvas: Canvas) -> None:
        path = self._path_for(doc_id)
        payload = canvas.to_dict()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["CanvasStore"]
