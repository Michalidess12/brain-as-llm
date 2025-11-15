"""Minimal persistence helpers for controller state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class BaseStateStore(Protocol):
    """Simple protocol for saving and loading controller state."""

    def load_state(self) -> Optional[Dict[str, Any]]:
        """Return the current state dictionary if available."""

    def save_state(self, state: Dict[str, Any]) -> None:
        """Persist the provided state dictionary."""


class InMemoryStateStore:
    """Volatile store, primarily for tests."""

    def __init__(self) -> None:
        self._state: Optional[Dict[str, Any]] = None

    def load_state(self) -> Optional[Dict[str, Any]]:
        return self._state

    def save_state(self, state: Dict[str, Any]) -> None:
        self._state = state.copy()


class JSONStateStore:
    """Very small JSON backed state store."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Optional[Dict[str, Any]]:
        if not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, state: Dict[str, Any]) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)


__all__ = ["BaseStateStore", "InMemoryStateStore", "JSONStateStore"]
