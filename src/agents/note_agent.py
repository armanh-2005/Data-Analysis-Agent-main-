# src/agents/note_agent.py
from __future__ import annotations

from typing import Any, Dict, Optional

# --- FIX: Change 'from db.repository' to 'from src.db.repository' ---
from src.db.repository import SQLiteRepository  


class NoteAgentError(Exception):
    pass


def _is_empty_value(v: Any) -> bool:
    # Treat these as "empty" when deciding whether to hydrate from saved state.
    if v is None:
        return True
    if v == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dicts:
      - dict values merged recursively
      - other values overridden by incoming
    """
    out = dict(base or {})
    for k, v in (incoming or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class NoteAgent:
    """
    NoteAgent responsibilities:
      - Persist full workflow state to SQLite (notes_state.state_json)
      - Hydrate current state from saved state (resumability)
      - Keep notes merged (does not overwrite newer notes with older ones)

    Important:
      - Returns PATCHES only (dict) so orchestrator can apply them to WorkflowState.
      - Does not execute business logic; it's strictly state persistence/hydration.
    """

    name = "note_agent"

    def __init__(self, db_path: str):
        self.repo = SQLiteRepository(db_path)

    # -------------------------
    # Hydration (DB -> patch)
    # -------------------------

    def load_patch(self, run_id: str) -> Dict[str, Any]:
        """
        Load full saved state dict from DB.
        Returns {} if not found.
        """
        saved = self.repo.load_notes_state(run_id)
        if not saved or not isinstance(saved, dict):
            return {}
        return saved

    def hydrate_state(self, state: Any, hydrate_missing_only: bool = True) -> Dict[str, Any]:
        """
        Build a patch that merges saved state into the current state.

        Strategy:
          - Always deep-merge notes: saved_notes -> current_notes (current wins)
          - If hydrate_missing_only=True:
              only fills fields in current state that are "empty"
            else:
              overwrites everything except run_id/user_question (dangerous; off by default)
        """
        run_id = getattr(state, "run_id", None)
        if not run_id:
            return {}

        saved = self.load_patch(run_id)
        if not saved:
            return {}

        patch: Dict[str, Any] = {}

        # 1) Notes: merge so current wins.
        current_notes = getattr(state, "notes", {}) or {}
        saved_notes = saved.get("notes", {}) if isinstance(saved.get("notes", {}), dict) else {}
        merged_notes = _deep_merge(saved_notes, current_notes)
        patch["notes"] = merged_notes

        # 2) Other fields: hydrate missing only (default).
        # Never override identity fields.
        protected = {"run_id", "user_question"}

        if hydrate_missing_only:
            for k, v_saved in saved.items():
                if k in protected or k == "notes":
                    continue
                if not hasattr(state, k):
                    continue
                v_current = getattr(state, k)
                if _is_empty_value(v_current) and not _is_empty_value(v_saved):
                    patch[k] = v_saved
        else:
            for k, v_saved in saved.items():
                if k in protected or k == "notes":
                    continue
                if hasattr(state, k):
                    patch[k] = v_saved

        return patch

    # -------------------------
    # Persistence (state -> DB)
    # -------------------------

    def save(self, state: Any) -> Dict[str, Any]:
        """
        Persist full state into notes_state table.
        Returns {} (side-effect only).
        """
        run_id = getattr(state, "run_id", None)
        if not run_id:
            return {}

        payload = self._state_to_dict(state)
        self.repo.save_notes_state(run_id, payload)
        return {}

    def save_patch(self, run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply patch on top of saved state and persist.
        Useful for small updates without having the whole WorkflowState object.
        """
        saved = self.load_patch(run_id)
        merged = _deep_merge(saved, patch)
        self.repo.save_notes_state(run_id, merged)
        return {}

    # -------------------------
    # Utilities
    # -------------------------

    def _state_to_dict(self, state: Any) -> Dict[str, Any]:
        """
        Convert WorkflowState (or compatible object) into a JSON-serializable dict.
        Prefer state.to_dict() from workflows/state.py.
        """
        if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
            d = state.to_dict()  # type: ignore[attr-defined]
            if not isinstance(d, dict):
                raise NoteAgentError("state.to_dict() must return a dict.")
            return d

        # Fallback: best-effort __dict__ copy (should already be JSON-friendly in your design).
        if hasattr(state, "__dict__"):
            d = dict(state.__dict__)
            if not isinstance(d, dict):
                raise NoteAgentError("Failed to serialize state via __dict__.")
            return d

        raise NoteAgentError("State object is not serializable (missing to_dict and __dict__).")