from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def state_get(state: Any, key: str, default: Any = None) -> Any:
    # Support dataclass-like or dict-like states.
    if hasattr(state, key):
        return getattr(state, key)
    if isinstance(state, dict):
        return state.get(key, default)
    return default


def deep_merge_dict(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge delta into base (non-destructive). Nested dicts are merged recursively.
    Other types are overwritten.
    """
    out = dict(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def confidence_label(conf: Optional[float]) -> str:
    if conf is None:
        return "low"
    if conf >= 0.8:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


def schema_index_by_column(schema: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for item in schema or []:
        col = item.get("column_name")
        if isinstance(col, str) and col.strip():
            idx[col] = item
    return idx
