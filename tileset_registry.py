"""
tileset_registry.py
-------------------
Persistent tileset registry — survives server restarts.

Storage  : tileset_registry.json (same folder as this file)
GID space: firstgid is assigned once and never changes for a given name.

Reserved GID ranges (must not be touched):
  201–335  AutoLight  (ground)   tilecount=135
  336–349  large-rock (wall)     tilecount=14
  350      gap / safety buffer
  351+     dynamic (bridge, custom, ...)

Supported tileset types:
  "bridge"   – can have arbitrary tilesize / rows / cols
  "unit"     – custom unit sprite sheet
  "terrain"  – custom ground tileset
  "items"    – custom items tileset

Two registration paths:
  1. RPC / runtime  → register_tileset(...)
  2. Bulk JSON file → import_from_file(path)  or  import_from_dict(data)
"""

import json
import threading
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGISTRY_FILE = Path(__file__).parent / "tileset_registry.json"

# First GID that is safe for dynamic assignment
_DYNAMIC_START = 351

# GID ranges that must never be assigned to dynamic tilesets
_RESERVED_RANGES = [
    (201, 335),   # AutoLight
    (336, 349),   # large-rock
]

VALID_TYPES = {"bridge", "unit", "terrain", "items"}

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    """Read registry from disk. Returns {} if file missing or corrupt."""
    if not _REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(registry: dict):
    """Write registry atomically (write-then-rename)."""
    tmp = _REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_REGISTRY_FILE)


def _is_reserved(firstgid: int, tilecount: int) -> bool:
    end = firstgid + tilecount - 1
    for lo, hi in _RESERVED_RANGES:
        if firstgid <= hi and end >= lo:
            return True
    return False


def _next_firstgid(registry: dict, tilecount: int) -> int:
    """Find the lowest safe firstgid that fits `tilecount` tiles."""
    # Collect all occupied [start, end] ranges
    occupied = list(_RESERVED_RANGES)
    for entry in registry.values():
        occupied.append((entry["firstgid"], entry["firstgid"] + entry["tilecount"] - 1))
    occupied.sort()

    candidate = _DYNAMIC_START
    for lo, hi in occupied:
        if candidate + tilecount - 1 < lo:
            break           # fits in the gap before this range
        if candidate <= hi:
            candidate = hi + 2   # skip past range + 1 safety gap
    return candidate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_tileset(
    name: str,
    tileset_type: str,
    png_b64: str,
    tilecount: int,
    columns: int,
    tilewidth: int = 20,
    tileheight: int = 20,
    tiles: Optional[dict] = None,
    bridge_variant: str = "12",
    bridge_custom_dir: str = "",
    bridge_layout: list = None,
    layout_rows: int = None,
    layout_cols: int = None,
    bridge_simple: dict = None,
) -> dict:
    """Register a tileset. Returns the registry entry (with firstgid).

    - New name   → assign firstgid, persist.
    - Existing   → update PNG (and optionally tiles/dimensions), keep firstgid.

    `tiles` is an optional key→local_id mapping (used by bridge tilesets).
    """
    if tileset_type not in VALID_TYPES:
        raise ValueError(f"Invalid tileset_type '{tileset_type}'. Must be one of {VALID_TYPES}")
    if tilecount < 1:
        raise ValueError("tilecount must be >= 1")
    if columns < 1:
        raise ValueError("columns must be >= 1")

    with _lock:
        registry = _load()

        if name in registry:
            # Update mutable fields, freeze firstgid
            entry = registry[name]
            entry["png"]        = png_b64
            entry["type"]       = tileset_type
            entry["tilecount"]  = tilecount
            entry["columns"]    = columns
            entry["tilewidth"]  = tilewidth
            entry["tileheight"] = tileheight
            if tiles is not None:
                entry["tiles"] = tiles
            if tileset_type == "bridge":
                entry["bridge_variant"]    = bridge_variant
                entry["bridge_custom_dir"] = bridge_custom_dir
                if bridge_layout is not None:
                    entry["bridge_layout"] = bridge_layout
                if layout_rows is not None:
                    entry["layout_rows"] = layout_rows
                if layout_cols is not None:
                    entry["layout_cols"] = layout_cols
                if bridge_simple is not None:
                    entry["bridge_simple"] = bridge_simple
        else:
            fg = _next_firstgid(registry, tilecount)
            if _is_reserved(fg, tilecount):
                raise RuntimeError(f"Computed firstgid {fg} overlaps reserved range")
            entry = {
                "name":       name,
                "type":       tileset_type,
                "firstgid":   fg,
                "tilecount":  tilecount,
                "columns":    columns,
                "tilewidth":  tilewidth,
                "tileheight": tileheight,
                "png":        png_b64,
            }
            if tiles is not None:
                entry["tiles"] = tiles
            if tileset_type == "bridge":
                entry["bridge_variant"]    = bridge_variant
                entry["bridge_custom_dir"] = bridge_custom_dir
                if bridge_layout is not None:
                    entry["bridge_layout"] = bridge_layout
                if layout_rows is not None:
                    entry["layout_rows"] = layout_rows
                if layout_cols is not None:
                    entry["layout_cols"] = layout_cols
                if bridge_simple is not None:
                    entry["bridge_simple"] = bridge_simple
            registry[name] = entry

        _save(registry)
        return dict(entry)


def get_tileset(name: str) -> Optional[dict]:
    """Return a single entry by name, or None if not found."""
    return _load().get(name)


def get_all(tileset_type: Optional[str] = None) -> dict:
    """Return all entries, optionally filtered by type."""
    registry = _load()
    if tileset_type is None:
        return dict(registry)
    return {k: v for k, v in registry.items() if v.get("type") == tileset_type}


def remove_tileset(name: str) -> bool:
    """Remove an entry. Returns True if it existed."""
    with _lock:
        registry = _load()
        if name not in registry:
            return False
        del registry[name]
        _save(registry)
        return True


def import_from_dict(data: dict) -> dict:
    """Bulk-import from a dict (e.g. parsed from a .json file).

    Each key is a tileset name; each value must contain at minimum:
      type, png, tilecount, columns
    Optional: tilewidth, tileheight, tiles

    Existing entries keep their firstgid; new entries are assigned one.
    Returns { name: entry } for every processed tileset.
    """
    results = {}
    for name, ts in data.items():
        try:
            entry = register_tileset(
                name         = name,
                tileset_type = ts["type"],
                png_b64      = ts.get("png", ""),
                tilecount    = int(ts["tilecount"]),
                columns      = int(ts["columns"]),
                tilewidth    = int(ts.get("tilewidth", 20)),
                tileheight   = int(ts.get("tileheight", 20)),
                tiles        = ts.get("tiles"),
            )
            results[name] = entry
        except (KeyError, ValueError) as exc:
            results[name] = {"error": str(exc)}
    return results


def import_from_file(path: str) -> dict:
    """Load a JSON file and call import_from_dict. Returns processed entries."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    return import_from_dict(data)


def export_to_file(path: str):
    """Dump the current registry to a JSON file (for backup / transfer)."""
    registry = _load()
    Path(path).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
