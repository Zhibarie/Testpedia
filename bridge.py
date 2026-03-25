import json
from typing import Dict, Any
import map_pipeline
import tileset_registry as _reg
from wizard_state import WizardState, WizardStep
from procedural_map_generator_functions import _get_mirrored_positions

state = WizardState()


def _registry_entry_to_state_tileset(entry: dict) -> dict:
    """Normalize registry payload into the runtime tileset shape."""
    return {
        "name": entry["name"],
        "firstgid": entry["firstgid"],
        "tilecount": entry["tilecount"],
        "columns": entry["columns"],
        "tilewidth": entry.get("tilewidth", 20),
        "tileheight": entry.get("tileheight", 20),
        "png": entry.get("png", ""),
        "tiles": entry.get("tiles", {}),
        "bridge_variant": entry.get("bridge_variant", "12"),
        "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
        "bridge_simple": entry.get("bridge_simple"),
    }


def _restore_state_from_registry():
    """Reload all active tilesets from registry into live state on startup."""
    all_ts = _reg.get_all()
    for name, entry in all_ts.items():
        ts_type = entry.get("type", "")
        ts_info = _registry_entry_to_state_tileset(entry)
        if ts_type == "bridge":
            state.bridge_tilesets[name] = ts_info
            if not state.active_bridge_name:
                state.active_bridge_name = name
        elif ts_type in ("terrain", "unit", "items"):
            state.active_tilesets[ts_type] = ts_info

_restore_state_from_registry()


def _matrix_payload(matrix):
    if matrix is None: return None
    import numpy as _np
    arr = _np.asarray(matrix)
    return {"shape": list(arr.shape), "data": arr.astype(int).ravel().tolist()}


def _preview_frame(label: str, matrix) -> dict:
    """Serialize preview matrix payload with strict 2D shape validation."""
    import numpy as _np
    arr = _np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Preview frame '{label}' must be 2D, got shape {list(arr.shape)}")
    # Some clients may submit incomplete seed cells (None). Normalize here so
    # preview serialization never crashes the entire RPC request.
    if arr.dtype == object:
        arr = _np.where(arr == None, 0, arr)  # noqa: E711 - intentional None comparison for ndarray
    arr = _np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    return {
        "label": label,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "data": arr.astype(int).ravel().tolist(),
    }


def _sanitize_binary_grid(grid):
    """Normalize user-provided seed grid into a strict 0/1 integer matrix."""
    import numpy as _np
    arr = _np.asarray(grid, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"Seed grid must be 2D, got shape {list(arr.shape)}")
    arr = _np.where(arr == None, 0, arr)  # noqa: E711 - intentional None comparison for ndarray
    def _to01(v):
        try:
            if v is None:
                return 0
            if isinstance(v, bool):
                return 1 if v else 0
            return 1 if float(v) > 0 else 0
        except Exception:
            return 0
    return _np.vectorize(_to01, otypes=[int])(arr)


def _snapshot() -> Dict[str, Any]:
    return {
        "meta": {
            "height": int(state.height),
            "width":  int(state.width),
            "mirroring": state.mirroring,
            "pattern": int(state.pattern),
            "num_height_levels": int(state.num_height_levels),
            "num_ocean_levels":  int(state.num_ocean_levels),
            "height_region_scale": float(getattr(state, "height_region_scale", 1.0)),
            "region_metrics": getattr(state, "region_metrics", {}) or {},
            "num_command_centers": int(state.num_command_centers),
            "num_resource_pulls":  int(state.num_resource_pulls),
            "completed_step": int(state.completed_step),
            "current_step":   int(state.current_step),
            "hill_drawing_mode": state.hill_drawing_mode,
            "coast_smooth_passes": int(getattr(state, "coast_smooth_passes", 1)),
            "custom_terrain_mapping_keys": list((getattr(state, "custom_terrain_mapping", None) or {}).keys()),
            "bridge_tilesets": list(getattr(state, "bridge_tilesets", {}).keys()),
            "active_bridge": getattr(state, "active_bridge_name", ""),
        },
        "cc_positions":       [[int(r),int(c)] for r,c in state.cc_positions],
        "resource_positions": [[int(r),int(c)] for r,c in state.resource_positions],
        "matrices": {
            "coastline_height_map": _matrix_payload(state.coastline_height_map),
            "wall_matrix":          _matrix_payload(state.wall_matrix),
            "height_map":           _matrix_payload(state.height_map),
            "id_matrix":            _matrix_payload(state.id_matrix),
            "items_matrix":         _matrix_payload(state.items_matrix),
            "units_matrix":         _matrix_payload(state.units_matrix),
            "bridge_matrix":        _matrix_payload(getattr(state, "bridge_matrix", None)),
        },
    }


def _params(p):
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    try:
        return json.loads(p or "{}")
    except Exception as exc:
        raise ValueError(f"Invalid JSON params: {exc}") from exc


def _apply_state_params(state, params):
    """Apply the 8 common wizard params from a request dict onto state."""
    state.height    = max(20, min(640, int(params.get("height",    state.height))))
    state.width     = max(20, min(640, int(params.get("width",     state.width))))
    state.mirroring = str(params.get("mirroring", state.mirroring))
    state.pattern   = int(params.get("tileset",   params.get("pattern",              state.pattern)))
    state.num_height_levels   = int(params.get("heightLevels",  params.get("num_height_levels",   state.num_height_levels)))
    state.num_ocean_levels    = int(params.get("oceanLevels",   params.get("num_ocean_levels",    state.num_ocean_levels)))
    hrs = params.get("heightRegionScale", params.get("height_region_scale", getattr(state, "height_region_scale", 1.0)))
    try:
        state.height_region_scale = max(0.1, min(3.0, float(hrs)))
    except Exception:
        state.height_region_scale = 1.0
    state.num_command_centers = int(params.get("numPlayers",    params.get("num_command_centers", state.num_command_centers)))
    state.num_resource_pulls  = int(params.get("numResources",  params.get("num_resource_pulls",  state.num_resource_pulls)))


def _clear_custom_units_from_matrix(state):
    """Zero out all custom unit cells in units_matrix."""
    _CustomLayer.clear_matrix(state.units_matrix, state._custom_units)


# ---------------------------------------------------------------------------
# _CustomLayer — shared vectorized logic for unit + item custom placements
# ---------------------------------------------------------------------------

class _CustomLayer:
    """Vectorized place / remove / clear / undo for a custom placement layer.

    All matrix writes use NumPy advanced indexing — no per-cell Python loops.

    Attributes on state:
        matrix_attr   – e.g. "units_matrix" or "items_matrix"
        store_attr    – e.g. "_custom_units" or "_custom_items"
        history_attr  – e.g. "_custom_unit_history" or "_custom_item_history"
    """

    @staticmethod
    def _ensure_matrix(state, attr: str) -> "np.ndarray":
        import numpy as _np
        m = getattr(state, attr)
        if m is None:
            h, w = int(state.height), int(state.width)
            m = _np.zeros((h, w), dtype=int)
            setattr(state, attr, m)
        return m

    @staticmethod
    def _store(state, attr: str) -> list:
        return getattr(state, attr)

    @staticmethod
    def clear_matrix(matrix, store: list):
        """Zero out matrix cells listed in store — vectorized."""
        if matrix is None or not store:
            return
        import numpy as _np
        arr = _np.array(store, dtype=object)   # (N, 3)
        rs  = arr[:, 0].astype(int)
        cs  = arr[:, 1].astype(int)
        gs  = arr[:, 2].astype(int)
        h, w = matrix.shape
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        # Only clear cells whose GID still matches (don't clobber other layers)
        match = valid & (matrix[rs.clip(0, h-1), cs.clip(0, w-1)] == gs)
        matrix[rs[match], cs[match]] = 0

    @staticmethod
    def write_matrix(matrix, store: list):
        """Write all (r,c,gid) entries to matrix — vectorized."""
        if not store:
            return
        import numpy as _np
        arr   = _np.array(store, dtype=object)
        rs    = arr[:, 0].astype(int)
        cs    = arr[:, 1].astype(int)
        gs    = arr[:, 2].astype(int)
        h, w  = matrix.shape
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        matrix[rs[valid], cs[valid]] = gs[valid]

    @staticmethod
    def find_closest(store: list, row: int, col: int, threshold: int = 3) -> int:
        """Return index of closest entry within Manhattan threshold, or -1.

        Vectorized distance computation — no Python loop.
        """
        if not store:
            return -1
        import numpy as _np
        arr  = _np.array(store, dtype=object)
        rs   = arr[:, 0].astype(int)
        cs   = arr[:, 1].astype(int)
        dist = _np.abs(rs - row) + _np.abs(cs - col)
        idx  = int(dist.argmin())
        return idx if dist[idx] <= threshold else -1

    # ── High-level operations ────────────────────────────────────────────────

    @classmethod
    def place(cls, state, matrix_attr, store_attr, history_attr,
              row, col, gid, mirroring):
        """Place gid at (row,col) and all mirrored positions."""
        matrix = cls._ensure_matrix(state, matrix_attr)
        store  = cls._store(state, store_attr)
        hist   = cls._store(state, history_attr)

        h, w = matrix.shape
        hist.append(list(store))   # snapshot for undo

        positions = _get_mirrored_positions(row, col, h, w, mirroring)
        import numpy as _np
        rs = _np.array([p[0] for p in positions])
        cs = _np.array([p[1] for p in positions])
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w) & (matrix[rs, cs] == 0)
        matrix[rs[valid], cs[valid]] = gid
        for r, c in zip(rs[valid], cs[valid]):
            store.append((int(r), int(c), gid))

    @classmethod
    def remove(cls, state, matrix_attr, store_attr, row, col, threshold=3):
        """Remove the closest placement to (row, col)."""
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        if matrix is None or not store:
            return
        idx = cls.find_closest(store, row, col, threshold)
        if idx < 0:
            return
        r, c, g = store.pop(idx)
        h, w = matrix.shape
        if 0 <= r < h and 0 <= c < w and matrix[r, c] == g:
            matrix[r, c] = 0

    @classmethod
    def clear(cls, state, matrix_attr, store_attr, history_attr):
        """Clear all placements."""
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        cls.clear_matrix(matrix, store)
        store.clear()
        cls._store(state, history_attr).clear()

    @classmethod
    def undo(cls, state, matrix_attr, store_attr, history_attr):
        """Undo last placement batch."""
        hist = cls._store(state, history_attr)
        if not hist:
            return
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        cls.clear_matrix(matrix, store)
        store.clear()
        store.extend(hist.pop())
        if matrix is not None:
            cls.write_matrix(matrix, store)


# ---------------------------------------------------------------------------
# GID resolution helper
# ---------------------------------------------------------------------------

def _resolve_gid(params: dict, role: str) -> int:
    """Compute final GID from params.

    Accepts either:
        { gid: 363 }                          → use directly (backward compat)
        { tileset: "MyTank", local_id: 3 }   → firstgid + local_id from registry
        { local_id: 3 }                       → use active tileset for role
    """
    if "gid" in params:
        return int(params["gid"])

    local_id = int(params.get("local_id", 0))

    # Explicit tileset name
    ts_name = params.get("tileset")
    if ts_name:
        entry = _reg.get_tileset(ts_name)
        if entry is None:
            raise ValueError(f"Tileset '{ts_name}' not in registry")
        return entry["firstgid"] + local_id

    # Fall back to active tileset for this role
    ts_info = state.active_tilesets.get(role)
    if ts_info is None:
        raise ValueError(f"No active '{role}' tileset. Activate one first or pass 'tileset' name.")
    return ts_info["firstgid"] + local_id


# ---------------------------------------------------------------------------
# Custom layer RPCs — generated by factory (units + items share same logic)
# ---------------------------------------------------------------------------

def _make_layer_rpcs(role, matrix_attr, store_attr, hist_attr):
    """Factory: produce (place, remove, clear, undo) RPC functions for a custom layer."""

    def place(params_json="{}"):
        p         = _params(params_json)
        row       = int(p.get("row", 0))
        col       = int(p.get("col", 0))
        mirroring = "none" if p.get("no_mirror") else state.mirroring
        gid       = _resolve_gid(p, role)
        _CustomLayer.place(state, matrix_attr, store_attr, hist_attr,
                           row, col, gid, mirroring)
        return {"snapshot": _snapshot()}

    def remove(params_json="{}"):
        p = _params(params_json)
        _CustomLayer.remove(state, matrix_attr, store_attr,
                            int(p.get("row", 0)), int(p.get("col", 0)))
        return {"snapshot": _snapshot()}

    def clear(params_json="{}"):
        _CustomLayer.clear(state, matrix_attr, store_attr, hist_attr)
        return {"snapshot": _snapshot()}

    def undo(params_json="{}"):
        _CustomLayer.undo(state, matrix_attr, store_attr, hist_attr)
        return {"snapshot": _snapshot()}

    return place, remove, clear, undo


(place_custom_unit, remove_custom_unit,
 clear_custom_units, undo_custom_unit)  = _make_layer_rpcs(
    "unit",  "units_matrix", "_custom_units",  "_custom_unit_history")

(place_custom_item, remove_custom_item,
 clear_custom_items, undo_custom_item) = _make_layer_rpcs(
    "items", "items_matrix", "_custom_items", "_custom_item_history")


# ---------------------------------------------------------------------------
# Tile picker helper — returns individual tile PNGs for active tileset
# ---------------------------------------------------------------------------

def list_active_tiles(params_json="{}"):
    """Return tile picker data for an active tileset.

    Params:
        role  – "unit" | "items" | "bridge"

    Returns:
    {
      "name":      "MyTank",
      "firstgid":  377,
      "tilecount": 8,
      "columns":   4,
      "tilewidth": 32,
      "tileheight": 32,
      "tiles": ["<base64_png>", ...]   ← one per tile, for UI grid
    }
    """
    import tile_analyzer as _ta
    params = _params(params_json)
    role   = str(params.get("role", "unit"))

    if role == "bridge":
        ts_info = state.bridge_tilesets.get(state.active_bridge_name)
    else:
        ts_info = state.active_tilesets.get(role)

    if not ts_info or not ts_info.get("png"):
        raise ValueError(f"No active '{role}' tileset with PNG. Activate one first.")

    tiles_b64 = _ta.extract_tiles_b64(
        ts_info["png"],
        ts_info["tilecount"],
        ts_info["columns"],
        ts_info.get("tilewidth",  20),
        ts_info.get("tileheight", 20),
    )

    return {
        "name":      ts_info["name"],
        "firstgid":  ts_info["firstgid"],
        "tilecount": ts_info["tilecount"],
        "columns":   ts_info["columns"],
        "tilewidth": ts_info.get("tilewidth",  20),
        "tileheight": ts_info.get("tileheight", 20),
        "tiles":     tiles_b64,
    }




def run_coastline(params_json="{}"):
    params = _params(params_json)
    state.initial_matrix = params.get("grid", state.initial_matrix)
    if state.initial_matrix is None:
        raise ValueError("Missing 'grid' for run_coastline")
    state.initial_matrix = _sanitize_binary_grid(state.initial_matrix)
    _apply_state_params(state, params)
    state._smoothness = float(params.get("shoreline_smoothness", 0.0))
    state.invalidate_from(WizardStep.COASTLINE)
    state.coast_smooth_passes = int(params.get("coastSmoothPasses", params.get("coast_smooth_passes", getattr(state, "coast_smooth_passes", 1))))
    frames = []
    def cb(label, matrix):
        frames.append(_preview_frame(label, matrix))
    map_pipeline.run_coastline(state, preview_cb=cb)
    return {"snapshot": _snapshot(), "frames": frames}


def _expand_brush(row, col, brush_size, h, w):
    r = brush_size // 2
    pts = []
    for dr in range(-r, r+1):
        for dc in range(-r, r+1):
            nr, nc = row+dr, col+dc
            if 0<=nr<h and 0<=nc<w: pts.append((nr,nc))
    return pts


def draw_walls(params_json="{}"):
    params = _params(params_json)
    points     = params.get("points", [])
    value      = int(params.get("value", 1))
    brush_size = int(params.get("brush_size", 1))
    h, w = int(state.height), int(state.width)
    import numpy as _np
    if state.wall_matrix is None or state.wall_matrix.shape != (h, w):
        state.wall_matrix = _np.zeros((h, w), dtype=int)
    coast = state.coastline_height_map
    no_mirror = bool(params.get("no_mirror", False))
    effective_mirroring = "none" if no_mirror else state.mirroring

    for pt in points:
        row, col = int(pt[0]), int(pt[1])
        for br, bc in _expand_brush(row, col, brush_size, h, w):
            for mr, mc in _get_mirrored_positions(br, bc, h, w, effective_mirroring):
                if 0<=mr<h and 0<=mc<w:
                    if value == 0:
                        state.wall_matrix[mr, mc] = 0
                    elif coast is not None and int(coast[mr, mc]) <= 0:
                        continue
                    elif value == 2 and int(state.wall_matrix[mr, mc]) == 0:
                        continue
                    else:
                        state.wall_matrix[mr, mc] = value
    return {"snapshot": _snapshot()}


def clear_walls(params_json="{}"):
    import numpy as _np
    h, w = int(state.height), int(state.width)
    state.wall_matrix = _np.zeros((h, w), dtype=int)
    return {"snapshot": _snapshot()}


def run_height_ocean(params_json="{}"):
    params = _params(params_json)
    state.num_height_levels = int(params.get("heightLevels", params.get("num_height_levels", state.num_height_levels)))
    state.num_ocean_levels  = int(params.get("oceanLevels",  params.get("num_ocean_levels",  state.num_ocean_levels)))
    hrs = params.get("heightRegionScale", params.get("height_region_scale", getattr(state, "height_region_scale", 1.0)))
    try:
        state.height_region_scale = max(0.1, min(3.0, float(hrs)))
    except Exception:
        state.height_region_scale = 1.0
    seed = params.get("seed")
    state.invalidate_from(WizardStep.HEIGHT_OCEAN)
    map_pipeline.run_height_ocean(state, seed=seed)
    state._height_map_pre_bridge = None
    if getattr(state, "bridge_matrix", None) is not None:
        import numpy as _np
        if _np.any(state.bridge_matrix > 0):
            _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}


def place_cc_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1)); col = int(params.get("col", -1)); mir = bool(params.get("mirrored", True))
    placed, evicted_count = map_pipeline.run_place_cc_manual(state, row, col, mir)
    result = {"placed": [[int(r),int(c)] for r,c in placed], "snapshot": _snapshot()}
    if evicted_count > 0:
        result["evicted"] = evicted_count
        result["evicted_warning"] = (
            f"{evicted_count} oldest command center group(s) were removed "
            f"to stay within the 10-CC limit."
        )
    return result

def remove_cc_manual(params_json="{}"):
    params = _params(params_json)
    map_pipeline.run_remove_cc_manual(state, int(params.get("row", -1)), int(params.get("col", -1)))
    return {"snapshot": _snapshot()}

def place_cc_random(params_json="{}"):
    params = _params(params_json)
    state.num_command_centers = int(params.get("numPlayers", params.get("num_command_centers", state.num_command_centers)))
    map_pipeline.clear_all_cc(state); map_pipeline.run_place_cc_random(state)
    return {"snapshot": _snapshot()}

def undo_cc(params_json="{}"):
    map_pipeline.undo_last_cc(state); return {"snapshot": _snapshot()}

def clear_cc(params_json="{}"):
    map_pipeline.clear_all_cc(state); return {"snapshot": _snapshot()}

def place_resource_manual(params_json="{}"):
    params = _params(params_json)
    placed = map_pipeline.run_place_resource_manual(state, int(params.get("row", -1)), int(params.get("col", -1)), bool(params.get("mirrored", True)))
    return {"placed": [[int(r),int(c)] for r,c in placed], "snapshot": _snapshot()}

def remove_resource_manual(params_json="{}"):
    params = _params(params_json)
    map_pipeline.run_remove_resource_manual(state, int(params.get("row", -1)), int(params.get("col", -1)))
    return {"snapshot": _snapshot()}

def place_resource_random(params_json="{}"):
    params = _params(params_json)
    state.num_resource_pulls = int(params.get("numResources", params.get("num_resource_pulls", state.num_resource_pulls)))
    map_pipeline.clear_all_resources(state); map_pipeline.run_place_resources_random(state)
    return {"snapshot": _snapshot()}

def undo_resource(params_json="{}"):
    map_pipeline.undo_last_resource(state); return {"snapshot": _snapshot()}

def clear_resource(params_json="{}"):
    map_pipeline.clear_all_resources(state); return {"snapshot": _snapshot()}

def get_state_snapshot(params_json="{}"):
    return {"snapshot": _snapshot()}

def run_finalize(params_json="{}"):
    params = _params(params_json)
    bxml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not bxml: raise ValueError("Missing blueprintXml")
    map_pipeline.run_finalize(state)
    tmx = map_pipeline.write_tmx(state, bxml)
    return {"tmx_bytes": tmx, "snapshot": _snapshot()}

def quick_generate(params_json="{}"):
    global state
    params = _params(params_json)
    state = WizardState()
    state.initial_matrix = params.get("grid", state.initial_matrix)
    if state.initial_matrix is None:
        raise ValueError("Missing 'grid' for quick_generate")
    _apply_state_params(state, params)
    bxml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not bxml: raise ValueError("Missing blueprintXml")
    qf = []
    def _fr(label, hm, idm=None, im=None, um=None):
        f = {"label": label, "height_map": _matrix_payload(hm)}
        if idm is not None:
            f["id_matrix"] = _matrix_payload(idm)
        if im is not None:
            f["items_matrix"] = _matrix_payload(im)
        if um is not None:
            f["units_matrix"] = _matrix_payload(um)
        qf.append(f)
    def ccb(label, matrix): _fr(label, matrix)
    map_pipeline.run_coastline(state, preview_cb=ccb)
    map_pipeline.run_height_ocean(state)
    _fr("height_ocean", state.height_map)
    map_pipeline.run_place_cc_random(state)
    _fr("command_centers", state.height_map, um=state.units_matrix)
    map_pipeline.run_place_resources_random(state)
    _fr("resources", state.height_map, im=state.items_matrix, um=state.units_matrix)
    map_pipeline.run_finalize(state)
    tmx = map_pipeline.write_tmx(state, bxml)
    return {"tmx_bytes": tmx, "snapshot": _snapshot(), "quick_frames": qf}

def reset_state(params_json="{}"):
    """Reset map generation state but preserve registered tilesets."""
    global state
    old_bridge_tilesets  = dict(state.bridge_tilesets)
    old_active_bridge    = state.active_bridge_name
    old_active_tilesets  = dict(state.active_tilesets)
    state = WizardState()
    state.bridge_tilesets   = old_bridge_tilesets
    state.active_bridge_name = old_active_bridge
    state.active_tilesets   = old_active_tilesets
    return {"snapshot": _snapshot()}


def draw_bridge(params_json="{}"):
    """Place bridge tiles on the map.

    Routing:
      variant == "simple"  → bridge_simple.place  (3-category: cap_start/span/cap_end)
      variant == "6"|"12"  → bridge_pipeline.place_stroke (preset roles)
    """
    params    = _params(params_json)
    points    = [(int(p[0]), int(p[1])) for p in params.get("points", [])]
    erase     = bool(params.get("erase", False))
    direction = str(params.get("direction", "auto"))

    bts    = getattr(state, "bridge_tilesets", {})
    active = getattr(state, "active_bridge_name", "")
    ts     = bts.get(active) or next(iter(bts.values()), None)

    variant     = (ts.get("bridge_variant", "12") if ts else "12")
    simple_data = ts.get("bridge_simple") if ts else None   # new format

    if variant == "simple" and simple_data:
        import bridge_simple as _bs
        layout = _bs.BridgeLayout.from_dict(simple_data)
        # Override direction from stored preference if auto
        if direction in ("", "auto"):
            direction = layout.direction
        else:
            layout.direction = direction
        _bs.place(state, points, layout, erase=erase)
    else:
        import bridge_pipeline
        bridge_pipeline.place_stroke(state, points, erase=erase, direction=direction)

    _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}

def clear_bridge(params_json="{}"):
    import bridge_pipeline
    bridge_pipeline.clear(state)
    _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}


def _apply_bridge_height_boost(st):
    """Boost height_map for bridge cells so minimap and auto-terrain show land.

    - Span cells (all bridge tiles)  → h = SPAN_H (5 = stone, matches gray ground tile)
    - Approach ramp (±APPROACH_DEPTH tiles beyond endcap) → h = APPROACH_H (1 = sand)
    """
    import numpy as _np
    from scipy.ndimage import label as _label_segs

    hm = st.height_map
    bm = getattr(st, "bridge_matrix", None)
    if hm is None or bm is None or not _np.any(bm > 0):
        return

    if not hasattr(st, "_height_map_pre_bridge") or st._height_map_pre_bridge is None:
        st._height_map_pre_bridge = hm.copy()

    base   = st._height_map_pre_bridge
    result = base.copy()

    APPROACH_DEPTH = 3
    APPROACH_H     = 1
    SPAN_H         = 5   # stone level — matches BRIDGE_GROUND_LOCAL=46 in ground layer
    ROTFLAG        = _np.int64(0xA0000000)

    # Get firstgid from active bridge tileset (new multi-tileset system)
    bts    = getattr(st, "bridge_tilesets", {})
    active = getattr(st, "active_bridge_name", "")
    ts     = bts.get(active) or next(iter(bts.values()), None) \
             or getattr(st, "bridge_tileset", None)
    fg = int(ts.get("firstgid", 1)) if ts else 1

    # Compute local IDs (strip rotation flag, subtract firstgid)
    bm_local = _np.where(
        bm > 0,
        (bm.astype(_np.int64) & ~ROTFLAG) - int(fg),
        _np.int64(-1)
    )

    # Detect endcap tile local IDs from tile mapping
    tile_map = ts.get("tiles", {}) if ts else {}
    endcap_keys = {"NW","NE","SW","SE","W_top","W_bot","E_top","E_bot"}
    ENDCAP_IDS  = set(tile_map[k] for k in endcap_keys if k in tile_map)
    if not ENDCAP_IDS:
        # Fallback: use default 12-tile layout endcap positions
        ENDCAP_IDS = {0, 2, 3, 5, 6, 8, 9, 11}

    H, W = base.shape
    labeled, n_segs = _label_segs((bm > 0).astype(_np.int32))

    for seg_id in range(1, n_segs + 1):
        seg_mask = labeled == seg_id

        # ALL bridge cells → boost to at least SPAN_H so ground layer = stone (gray)
        result[seg_mask] = _np.maximum(result[seg_mask], SPAN_H)

        # Approach ramp
        seg_rows, seg_cols = _np.where(seg_mask)
        seg_r_min, seg_r_max    = int(seg_rows.min()), int(seg_rows.max())
        seg_c_left, seg_c_right = int(seg_cols.min()), int(seg_cols.max())
        is_vertical = (seg_r_max - seg_r_min) > (seg_c_right - seg_c_left)

        endcap_mask = seg_mask & _np.isin(bm_local, list(ENDCAP_IDS))
        ec_rows, ec_cols = _np.where(endcap_mask)
        if len(ec_rows) == 0:
            # No endcaps found — use outermost cells as ramp anchors
            ec_rows, ec_cols = seg_rows, seg_cols

        map_pipeline._stamp_approach_ramp(
            is_vertical,
            int(ec_rows.min()), int(ec_rows.max()),
            int(ec_cols.min()), int(ec_cols.max()),
            H, W, base, result,
            APPROACH_H, SPAN_H, APPROACH_DEPTH,
        )

    st.height_map = result


# ---------------------------------------------------------------------------
# Registry RPCs — tileset register / activate / deactivate / list
# ---------------------------------------------------------------------------

def register_tileset(params_json="{}"):
    """Register or update a tileset in the persistent registry.

    Required: name, type (bridge|unit|terrain|items), png (base64), tilecount, columns
    Optional: tilewidth, tileheight, tiles, activate, bridge_variant,
              bridge_simple, bridge_layout, bridge_custom_dir, layout_rows, layout_cols
    """
    params            = _params(params_json)
    name              = str(params["name"])
    ts_type           = str(params["type"])
    png_b64           = str(params["png"])
    tilecount         = int(params["tilecount"])
    columns           = int(params["columns"])
    tilewidth         = int(params.get("tilewidth",  20))
    tileheight        = int(params.get("tileheight", 20))
    tiles             = params.get("tiles")
    activate          = bool(params.get("activate", False))
    bridge_variant    = str(params.get("bridge_variant", "12"))
    bridge_layout     = params.get("bridge_layout")
    bridge_custom_dir = str(params.get("bridge_custom_dir", ""))
    bridge_simple     = params.get("bridge_simple")
    layout_rows       = params.get("layout_rows")
    layout_cols       = params.get("layout_cols")

    entry = _reg.register_tileset(
        name=name, tileset_type=ts_type, png_b64=png_b64,
        tilecount=tilecount, columns=columns,
        tilewidth=tilewidth, tileheight=tileheight,
        tiles=tiles, bridge_variant=bridge_variant,
        bridge_layout=bridge_layout,
        bridge_custom_dir=bridge_custom_dir,
        bridge_simple=bridge_simple,
    )

    if ts_type == "bridge" and activate:
        state.bridge_tilesets[name] = {
            "firstgid":          entry["firstgid"],
            "tilecount":         entry["tilecount"],
            "columns":           entry["columns"],
            "tilewidth":         entry["tilewidth"],
            "tileheight":        entry["tileheight"],
            "name":              name,
            "png":               png_b64,
            "tiles":             entry.get("tiles", {}),
            "bridge_variant":    entry.get("bridge_variant", "12"),
            "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
            "bridge_layout":     entry.get("bridge_layout"),
            "layout_rows":       entry.get("layout_rows", 2),
            "layout_cols":       entry.get("layout_cols", 3),
            "bridge_simple":     entry.get("bridge_simple"),
        }
        state.active_bridge_name = name
    elif ts_type in ("terrain", "unit", "items") and activate:
        state.active_tilesets[ts_type] = {
            "name":      name,
            "firstgid":  entry["firstgid"],
            "tilecount": entry["tilecount"],
            "columns":   entry["columns"],
            "tilewidth": entry["tilewidth"],
            "tileheight":entry["tileheight"],
            "png":       png_b64,
            "tiles":     entry.get("tiles", {}),
            "bridge_variant":    entry.get("bridge_variant", "12"),
            "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
            "bridge_simple":     entry.get("bridge_simple"),
        }

    response = {k: v for k, v in entry.items() if k != "png"}
    return {"registered": response, "snapshot": _snapshot()}


def list_tilesets(params_json="{}"):
    """List registered tilesets, optionally filtered by type."""
    params    = _params(params_json)
    ts_type   = params.get("type")
    all_ts    = _reg.get_all()
    if ts_type:
        all_ts = {k: v for k, v in all_ts.items() if v.get("type") == ts_type}
    # Strip PNG from listing response
    result = {}
    for name, entry in all_ts.items():
        result[name] = {k: v for k, v in entry.items() if k != "png"}
    return {"tilesets": result}


def activate_tileset(params_json="{}"):
    """Load an existing registry tileset into live state."""
    params  = _params(params_json)
    name    = str(params["name"])
    role    = str(params.get("role", params.get("type", "")))
    entry   = _reg.get_tileset(name)
    if entry is None:
        raise ValueError(f"Tileset '{name}' not in registry")

    ts_info = {
        "name":              name,
        "firstgid":          entry["firstgid"],
        "tilecount":         entry["tilecount"],
        "columns":           entry["columns"],
        "tilewidth":         entry.get("tilewidth",  20),
        "tileheight":        entry.get("tileheight", 20),
        "png":               entry.get("png", ""),
        "tiles":             entry.get("tiles", {}),
        "bridge_variant":    entry.get("bridge_variant", "12"),
        "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
        "bridge_layout":     entry.get("bridge_layout"),
        "bridge_simple":     entry.get("bridge_simple"),
    }

    ts_type = entry.get("type", role)
    if ts_type == "bridge":
        state.bridge_tilesets[name] = ts_info
        state.active_bridge_name    = name
    else:
        state.active_tilesets[ts_type] = ts_info
        if ts_type == "terrain":
            state.custom_terrain_mapping = None

    return {"activated": name, "role": ts_type, "snapshot": _snapshot()}


def deactivate_tileset(params_json="{}"):
    """Remove a tileset from live state (does not delete from registry)."""
    params = _params(params_json)
    role   = str(params["role"])

    if role == "bridge":
        name = params.get("name", state.active_bridge_name)
        state.bridge_tilesets.pop(name, None)
        if state.active_bridge_name == name:
            state.active_bridge_name = next(iter(state.bridge_tilesets), "")
    else:
        state.active_tilesets.pop(role, None)
        if role == "terrain":
            state.custom_terrain_mapping = None

    return {"deactivated": role, "snapshot": _snapshot()}


RPC_METHODS = {
    "register_tileset":   register_tileset,
    "list_tilesets":      list_tilesets,
    "activate_tileset":   activate_tileset,
    "deactivate_tileset": deactivate_tileset,
    "list_active_tiles":  list_active_tiles,
    "draw_bridge":        draw_bridge,
    "clear_bridge":       clear_bridge,
    "place_custom_unit":  place_custom_unit,
    "remove_custom_unit": remove_custom_unit,
    "clear_custom_units": clear_custom_units,
    "undo_custom_unit":   undo_custom_unit,
    "place_custom_item":  place_custom_item,
    "remove_custom_item": remove_custom_item,
    "clear_custom_items": clear_custom_items,
    "undo_custom_item":   undo_custom_item,
    "run_coastline":      run_coastline,
    "draw_walls":         draw_walls,
    "clear_walls":        clear_walls,
    "run_height_ocean":   run_height_ocean,
    "place_cc_manual":    place_cc_manual,
    "remove_cc_manual":   remove_cc_manual,
    "place_cc_random":    place_cc_random,
    "undo_cc":            undo_cc,
    "clear_cc":           clear_cc,
    "place_resource_manual":  place_resource_manual,
    "remove_resource_manual": remove_resource_manual,
    "place_resource_random":  place_resource_random,
    "undo_resource":      undo_resource,
    "clear_resource":     clear_resource,
    "get_state_snapshot": get_state_snapshot,
    "run_finalize":       run_finalize,
    "quick_generate":     quick_generate,
    "reset_state":        reset_state,
}


def rpc_call(method, params_json="{}"):
    if method not in RPC_METHODS: raise ValueError(f"Unknown RPC method: {method}")
    return RPC_METHODS[method](params_json)


def _params(p):
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    try:
        return json.loads(p or "{}")
    except Exception as exc:
        raise ValueError(f"Invalid JSON params: {exc}") from exc


def _apply_state_params(state, params):
    """Apply the 8 common wizard params from a request dict onto state."""
    state.height    = max(20, min(640, int(params.get("height",    state.height))))
    state.width     = max(20, min(640, int(params.get("width",     state.width))))
    state.mirroring = str(params.get("mirroring", state.mirroring))
    state.pattern   = int(params.get("tileset",   params.get("pattern",              state.pattern)))
    state.num_height_levels   = int(params.get("heightLevels",  params.get("num_height_levels",   state.num_height_levels)))
    state.num_ocean_levels    = int(params.get("oceanLevels",   params.get("num_ocean_levels",    state.num_ocean_levels)))
    hrs = params.get("heightRegionScale", params.get("height_region_scale", getattr(state, "height_region_scale", 1.0)))
    try:
        state.height_region_scale = max(0.4, min(2.5, float(hrs)))
    except Exception:
        state.height_region_scale = 1.0
    state.num_command_centers = int(params.get("numPlayers",    params.get("num_command_centers", state.num_command_centers)))
    state.num_resource_pulls  = int(params.get("numResources",  params.get("num_resource_pulls",  state.num_resource_pulls)))


def _clear_custom_units_from_matrix(state):
    """Zero out all custom unit cells in units_matrix."""
    _CustomLayer.clear_matrix(state.units_matrix, state._custom_units)


# ---------------------------------------------------------------------------
# _CustomLayer — shared vectorized logic for unit + item custom placements
# ---------------------------------------------------------------------------

class _CustomLayer:
    """Vectorized place / remove / clear / undo for a custom placement layer.

    All matrix writes use NumPy advanced indexing — no per-cell Python loops.

    Attributes on state:
        matrix_attr   – e.g. "units_matrix" or "items_matrix"
        store_attr    – e.g. "_custom_units" or "_custom_items"
        history_attr  – e.g. "_custom_unit_history" or "_custom_item_history"
    """

    @staticmethod
    def _ensure_matrix(state, attr: str) -> "np.ndarray":
        import numpy as _np
        m = getattr(state, attr)
        if m is None:
            h, w = int(state.height), int(state.width)
            m = _np.zeros((h, w), dtype=int)
            setattr(state, attr, m)
        return m

    @staticmethod
    def _store(state, attr: str) -> list:
        return getattr(state, attr)

    @staticmethod
    def clear_matrix(matrix, store: list):
        """Zero out matrix cells listed in store — vectorized."""
        if matrix is None or not store:
            return
        import numpy as _np
        arr = _np.array(store, dtype=object)   # (N, 3)
        rs  = arr[:, 0].astype(int)
        cs  = arr[:, 1].astype(int)
        gs  = arr[:, 2].astype(int)
        h, w = matrix.shape
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        # Only clear cells whose GID still matches (don't clobber other layers)
        match = valid & (matrix[rs.clip(0, h-1), cs.clip(0, w-1)] == gs)
        matrix[rs[match], cs[match]] = 0

    @staticmethod
    def write_matrix(matrix, store: list):
        """Write all (r,c,gid) entries to matrix — vectorized."""
        if not store:
            return
        import numpy as _np
        arr   = _np.array(store, dtype=object)
        rs    = arr[:, 0].astype(int)
        cs    = arr[:, 1].astype(int)
        gs    = arr[:, 2].astype(int)
        h, w  = matrix.shape
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w)
        matrix[rs[valid], cs[valid]] = gs[valid]

    @staticmethod
    def find_closest(store: list, row: int, col: int, threshold: int = 3) -> int:
        """Return index of closest entry within Manhattan threshold, or -1.

        Vectorized distance computation — no Python loop.
        """
        if not store:
            return -1
        import numpy as _np
        arr  = _np.array(store, dtype=object)
        rs   = arr[:, 0].astype(int)
        cs   = arr[:, 1].astype(int)
        dist = _np.abs(rs - row) + _np.abs(cs - col)
        idx  = int(dist.argmin())
        return idx if dist[idx] <= threshold else -1

    # ── High-level operations ────────────────────────────────────────────────

    @classmethod
    def place(cls, state, matrix_attr, store_attr, history_attr,
              row, col, gid, mirroring):
        """Place gid at (row,col) and all mirrored positions."""
        matrix = cls._ensure_matrix(state, matrix_attr)
        store  = cls._store(state, store_attr)
        hist   = cls._store(state, history_attr)

        h, w = matrix.shape
        hist.append(list(store))   # snapshot for undo

        positions = _get_mirrored_positions(row, col, h, w, mirroring)
        import numpy as _np
        rs = _np.array([p[0] for p in positions])
        cs = _np.array([p[1] for p in positions])
        valid = (rs >= 0) & (rs < h) & (cs >= 0) & (cs < w) & (matrix[rs, cs] == 0)
        matrix[rs[valid], cs[valid]] = gid
        for r, c in zip(rs[valid], cs[valid]):
            store.append((int(r), int(c), gid))

    @classmethod
    def remove(cls, state, matrix_attr, store_attr, row, col, threshold=3):
        """Remove the closest placement to (row, col)."""
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        if matrix is None or not store:
            return
        idx = cls.find_closest(store, row, col, threshold)
        if idx < 0:
            return
        r, c, g = store.pop(idx)
        h, w = matrix.shape
        if 0 <= r < h and 0 <= c < w and matrix[r, c] == g:
            matrix[r, c] = 0

    @classmethod
    def clear(cls, state, matrix_attr, store_attr, history_attr):
        """Clear all placements."""
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        cls.clear_matrix(matrix, store)
        store.clear()
        cls._store(state, history_attr).clear()

    @classmethod
    def undo(cls, state, matrix_attr, store_attr, history_attr):
        """Undo last placement batch."""
        hist = cls._store(state, history_attr)
        if not hist:
            return
        matrix = getattr(state, matrix_attr)
        store  = cls._store(state, store_attr)
        cls.clear_matrix(matrix, store)
        store.clear()
        store.extend(hist.pop())
        if matrix is not None:
            cls.write_matrix(matrix, store)


# ---------------------------------------------------------------------------
# GID resolution helper
# ---------------------------------------------------------------------------

def _resolve_gid(params: dict, role: str) -> int:
    """Compute final GID from params.

    Accepts either:
        { gid: 363 }                          → use directly (backward compat)
        { tileset: "MyTank", local_id: 3 }   → firstgid + local_id from registry
        { local_id: 3 }                       → use active tileset for role
    """
    if "gid" in params:
        return int(params["gid"])

    local_id = int(params.get("local_id", 0))

    # Explicit tileset name
    ts_name = params.get("tileset")
    if ts_name:
        entry = _reg.get_tileset(ts_name)
        if entry is None:
            raise ValueError(f"Tileset '{ts_name}' not in registry")
        return entry["firstgid"] + local_id

    # Fall back to active tileset for this role
    ts_info = state.active_tilesets.get(role)
    if ts_info is None:
        raise ValueError(f"No active '{role}' tileset. Activate one first or pass 'tileset' name.")
    return ts_info["firstgid"] + local_id


# ---------------------------------------------------------------------------
# Custom layer RPCs — generated by factory (units + items share same logic)
# ---------------------------------------------------------------------------

def _make_layer_rpcs(role, matrix_attr, store_attr, hist_attr):
    """Factory: produce (place, remove, clear, undo) RPC functions for a custom layer."""

    def place(params_json="{}"):
        p         = _params(params_json)
        row       = int(p.get("row", 0))
        col       = int(p.get("col", 0))
        mirroring = "none" if p.get("no_mirror") else state.mirroring
        gid       = _resolve_gid(p, role)
        _CustomLayer.place(state, matrix_attr, store_attr, hist_attr,
                           row, col, gid, mirroring)
        return {"snapshot": _snapshot()}

    def remove(params_json="{}"):
        p = _params(params_json)
        _CustomLayer.remove(state, matrix_attr, store_attr,
                            int(p.get("row", 0)), int(p.get("col", 0)))
        return {"snapshot": _snapshot()}

    def clear(params_json="{}"):
        _CustomLayer.clear(state, matrix_attr, store_attr, hist_attr)
        return {"snapshot": _snapshot()}

    def undo(params_json="{}"):
        _CustomLayer.undo(state, matrix_attr, store_attr, hist_attr)
        return {"snapshot": _snapshot()}

    return place, remove, clear, undo


(place_custom_unit, remove_custom_unit,
 clear_custom_units, undo_custom_unit)  = _make_layer_rpcs(
    "unit",  "units_matrix", "_custom_units",  "_custom_unit_history")

(place_custom_item, remove_custom_item,
 clear_custom_items, undo_custom_item) = _make_layer_rpcs(
    "items", "items_matrix", "_custom_items", "_custom_item_history")


# ---------------------------------------------------------------------------
# Tile picker helper — returns individual tile PNGs for active tileset
# ---------------------------------------------------------------------------

def list_active_tiles(params_json="{}"):
    """Return tile picker data for an active tileset.

    Params:
        role  – "unit" | "items" | "bridge"

    Returns:
    {
      "name":      "MyTank",
      "firstgid":  377,
      "tilecount": 8,
      "columns":   4,
      "tilewidth": 32,
      "tileheight": 32,
      "tiles": ["<base64_png>", ...]   ← one per tile, for UI grid
    }
    """
    import tile_analyzer as _ta
    params = _params(params_json)
    role   = str(params.get("role", "unit"))

    if role == "bridge":
        ts_info = state.bridge_tilesets.get(state.active_bridge_name)
    else:
        ts_info = state.active_tilesets.get(role)

    if not ts_info or not ts_info.get("png"):
        raise ValueError(f"No active '{role}' tileset with PNG. Activate one first.")

    tiles_b64 = _ta.extract_tiles_b64(
        ts_info["png"],
        ts_info["tilecount"],
        ts_info["columns"],
        ts_info.get("tilewidth",  20),
        ts_info.get("tileheight", 20),
    )

    return {
        "name":      ts_info["name"],
        "firstgid":  ts_info["firstgid"],
        "tilecount": ts_info["tilecount"],
        "columns":   ts_info["columns"],
        "tilewidth": ts_info.get("tilewidth",  20),
        "tileheight": ts_info.get("tileheight", 20),
        "tiles":     tiles_b64,
    }




def run_coastline(params_json="{}"):
    params = _params(params_json)
    state.initial_matrix = params.get("grid", state.initial_matrix)
    if state.initial_matrix is None:
        raise ValueError("Missing 'grid' for run_coastline")
    state.initial_matrix = _sanitize_binary_grid(state.initial_matrix)
    _apply_state_params(state, params)
    state._smoothness = float(params.get("shoreline_smoothness", 0.0))
    state.invalidate_from(WizardStep.COASTLINE)
    state.coast_smooth_passes = int(params.get("coastSmoothPasses", params.get("coast_smooth_passes", getattr(state, "coast_smooth_passes", 1))))
    frames = []
    def cb(label, matrix):
        frames.append(_preview_frame(label, matrix))
    map_pipeline.run_coastline(state, preview_cb=cb)
    return {"snapshot": _snapshot(), "frames": frames}


def _expand_brush(row, col, brush_size, h, w):
    r = brush_size // 2
    pts = []
    for dr in range(-r, r+1):
        for dc in range(-r, r+1):
            nr, nc = row+dr, col+dc
            if 0<=nr<h and 0<=nc<w: pts.append((nr,nc))
    return pts


def draw_walls(params_json="{}"):
    params = _params(params_json)
    points     = params.get("points", [])
    value      = int(params.get("value", 1))
    brush_size = int(params.get("brush_size", 1))
    h, w = int(state.height), int(state.width)
    import numpy as _np
    if state.wall_matrix is None or state.wall_matrix.shape != (h, w):
        state.wall_matrix = _np.zeros((h, w), dtype=int)
    coast = state.coastline_height_map
    no_mirror = bool(params.get("no_mirror", False))
    effective_mirroring = "none" if no_mirror else state.mirroring

    for pt in points:
        row, col = int(pt[0]), int(pt[1])
        for br, bc in _expand_brush(row, col, brush_size, h, w):
            for mr, mc in _get_mirrored_positions(br, bc, h, w, effective_mirroring):
                if 0<=mr<h and 0<=mc<w:
                    if value == 0:
                        state.wall_matrix[mr, mc] = 0
                    elif coast is not None and int(coast[mr, mc]) <= 0:
                        continue
                    elif value == 2 and int(state.wall_matrix[mr, mc]) == 0:
                        continue
                    else:
                        state.wall_matrix[mr, mc] = value
    return {"snapshot": _snapshot()}


def clear_walls(params_json="{}"):
    import numpy as _np
    h, w = int(state.height), int(state.width)
    state.wall_matrix = _np.zeros((h, w), dtype=int)
    return {"snapshot": _snapshot()}


def run_height_ocean(params_json="{}"):
    params = _params(params_json)
    state.num_height_levels = int(params.get("heightLevels", params.get("num_height_levels", state.num_height_levels)))
    state.num_ocean_levels  = int(params.get("oceanLevels",  params.get("num_ocean_levels",  state.num_ocean_levels)))
    hrs = params.get("heightRegionScale", params.get("height_region_scale", getattr(state, "height_region_scale", 1.0)))
    try:
        state.height_region_scale = max(0.4, min(2.5, float(hrs)))
    except Exception:
        state.height_region_scale = 1.0
    seed = params.get("seed")
    state.invalidate_from(WizardStep.HEIGHT_OCEAN)
    map_pipeline.run_height_ocean(state, seed=seed)
    state._height_map_pre_bridge = None
    if getattr(state, "bridge_matrix", None) is not None:
        import numpy as _np
        if _np.any(state.bridge_matrix > 0):
            _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}


def place_cc_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1)); col = int(params.get("col", -1)); mir = bool(params.get("mirrored", True))
    placed, evicted_count = map_pipeline.run_place_cc_manual(state, row, col, mir)
    result = {"placed": [[int(r),int(c)] for r,c in placed], "snapshot": _snapshot()}
    if evicted_count > 0:
        result["evicted"] = evicted_count
        result["evicted_warning"] = (
            f"{evicted_count} oldest command center group(s) were removed "
            f"to stay within the 10-CC limit."
        )
    return result

def remove_cc_manual(params_json="{}"):
    params = _params(params_json)
    map_pipeline.run_remove_cc_manual(state, int(params.get("row", -1)), int(params.get("col", -1)))
    return {"snapshot": _snapshot()}

def place_cc_random(params_json="{}"):
    params = _params(params_json)
    state.num_command_centers = int(params.get("numPlayers", params.get("num_command_centers", state.num_command_centers)))
    map_pipeline.clear_all_cc(state); map_pipeline.run_place_cc_random(state)
    return {"snapshot": _snapshot()}

def undo_cc(params_json="{}"):
    map_pipeline.undo_last_cc(state); return {"snapshot": _snapshot()}

def clear_cc(params_json="{}"):
    map_pipeline.clear_all_cc(state); return {"snapshot": _snapshot()}

def place_resource_manual(params_json="{}"):
    params = _params(params_json)
    placed = map_pipeline.run_place_resource_manual(state, int(params.get("row", -1)), int(params.get("col", -1)), bool(params.get("mirrored", True)))
    return {"placed": [[int(r),int(c)] for r,c in placed], "snapshot": _snapshot()}

def remove_resource_manual(params_json="{}"):
    params = _params(params_json)
    map_pipeline.run_remove_resource_manual(state, int(params.get("row", -1)), int(params.get("col", -1)))
    return {"snapshot": _snapshot()}

def place_resource_random(params_json="{}"):
    params = _params(params_json)
    state.num_resource_pulls = int(params.get("numResources", params.get("num_resource_pulls", state.num_resource_pulls)))
    map_pipeline.clear_all_resources(state); map_pipeline.run_place_resources_random(state)
    return {"snapshot": _snapshot()}

def undo_resource(params_json="{}"):
    map_pipeline.undo_last_resource(state); return {"snapshot": _snapshot()}

def clear_resource(params_json="{}"):
    map_pipeline.clear_all_resources(state); return {"snapshot": _snapshot()}

def get_state_snapshot(params_json="{}"):
    return {"snapshot": _snapshot()}

def run_finalize(params_json="{}"):
    params = _params(params_json)
    bxml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not bxml: raise ValueError("Missing blueprintXml")
    map_pipeline.run_finalize(state)
    tmx = map_pipeline.write_tmx(state, bxml)
    return {"tmx_bytes": tmx, "snapshot": _snapshot()}

def quick_generate(params_json="{}"):
    global state
    params = _params(params_json)
    state = WizardState()
    state.initial_matrix = params.get("grid", state.initial_matrix)
    if state.initial_matrix is None:
        raise ValueError("Missing 'grid' for quick_generate")
    _apply_state_params(state, params)
    bxml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not bxml: raise ValueError("Missing blueprintXml")
    qf = []
    def _fr(label, hm, idm=None, im=None, um=None):
        f = {"label": label, "height_map": _matrix_payload(hm)}
        if idm is not None:
            f["id_matrix"] = _matrix_payload(idm)
        if im is not None:
            f["items_matrix"] = _matrix_payload(im)
        if um is not None:
            f["units_matrix"] = _matrix_payload(um)
        qf.append(f)
    def ccb(label, matrix): _fr(label, matrix)
    map_pipeline.run_coastline(state, preview_cb=ccb)
    map_pipeline.run_height_ocean(state)
    _fr("height_ocean", state.height_map)
    map_pipeline.run_place_cc_random(state)
    _fr("command_centers", state.height_map, um=state.units_matrix)
    map_pipeline.run_place_resources_random(state)
    _fr("resources", state.height_map, im=state.items_matrix, um=state.units_matrix)
    map_pipeline.run_finalize(state)
    tmx = map_pipeline.write_tmx(state, bxml)
    return {"tmx_bytes": tmx, "snapshot": _snapshot(), "quick_frames": qf}

def reset_state(params_json="{}"):
    """Reset map generation state but preserve registered tilesets."""
    global state
    old_bridge_tilesets  = dict(state.bridge_tilesets)
    old_active_bridge    = state.active_bridge_name
    old_active_tilesets  = dict(state.active_tilesets)
    state = WizardState()
    state.bridge_tilesets   = old_bridge_tilesets
    state.active_bridge_name = old_active_bridge
    state.active_tilesets   = old_active_tilesets
    return {"snapshot": _snapshot()}


def draw_bridge(params_json="{}"):
    """Place bridge tiles on the map.

    Routing:
      variant == "simple"  → bridge_simple.place  (3-category: cap_start/span/cap_end)
      variant == "6"|"12"  → bridge_pipeline.place_stroke (preset roles)
    """
    params    = _params(params_json)
    points    = [(int(p[0]), int(p[1])) for p in params.get("points", [])]
    erase     = bool(params.get("erase", False))
    direction = str(params.get("direction", "auto"))

    bts    = getattr(state, "bridge_tilesets", {})
    active = getattr(state, "active_bridge_name", "")
    ts     = bts.get(active) or next(iter(bts.values()), None)

    variant     = (ts.get("bridge_variant", "12") if ts else "12")
    simple_data = ts.get("bridge_simple") if ts else None   # new format

    if variant == "simple" and simple_data:
        import bridge_simple as _bs
        layout = _bs.BridgeLayout.from_dict(simple_data)
        # Override direction from stored preference if auto
        if direction in ("", "auto"):
            direction = layout.direction
        else:
            layout.direction = direction
        _bs.place(state, points, layout, erase=erase)
    else:
        import bridge_pipeline
        bridge_pipeline.place_stroke(state, points, erase=erase, direction=direction)

    _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}

def clear_bridge(params_json="{}"):
    import bridge_pipeline
    bridge_pipeline.clear(state)
    _apply_bridge_height_boost(state)
    return {"snapshot": _snapshot()}


def _apply_bridge_height_boost(st):
    """Boost height_map for bridge cells so minimap and auto-terrain show land.

    - Span cells (all bridge tiles)  → h = SPAN_H (5 = stone, matches gray ground tile)
    - Approach ramp (±APPROACH_DEPTH tiles beyond endcap) → h = APPROACH_H (1 = sand)
    """
    import numpy as _np
    from scipy.ndimage import label as _label_segs

    hm = st.height_map
    bm = getattr(st, "bridge_matrix", None)
    if hm is None or bm is None or not _np.any(bm > 0):
        return

    if not hasattr(st, "_height_map_pre_bridge") or st._height_map_pre_bridge is None:
        st._height_map_pre_bridge = hm.copy()

    base   = st._height_map_pre_bridge
    result = base.copy()

    APPROACH_DEPTH = 3
    APPROACH_H     = 1
    SPAN_H         = 5   # stone level — matches BRIDGE_GROUND_LOCAL=46 in ground layer
    ROTFLAG        = _np.int64(0xA0000000)

    # Get firstgid from active bridge tileset (new multi-tileset system)
    bts    = getattr(st, "bridge_tilesets", {})
    active = getattr(st, "active_bridge_name", "")
    ts     = bts.get(active) or next(iter(bts.values()), None) \
             or getattr(st, "bridge_tileset", None)
    fg = int(ts.get("firstgid", 1)) if ts else 1

    # Compute local IDs (strip rotation flag, subtract firstgid)
    bm_local = _np.where(
        bm > 0,
        (bm.astype(_np.int64) & ~ROTFLAG) - int(fg),
        _np.int64(-1)
    )

    # Detect endcap tile local IDs from tile mapping
    tile_map = ts.get("tiles", {}) if ts else {}
    endcap_keys = {"NW","NE","SW","SE","W_top","W_bot","E_top","E_bot"}
    ENDCAP_IDS  = set(tile_map[k] for k in endcap_keys if k in tile_map)
    if not ENDCAP_IDS:
        # Fallback: use default 12-tile layout endcap positions
        ENDCAP_IDS = {0, 2, 3, 5, 6, 8, 9, 11}

    H, W = base.shape
    labeled, n_segs = _label_segs((bm > 0).astype(_np.int32))

    for seg_id in range(1, n_segs + 1):
        seg_mask = labeled == seg_id

        # ALL bridge cells → boost to at least SPAN_H so ground layer = stone (gray)
        result[seg_mask] = _np.maximum(result[seg_mask], SPAN_H)

        # Approach ramp
        seg_rows, seg_cols = _np.where(seg_mask)
        seg_r_min, seg_r_max    = int(seg_rows.min()), int(seg_rows.max())
        seg_c_left, seg_c_right = int(seg_cols.min()), int(seg_cols.max())
        is_vertical = (seg_r_max - seg_r_min) > (seg_c_right - seg_c_left)

        endcap_mask = seg_mask & _np.isin(bm_local, list(ENDCAP_IDS))
        ec_rows, ec_cols = _np.where(endcap_mask)
        if len(ec_rows) == 0:
            # No endcaps found — use outermost cells as ramp anchors
            ec_rows, ec_cols = seg_rows, seg_cols

        map_pipeline._stamp_approach_ramp(
            is_vertical,
            int(ec_rows.min()), int(ec_rows.max()),
            int(ec_cols.min()), int(ec_cols.max()),
            H, W, base, result,
            APPROACH_H, SPAN_H, APPROACH_DEPTH,
        )

    st.height_map = result


# ---------------------------------------------------------------------------
# Registry RPCs — tileset register / activate / deactivate / list
# ---------------------------------------------------------------------------

def register_tileset(params_json="{}"):
    """Register or update a tileset in the persistent registry.

    Required: name, type (bridge|unit|terrain|items), png (base64), tilecount, columns
    Optional: tilewidth, tileheight, tiles, activate, bridge_variant,
              bridge_simple, bridge_layout, bridge_custom_dir, layout_rows, layout_cols
    """
    params            = _params(params_json)
    name              = str(params["name"])
    ts_type           = str(params["type"])
    png_b64           = str(params["png"])
    tilecount         = int(params["tilecount"])
    columns           = int(params["columns"])
    tilewidth         = int(params.get("tilewidth",  20))
    tileheight        = int(params.get("tileheight", 20))
    tiles             = params.get("tiles")
    activate          = bool(params.get("activate", False))
    bridge_variant    = str(params.get("bridge_variant", "12"))
    bridge_layout     = params.get("bridge_layout")
    bridge_custom_dir = str(params.get("bridge_custom_dir", ""))
    bridge_simple     = params.get("bridge_simple")
    layout_rows       = params.get("layout_rows")
    layout_cols       = params.get("layout_cols")

    entry = _reg.register_tileset(
        name=name, tileset_type=ts_type, png_b64=png_b64,
        tilecount=tilecount, columns=columns,
        tilewidth=tilewidth, tileheight=tileheight,
        tiles=tiles, bridge_variant=bridge_variant,
        bridge_layout=bridge_layout,
        bridge_custom_dir=bridge_custom_dir,
        bridge_simple=bridge_simple,
    )

    if ts_type == "bridge" and activate:
        state.bridge_tilesets[name] = {
            "firstgid":          entry["firstgid"],
            "tilecount":         entry["tilecount"],
            "columns":           entry["columns"],
            "tilewidth":         entry["tilewidth"],
            "tileheight":        entry["tileheight"],
            "name":              name,
            "png":               png_b64,
            "tiles":             entry.get("tiles", {}),
            "bridge_variant":    entry.get("bridge_variant", "12"),
            "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
            "bridge_layout":     entry.get("bridge_layout"),
            "layout_rows":       entry.get("layout_rows", 2),
            "layout_cols":       entry.get("layout_cols", 3),
            "bridge_simple":     entry.get("bridge_simple"),
        }
        state.active_bridge_name = name
    elif ts_type in ("terrain", "unit", "items") and activate:
        state.active_tilesets[ts_type] = {
            "name":      name,
            "firstgid":  entry["firstgid"],
            "tilecount": entry["tilecount"],
            "columns":   entry["columns"],
            "tilewidth": entry["tilewidth"],
            "tileheight":entry["tileheight"],
            "png":       png_b64,
            "tiles":     entry.get("tiles", {}),
            "bridge_variant":    entry.get("bridge_variant", "12"),
            "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
            "bridge_simple":     entry.get("bridge_simple"),
        }

    response = {k: v for k, v in entry.items() if k != "png"}
    return {"registered": response, "snapshot": _snapshot()}


def list_tilesets(params_json="{}"):
    """List registered tilesets, optionally filtered by type."""
    params    = _params(params_json)
    ts_type   = params.get("type")
    all_ts    = _reg.get_all()
    if ts_type:
        all_ts = {k: v for k, v in all_ts.items() if v.get("type") == ts_type}
    # Strip PNG from listing response
    result = {}
    for name, entry in all_ts.items():
        result[name] = {k: v for k, v in entry.items() if k != "png"}
    return {"tilesets": result}


def activate_tileset(params_json="{}"):
    """Load an existing registry tileset into live state."""
    params  = _params(params_json)
    name    = str(params["name"])
    role    = str(params.get("role", params.get("type", "")))
    entry   = _reg.get_tileset(name)
    if entry is None:
        raise ValueError(f"Tileset '{name}' not in registry")

    ts_info = {
        "name":              name,
        "firstgid":          entry["firstgid"],
        "tilecount":         entry["tilecount"],
        "columns":           entry["columns"],
        "tilewidth":         entry.get("tilewidth",  20),
        "tileheight":        entry.get("tileheight", 20),
        "png":               entry.get("png", ""),
        "tiles":             entry.get("tiles", {}),
        "bridge_variant":    entry.get("bridge_variant", "12"),
        "bridge_custom_dir": entry.get("bridge_custom_dir", ""),
        "bridge_layout":     entry.get("bridge_layout"),
        "bridge_simple":     entry.get("bridge_simple"),
    }

    ts_type = entry.get("type", role)
    if ts_type == "bridge":
        state.bridge_tilesets[name] = ts_info
        state.active_bridge_name    = name
    else:
        state.active_tilesets[ts_type] = ts_info
        if ts_type == "terrain":
            state.custom_terrain_mapping = None

    return {"activated": name, "role": ts_type, "snapshot": _snapshot()}


def deactivate_tileset(params_json="{}"):
    """Remove a tileset from live state (does not delete from registry)."""
    params = _params(params_json)
    role   = str(params["role"])

    if role == "bridge":
        name = params.get("name", state.active_bridge_name)
        state.bridge_tilesets.pop(name, None)
        if state.active_bridge_name == name:
            state.active_bridge_name = next(iter(state.bridge_tilesets), "")
    else:
        state.active_tilesets.pop(role, None)
        if role == "terrain":
            state.custom_terrain_mapping = None

    return {"deactivated": role, "snapshot": _snapshot()}


RPC_METHODS = {
    "register_tileset":   register_tileset,
    "list_tilesets":      list_tilesets,
    "activate_tileset":   activate_tileset,
    "deactivate_tileset": deactivate_tileset,
    "list_active_tiles":  list_active_tiles,
    "draw_bridge":        draw_bridge,
    "clear_bridge":       clear_bridge,
    "place_custom_unit":  place_custom_unit,
    "remove_custom_unit": remove_custom_unit,
    "clear_custom_units": clear_custom_units,
    "undo_custom_unit":   undo_custom_unit,
    "place_custom_item":  place_custom_item,
    "remove_custom_item": remove_custom_item,
    "clear_custom_items": clear_custom_items,
    "undo_custom_item":   undo_custom_item,
    "run_coastline":      run_coastline,
    "draw_walls":         draw_walls,
    "clear_walls":        clear_walls,
    "run_height_ocean":   run_height_ocean,
    "place_cc_manual":    place_cc_manual,
    "remove_cc_manual":   remove_cc_manual,
    "place_cc_random":    place_cc_random,
    "undo_cc":            undo_cc,
    "clear_cc":           clear_cc,
    "place_resource_manual":  place_resource_manual,
    "remove_resource_manual": remove_resource_manual,
    "place_resource_random":  place_resource_random,
    "undo_resource":      undo_resource,
    "clear_resource":     clear_resource,
    "get_state_snapshot": get_state_snapshot,
    "run_finalize":       run_finalize,
    "quick_generate":     quick_generate,
    "reset_state":        reset_state,
}


def rpc_call(method, params_json="{}"):
    if method not in RPC_METHODS: raise ValueError(f"Unknown RPC method: {method}")
    return RPC_METHODS[method](params_json)
