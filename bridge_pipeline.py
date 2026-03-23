"""
bridge_pipeline.py — Bridge tile placement logic.

Coast-snap rules (no-pierce guarantee):
  - Scan the stroke range for the first/last water cell.
  - Left/top endcap  = water_edge  - 1  → 1 tile into land.
  - Right/bot endcap = water_edge  + 1  → 1 tile into land.
  - Span tiles (H_top/H_bot) drawn only over water cells.
  - Border tiles (N/S) drawn only where span tile is over water.
  - Corner tiles drawn only at endcap column.
"""
import numpy as np
from wizard_state import WizardState

TILED_ROT90_CW = 0xA0000000

_TILES_12 = {
    "NW": 0,  "N":  1,  "NE": 2,
    "W_top": 3, "H_top": 4, "E_top": 5,
    "W_bot": 6, "H_bot": 7, "E_bot": 8,
    "SW": 9,  "S": 10,  "SE": 11,
}
_TILES_6 = {
    "W_top": 0, "H_top": 1, "E_top": 2,
    "W_bot": 3, "H_bot": 4, "E_bot": 5,
}
_FALLBACK = {
    "NW": "W_top", "NE": "E_top", "SW": "W_bot", "SE": "E_bot",
    "N": "H_top",  "S": "H_bot",
    "W": "W_top",  "E": "E_top",  "center": "H_top",
}


def detect_variant(tiles: dict) -> str:
    keys = set(tiles.keys())
    if "center" in keys or ("W" in keys and "E" in keys and "H_top" not in keys):
        return "9"
    if "NW" in keys or "N" in keys:
        return "12"
    return "6"


def get_active_tiles(state: WizardState) -> tuple[dict, int]:
    bts    = getattr(state, "bridge_tilesets", {})
    active = getattr(state, "active_bridge_name", "")
    ts     = bts.get(active) or next(iter(bts.values()), None)
    if not ts:
        ts = getattr(state, "bridge_tileset", None)
    if not ts:
        return _TILES_12, 1
    tiles    = ts.get("tiles") or _TILES_12
    firstgid = int(ts.get("firstgid", 1))
    return tiles, firstgid


# Coast-edge helpers --------------------------------------------------------

def _first_water_col(hm, rows, c0, c1, h, w):
    for c in range(c0, c1 + 1):
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for r in rows):
            return c
    return None

def _last_water_col(hm, rows, c0, c1, h, w):
    for c in range(c1, c0 - 1, -1):
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for r in rows):
            return c
    return None

def _first_water_row(hm, cols, r0, r1, h, w):
    for r in range(r0, r1 + 1):
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for c in cols):
            return r
    return None

def _last_water_row(hm, cols, r0, r1, h, w):
    for r in range(r1, r0 - 1, -1):
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for c in cols):
            return r
    return None


# Main placement ------------------------------------------------------------

def place_stroke(state: WizardState, points: list,
                 erase: bool = False, direction: str = "auto") -> None:
    if not points:
        return

    tiles, firstgid = get_active_tiles(state)
    has_borders = detect_variant(tiles) == "12"

    h, w = int(state.height), int(state.width)
    if state.bridge_matrix is None:
        state.bridge_matrix = np.zeros((h, w), dtype=int)

    hm = state.height_map if state.height_map is not None \
         else state.coastline_height_map

    def _gid(key, rotate=False):
        k = key if key in tiles else _FALLBACK.get(key, "H_top")
        k = k   if k   in tiles else next(iter(tiles), None)
        if not k:
            return 0
        g = firstgid + tiles[k]
        return (g | TILED_ROT90_CW) if rotate else g

    def _set(r, c, gid):
        if 0 <= r < h and 0 <= c < w:
            state.bridge_matrix[r, c] = gid

    def _water(r, c):
        if hm is None:
            return True
        return 0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1

    # Erase -----------------------------------------------------------------
    if erase:
        for r, c in points:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if 0 <= r+dr < h and 0 <= c+dc < w:
                        state.bridge_matrix[r+dr, c+dc] = 0
        return

    rs = [p[0] for p in points]
    cs = [p[1] for p in points]

    # Override direction from tileset metadata if set
    if direction == "auto":
        bts    = getattr(state, "bridge_tilesets", {})
        active = getattr(state, "active_bridge_name", "")
        ts_    = bts.get(active) or next(iter(bts.values()), None)
        if ts_ and ts_.get("bridge_custom_dir"):
            direction = ts_["bridge_custom_dir"]

    # Horizontal ------------------------------------------------------------
    if direction == "H" or (direction == "auto" and max(cs)-min(cs) >= max(rs)-min(rs)):
        r_c = sorted(rs)[len(rs) // 2]
        c0, c1 = min(cs), max(cs)
        r_top, r_bot = r_c, r_c + 1
        if r_bot >= h:
            r_top, r_bot = r_c - 1, r_c

        rows = (r_top, r_bot)
        if hm is not None:
            wl = _first_water_col(hm, rows, c0, c1, h, w)
            wr = _last_water_col (hm, rows, c0, c1, h, w)
            if wl is not None and wr is not None and wl <= wr:
                sl = max(c0, wl - 1)   # endcap = 1 tile into land
                sr = min(c1, wr + 1)
            else:
                sl, sr = c0, c1        # no water found: valley/land bridge
        else:
            sl, sr = c0, c1

        for c in range(sl, sr + 1):
            il = (c == sl); ir = (c == sr); cap = il or ir
            wet = _water(r_top, c) or _water(r_bot, c)
            if not cap and not wet:
                continue

            tk_t = "W_top" if il else ("E_top" if ir else "H_top")
            tk_b = "W_bot" if il else ("E_bot" if ir else "H_bot")
            _set(r_top, c, _gid(tk_t))
            _set(r_bot, c, _gid(tk_b))

            if has_borders:
                rn, rs_ = r_top - 1, r_bot + 1
                if cap:
                    _set(rn, c, _gid("NW" if il else "NE"))
                    _set(rs_, c, _gid("SW" if il else "SE"))
                elif wet:            # border only above water tiles
                    _set(rn, c, _gid("N"))
                    _set(rs_, c, _gid("S"))

    # Vertical --------------------------------------------------------------
    else:
        c_c = sorted(cs)[len(cs) // 2]
        r0, r1 = min(rs), max(rs)
        c_l, c_r = c_c, c_c + 1
        if c_r >= w:
            c_l, c_r = c_c - 1, c_c
        if c_l < 0:
            return

        cols = (c_l, c_r)
        if hm is not None:
            wt = _first_water_row(hm, cols, r0, r1, h, w)
            wb = _last_water_row (hm, cols, r0, r1, h, w)
            if wt is not None and wb is not None and wt <= wb:
                st = max(r0, wt - 1)
                sb = min(r1, wb + 1)
            else:
                st, sb = r0, r1
        else:
            st, sb = r0, r1

        for r in range(st, sb + 1):
            it = (r == st); ib = (r == sb); cap = it or ib
            wet = _water(r, c_l) or _water(r, c_r)
            if not cap and not wet:
                continue

            tk_l = "W_top" if it else ("E_top" if ib else "H_top")
            tk_r = "W_bot" if it else ("E_bot" if ib else "H_bot")
            _set(r, c_l, _gid(tk_l, rotate=True))
            _set(r, c_r, _gid(tk_r, rotate=True))

            if has_borders:
                cl1, cr1 = c_l - 1, c_r + 1
                if cap:
                    _set(r, cl1, _gid("NW" if it else "SW"))
                    _set(r, cr1, _gid("NE" if it else "SE"))
                elif wet:
                    _set(r, cl1, _gid("N", rotate=True))
                    _set(r, cr1, _gid("S", rotate=True))


def clear(state: WizardState) -> None:
    h, w = int(state.height), int(state.width)
    state.bridge_matrix = np.zeros((h, w), dtype=int)


# ---------------------------------------------------------------------------
# Position-based placement (custom layout with duplicate roles)
# ---------------------------------------------------------------------------


