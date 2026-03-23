"""
bridge_simple.py — 3-category bridge placement with flexible endcap geometry.

Categories:
    CAP_START  — left/top endcap (may span multiple rows × cols)
    SPAN       — repeating middle tiles (1 col wide, any rows)
    CAP_END    — right/bottom endcap (mirror of CAP_START by default)

Each BridgeTile has:
    row_offset  — cross-section row   (0 = top row)
    col_offset  — endcap depth        (0 = outermost, 1 = one step in, ...)
    flip        — mirror the tile horizontally (auto-applied to CAP_END if no
                  explicit CAP_END tiles are defined)

Example — 2-row × 3-col endcap:
    CAP_START tiles:
        (row=0, col=0)  (row=0, col=1)  (row=0, col=2)
        (row=1, col=0)  (row=1, col=1)  (row=1, col=2)
    SPAN tiles:
        (row=0, col=0)   ← repeats in middle
        (row=1, col=0)
    CAP_END: auto-mirrored from CAP_START OR explicit tiles

Extending:
    - Add CORNER/JUNCTION categories without touching placement logic
    - col_offset > 0 makes endcap deeper (3-tile-wide cap, etc.)
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np

ROT90_CW  = 0xA0000000
FLIP_H    = 0x80000000   # Tiled horizontal flip flag
FLIP_V    = 0x40000000   # Tiled vertical flip flag


# ── Category ──────────────────────────────────────────────────────────────────

class Cat(str, Enum):
    CAP_START = "cap_start"
    SPAN      = "span"
    CAP_END   = "cap_end"



# ── Tile definition ───────────────────────────────────────────────────────────

@dataclass
class BridgeTile:
    """One tile in a bridge tileset.

    col_offset : endcap depth (0 = outermost tile of endcap,
                 1 = second tile from outside, etc.)
                 For SPAN tiles, always 0.
    flip       : apply Tiled FLIP_H flag (auto-used when mirroring
                 CAP_START → CAP_END)
    """
    local_id:   int
    category:   Cat
    row_offset: int  = 0     # cross-section row (0=top)
    col_offset: int  = 0     # endcap depth (0=outermost)
    rotate:     bool = False
    flip:       bool = False

    def to_dict(self) -> dict:
        return {
            "local_id":   self.local_id,
            "category":   self.category.value,
            "row_offset": self.row_offset,
            "col_offset": self.col_offset,
            "rotate":     self.rotate,
            "flip":       self.flip,
        }

    @staticmethod
    def from_dict(d: dict) -> "BridgeTile":
        return BridgeTile(
            local_id   = int(d["local_id"]),
            category   = Cat(d["category"]),
            row_offset = int(d.get("row_offset", 0)),
            col_offset = int(d.get("col_offset", 0)),
            rotate     = bool(d.get("rotate", False)),
            flip       = bool(d.get("flip", False)),
        )

    def gid(self, firstgid: int) -> int:
        gid = firstgid + self.local_id
        if self.rotate:
            gid = int(gid) | int(ROT90_CW)
        if self.flip:
            gid = int(gid) | int(FLIP_H)
        return gid


# ── Bridge layout ─────────────────────────────────────────────────────────────

@dataclass
class BridgeLayout:
    """Complete bridge layout.

    tiles      : list of BridgeTile
    direction  : "H" | "V"
    n_rows     : cross-section height in tiles
    cap_cols   : endcap width in tiles (default 1)
    mirror_cap : if True and no explicit CAP_END tiles exist,
                 auto-mirror CAP_START tiles as CAP_END with flip=True
    """
    tiles:      list
    direction:  str  = "H"
    n_rows:     int  = 1
    cap_cols:   int  = 1
    span_cols:  int  = 1      # repeating unit width for span (default 1)
    mirror_cap: bool = True

    # ── Lookup ────────────────────────────────────────────────────────────────

    def tiles_for(self, cat: Cat, row_offset: int = 0,
                  col_offset: int = 0) -> list:
        return [t for t in self.tiles
                if t.category == cat
                and t.row_offset == row_offset
                and t.col_offset == col_offset]

    def pick(self, cat: Cat, row_offset: int = 0,
             col_offset: int = 0) -> Optional[BridgeTile]:
        """Return tile for (category, row, col_offset).

        For SPAN with span_cols > 1, col_offset cycles: 0, 1, ..., span_cols-1, 0, 1, ...
        For CAP_END with no explicit tile and mirror_cap=True, mirrors CAP_START.
        """
        # For span, wrap col_offset into the repeat unit
        if cat == Cat.SPAN and self.span_cols > 1:
            col_offset = col_offset % self.span_cols

        found = self.tiles_for(cat, row_offset, col_offset)
        if found:
            return found[0]

        # Auto-mirror: CAP_END → flip of CAP_START
        if cat == Cat.CAP_END and self.mirror_cap:
            src = self.tiles_for(Cat.CAP_START, row_offset, col_offset)
            if src:
                t = src[0]
                return BridgeTile(
                    local_id   = t.local_id,
                    category   = Cat.CAP_END,
                    row_offset = row_offset,
                    col_offset = col_offset,
                    rotate     = t.rotate,
                    flip       = not t.flip,
                )

        # Fallback: span col 0 if specific col not found
        if cat == Cat.SPAN and col_offset > 0:
            return self.pick(Cat.SPAN, row_offset, 0)

        return None

    # ── Validation ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return any(t.category == Cat.SPAN for t in self.tiles)

    def missing(self) -> list:
        m = []
        if not self.is_ready():
            m.append("span (row 0)")
        return m

    def cap_depth(self) -> int:
        """Actual max col_offset used by CAP_START tiles."""
        caps = [t.col_offset for t in self.tiles if t.category == Cat.CAP_START]
        return max(caps, default=0) + 1 if caps else self.cap_cols


    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "direction":  self.direction,
            "n_rows":     self.n_rows,
            "cap_cols":   self.cap_cols,
            "span_cols":  self.span_cols,
            "mirror_cap": self.mirror_cap,
            "tiles":      [t.to_dict() for t in self.tiles],
        }

    @staticmethod
    def from_dict(d: dict) -> "BridgeLayout":
        return BridgeLayout(
            direction  = d.get("direction",  "H"),
            n_rows     = int(d.get("n_rows",    1)),
            cap_cols   = int(d.get("cap_cols",  1)),
            span_cols  = int(d.get("span_cols", 1)),
            mirror_cap = bool(d.get("mirror_cap", True)),
            tiles      = [BridgeTile.from_dict(t) for t in d.get("tiles", [])],
        )


# ── Coast helpers ─────────────────────────────────────────────────────────────

def _coast_col(hm, rows, c0, c1, h, w, left2right=True):
    rng = range(c0, c1+1) if left2right else range(c1, c0-1, -1)
    for c in rng:
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for r in rows):
            return c
    return None

def _coast_row(hm, cols, r0, r1, h, w, top2bot=True):
    rng = range(r0, r1+1) if top2bot else range(r1, r0-1, -1)
    for r in rng:
        if any(0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1 for c in cols):
            return r
    return None


# ── Placement ─────────────────────────────────────────────────────────────────

def place(state, points: list, layout: BridgeLayout,
          erase: bool = False) -> None:
    """
    Place bridge tiles using the 3-category + multi-col-endcap system.

    Column sequence for H bridge (cap_cols=3, total=10):
        col 0,1,2   → CAP_START (col_offset 0,1,2)
        col 3..6    → SPAN
        col 7,8,9   → CAP_END   (col_offset 2,1,0  ← reversed depth)
    """
    if not points:
        return

    bts      = getattr(state, "bridge_tilesets", {})
    active   = getattr(state, "active_bridge_name", "")
    ts       = bts.get(active) or next(iter(bts.values()), None)
    firstgid = int(ts.get("firstgid", 1)) if ts else 1

    h, w = int(state.height), int(state.width)
    if state.bridge_matrix is None:
        state.bridge_matrix = np.zeros((h, w), dtype=int)

    hm = state.height_map if state.height_map is not None          else state.coastline_height_map

    cap_w = layout.cap_depth()   # actual endcap width

    def _set(r: int, c: int, tile: Optional[BridgeTile]):
        if tile is None or not (0 <= r < h and 0 <= c < w):
            return
        state.bridge_matrix[r, c] = tile.gid(firstgid)

    def _water(r: int, c: int) -> bool:
        if hm is None:
            return True
        return 0 <= r < h and 0 <= c < w and int(hm[r, c]) < 1

    # ── Erase ─────────────────────────────────────────────────────────────────
    if erase:
        for r, c in points:
            pad = cap_w + 1
            for dr in range(-layout.n_rows, layout.n_rows + pad):
                for dc in range(-pad, pad + 1):
                    if 0 <= r+dr < h and 0 <= c+dc < w:
                        state.bridge_matrix[r+dr][c+dc] = 0
        return

    rs = [p[0] for p in points]
    cs = [p[1] for p in points]

    # ── Horizontal ────────────────────────────────────────────────────────────
    if layout.direction == "H":
        r_center = sorted(rs)[len(rs) // 2]
        c0, c1   = min(cs), max(cs)
        rows_span = tuple(range(r_center, min(r_center + layout.n_rows, h)))

        if hm is not None:
            wl = _coast_col(hm, rows_span, c0, c1, h, w, True)
            wr = _coast_col(hm, rows_span, c0, c1, h, w, False)
            sl = max(c0, wl - 1) if wl is not None else c0
            sr = min(c1, wr + 1) if wr is not None else c1
            if sl > sr:
                sl, sr = c0, c1
        else:
            sl, sr = c0, c1

        col_seq = list(range(sl, sr + 1))
        n       = len(col_seq)

        span_w = layout.span_cols   # repeat unit width

        for i, col in enumerate(col_seq):
            if i < cap_w:
                cat      = Cat.CAP_START
                col_off  = i
            elif i >= n - cap_w:
                cat      = Cat.CAP_END
                col_off  = (n - 1 - i)
            else:
                cat      = Cat.SPAN
                col_off  = (i - cap_w) % span_w   # cycles 0..span_w-1

            for row_off in range(layout.n_rows):
                tile = layout.pick(cat, row_off, col_off)
                if tile is None:
                    continue
                map_row = r_center + row_off
                # Span tiles: water-only; endcap: allowed over land (1 tile)
                if cat == Cat.SPAN and not _water(map_row, col):
                    continue
                _set(map_row, col, tile)

    # ── Vertical ──────────────────────────────────────────────────────────────
    else:
        c_center = sorted(cs)[len(cs) // 2]
        r0, r1   = min(rs), max(rs)
        cols_span = tuple(range(c_center, min(c_center + layout.n_rows, w)))

        if hm is not None:
            wt = _coast_row(hm, cols_span, r0, r1, h, w, True)
            wb = _coast_row(hm, cols_span, r0, r1, h, w, False)
            st = max(r0, wt - 1) if wt is not None else r0
            sb = min(r1, wb + 1) if wb is not None else r1
            if st > sb:
                st, sb = r0, r1
        else:
            st, sb = r0, r1

        row_seq = list(range(st, sb + 1))
        n       = len(row_seq)

        span_w = layout.span_cols

        for i, row in enumerate(row_seq):
            if i < cap_w:
                cat     = Cat.CAP_START
                col_off = i
            elif i >= n - cap_w:
                cat     = Cat.CAP_END
                col_off = n - 1 - i
            else:
                cat     = Cat.SPAN
                col_off = (i - cap_w) % span_w

            for col_off_r in range(layout.n_rows):
                tile = layout.pick(cat, col_off_r, col_off)
                if tile is None:
                    continue
                map_col = c_center + col_off_r
                if cat == Cat.SPAN and not _water(row, map_col):
                    continue
                _set(row, map_col, tile)


def clear(state) -> None:
    h, w = int(state.height), int(state.width)
    state.bridge_matrix = np.zeros((h, w), dtype=int)
