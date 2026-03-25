import logging
import os
import sys
import random
import base64
import gzip

import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import (
    distance_transform_cdt,
    gaussian_filter as _gaussian_filter,
    median_filter as _median_filter,
    label as _label_segments,
)

from wizard_state import WizardState, WizardStep
from procedural_map_generator_functions import (
    cleanup_coastline,
    subdivide,
    randomize,
    mirror,
    scale_matrix,
    perlin,
    add_resource_pulls,
    add_command_centers,
    smooth_terrain_tiles,
    add_decoration_tiles,
    place_resource_pull,
    DECORATION_FREQUENCY,
    _get_mirrored_positions,
)

logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """Resolve path to bundled data file (PyInstaller-compatible)."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative_path)

# AutoLight tileset firstgid = 201 in Tiled TMX (1-based GID, tileset starts at index 1).
# All ground tile local IDs are stored as (gid - TILE_ID_OFFSET) internally, then offset
# is re-applied during TMX encoding. This must match the blueprint .tmx firstgid value.
TILE_ID_OFFSET = 201

# large-rock tileset: firstgid=336, so local ID offset from AutoLight = 135
# Storing (135 + rock_local_id) in id_matrix means encoding gives 201 + 135 + local = 336 + local
ROCK_ID_OFFSET = 135

# large-rock tile local IDs mapped to the same layout as terrain tile_sets:
# (flat_below, center, NW, N, NE, W, E, SW, S, SE, iTL, iTR, iBL, iBR)
LARGE_ROCK_TILE_SET = (-1, 4, 0, 1, 2, 3, 5, 6, 7, 8, 13, 12, 10, 9)

# Cardinal neighbor pattern -> tile_set index (same as _CARDINAL_TILE_MAP in proc functions)
# Neighbors: (top, right, bottom, left), 1 = passable (not wall)
_WALL_CARDINAL_MAP = {
    (0, 0, 0, 0): 0,   # all walls -> center
    (1, 0, 0, 0): 2,   # top passable -> N edge
    (0, 1, 0, 0): 5,   # right passable -> E edge
    (1, 1, 0, 0): 3,   # top+right -> NE corner
    (0, 0, 1, 0): 7,   # bottom passable -> S edge
    (0, 1, 1, 0): 8,   # right+bottom -> SE corner
    (0, 0, 0, 1): 4,   # left passable -> W edge
    (1, 0, 0, 1): 1,   # top+left -> NW corner
    (0, 0, 1, 1): 6,   # bottom+left -> SW corner
}

# Patterns that create thin/isolated walls (same as terrain) - demote to passable
_WALL_ISOLATED = [
    (1, 0, 1, 0), (1, 1, 1, 0), (0, 1, 0, 1),
    (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _state_hw(state):
    """Return (height, width) from state, preferring height_map shape."""
    if state.height_map is not None:
        return state.height_map.shape
    return int(state.height), int(state.width)


def _clear_resource_pool(items_matrix, r, c):
    """Clear the 3×3 resource pool centred at (r, c)."""
    h, w = items_matrix.shape
    items_matrix[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = 0


def _stamp_approach_ramp(is_vertical, ec_r_min, ec_r_max, ec_c_left, ec_c_right,
                          H, W, read_hm, write_hm, approach_h, span_h, approach_depth,
                          write_id=None, approach_id=None):
    """Stamp sand-approach ramp cells outward from bridge endcaps.

    Scans outward from each endcap boundary, stopping when:
      - the scan distance exceeds approach_depth, or
      - the base terrain is water (< 0) or already >= span_h.

    read_hm  : source array for height checks (may equal write_hm).
    write_hm : destination array for height updates.
    write_id : optional id_matrix to update alongside height.
    """
    def _cell(r, c):
        v = int(read_hm[r, c])
        if v < 0 or v >= span_h:
            return False
        write_hm[r, c] = approach_h
        if write_id is not None:
            write_id[r, c] = approach_id
        return True

    if is_vertical:
        for c in range(max(0, ec_c_left - 1), min(W, ec_c_right + 2)):
            for dist, r in enumerate(range(ec_r_min - 1, -1, -1), 1):
                if dist > approach_depth or not _cell(r, c):
                    break
            for dist, r in enumerate(range(ec_r_max + 1, H), 1):
                if dist > approach_depth or not _cell(r, c):
                    break
    else:
        for r in range(max(0, ec_r_min - 1), min(H, ec_r_max + 2)):
            for dist, c in enumerate(range(ec_c_left - 1, -1, -1), 1):
                if dist > approach_depth or not _cell(r, c):
                    break
            for dist, c in enumerate(range(ec_c_right + 1, W), 1):
                if dist > approach_depth or not _cell(r, c):
                    break


def smooth_wall_tiles(wall_matrix):
    """Smooth wall_matrix cells into large-rock tile GIDs for a separate Walls layer.

    For each wall cell (value 1), determines the correct rock tile based on cardinal
    neighbors, then checks diagonal neighbors for inner corners.  Gap cells (value 2)
    are treated as wall-like for neighbour detection but do NOT generate tiles, so they
    create passable openings in the wall border.

    Returns a wall_id_matrix where non-zero cells contain the full GID (firstgid + local_id)
    for the large-rock tileset. Zero means no wall tile at that position.

    Fully vectorized with NumPy — no Python-level per-cell loops.
    """
    if wall_matrix is None:
        return None

    h, w = wall_matrix.shape
    tile_set = LARGE_ROCK_TILE_SET
    # GIDs: firstgid(336) = TILE_ID_OFFSET(201) + ROCK_ID_OFFSET(135)
    ROCK_FIRSTGID = TILE_ID_OFFSET + ROCK_ID_OFFSET

    # ── Pass 1: remove isolated wall cells (thin lines) ───────────────────────
    # solid = True where cell is wall(1) or gap(2); passable = True where empty(0)
    cleaned = wall_matrix.copy()
    solid = cleaned > 0

    # Shifted neighbour arrays for pass-1 (OOB = passable = True)
    def _p1(axis, shift):
        p = ~np.roll(solid, shift, axis=axis)
        if shift == 1:
            if axis == 0: p[0, :]   = True
            else:          p[:, 0]   = True
        else:
            if axis == 0: p[-1, :]  = True
            else:          p[:, -1]  = True
        return p

    tp = _p1(0,  1)   # top passable
    rp = _p1(1, -1)   # right passable
    bp = _p1(0, -1)   # bottom passable
    lp = _p1(1,  1)   # left passable

    # _WALL_ISOLATED patterns as vectorized boolean expression
    isolated = (
        ( tp & ~rp &  bp & ~lp) |   # (1,0,1,0)
        ( tp &  rp &  bp & ~lp) |   # (1,1,1,0)
        (~tp &  rp & ~bp &  lp) |   # (0,1,0,1)
        ( tp &  rp & ~bp &  lp) |   # (1,1,0,1)
        ( tp & ~rp &  bp &  lp) |   # (1,0,1,1)
        (~tp &  rp &  bp &  lp) |   # (0,1,1,1)
        ( tp &  rp &  bp &  lp)     # (1,1,1,1)
    )
    cleaned[(cleaned == 1) & isolated] = 0

    # ── Pass 2: assign tile IDs (vectorized) ──────────────────────────────────
    result  = np.zeros((h, w), dtype=np.int32)
    solid2  = cleaned > 0
    wall_only = cleaned == 1   # gap cells excluded from tile generation

    # Shifted neighbour arrays for pass-2 (OOB = solid = not passable = False)
    def _p2(axis, shift):
        p = ~np.roll(solid2, shift, axis=axis)
        if shift == 1:
            if axis == 0: p[0, :]   = False
            else:          p[:, 0]   = False
        else:
            if axis == 0: p[-1, :]  = False
            else:          p[:, -1]  = False
        return p

    tp2 = _p2(0,  1)
    rp2 = _p2(1, -1)
    bp2 = _p2(0, -1)
    lp2 = _p2(1,  1)

    # Build lookup: 4-bit cardinal pattern → tile_set index (-1 = fallback)
    _lut = np.full(16, -1, dtype=np.int8)
    for (t, r, b, l), idx in _WALL_CARDINAL_MAP.items():
        _lut[(t << 3) | (r << 2) | (b << 1) | l] = idx

    pat_int = (
        (tp2.view(np.uint8) << 3) |
        (rp2.view(np.uint8) << 2) |
        (bp2.view(np.uint8) << 1) |
        lp2.view(np.uint8)
    ).astype(np.int8)
    ts_idx = _lut[pat_int]   # (h, w) array of tile_set indices

    # Edge/corner tiles  (ts_idx 1-8) → local_id = tile_set[ts_idx + 1]
    tile_arr  = np.array(tile_set, dtype=np.int32)
    safe_idx  = np.clip(ts_idx.astype(np.int32), 0, 8)
    local_ids = tile_arr[safe_idx + 1]        # +1: tile_set[0]=flat_below, [1]=center
    edge_mask = wall_only & (ts_idx > 0) & (local_ids >= 0)
    result[edge_mask] = ROCK_FIRSTGID + local_ids[edge_mask]

    # Center tiles (ts_idx == 0) → check diagonals for inner corners
    center_mask = wall_only & (ts_idx == 0)

    def _diag(dr, dc):
        p = ~np.roll(np.roll(solid2, dr, axis=0), dc, axis=1)
        if dr ==  1: p[0,  :] = False
        if dr == -1: p[-1, :] = False
        if dc ==  1: p[:,  0] = False
        if dc == -1: p[:, -1] = False
        return p

    tl_p = _diag( 1,  1)   # top-left  (row-1, col-1)
    tr_p = _diag( 1, -1)   # top-right
    bl_p = _diag(-1,  1)   # bottom-left
    br_p = _diag(-1, -1)   # bottom-right

    # Exactly-one-diagonal-passable → inner corner
    result[center_mask &  tl_p & ~tr_p & ~bl_p & ~br_p] = ROCK_FIRSTGID + tile_set[10]
    result[center_mask & ~tl_p &  tr_p & ~bl_p & ~br_p] = ROCK_FIRSTGID + tile_set[11]
    result[center_mask & ~tl_p & ~tr_p &  bl_p & ~br_p] = ROCK_FIRSTGID + tile_set[12]
    result[center_mask & ~tl_p & ~tr_p & ~bl_p &  br_p] = ROCK_FIRSTGID + tile_set[13]
    # Interior cells (0 or 2+ passable diagonals) → leave as 0 (decorations go here)

    # Fallback for unexpected patterns (should be rare after pass-1 cleanup)
    if tile_set[1] >= 0:
        result[wall_only & (ts_idx < 0)] = ROCK_FIRSTGID + tile_set[1]

    return result


# ---------------------------------------------------------------------------
# Bridge system — logic moved to bridge_pipeline.py
# bridge_pipeline used directly via bridge.py RPC layer


def run_coastline(state: WizardState, preview_cb=None):
    """Subdivide-randomize-mirror loop + scale. Writes randomized_matrix, coastline_height_map."""
    initial_matrix = state.initial_matrix
    height, width = state.height, state.width
    mirroring = state.mirroring

    upscales = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    num_upscales = 0
    for i in range(len(upscales)):
        if min(height, width) >= upscales[i]:
            num_upscales = i
    num_upscales += 1

    randomized_matrix = np.array(initial_matrix)
    if preview_cb:
        preview_cb("initial_matrix", randomized_matrix.copy())

    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(randomize(subdivided_matrix, smoothness=getattr(state, "_smoothness", 0.0)), mirroring)
        if preview_cb:
            preview_cb(f"upscale_{i + 1}/{num_upscales}", randomized_matrix.copy())

    # Cleanup + morphological smoothing
    if min(randomized_matrix.shape) >= 10:
        mfs = max(2, min(5, randomized_matrix.shape[0] // 20))
        sp  = max(0, min(4, int(getattr(state, "coast_smooth_passes", 1))))
        randomized_matrix = cleanup_coastline(randomized_matrix, min_feature_size=mfs, smooth_passes=sp)

    # Oversample then center-crop to push the water-only border zone beyond the
    # visible map area — prevents a uniform water "frame" at map boundaries.
    BLEED_RATIO = 0.15
    h_big = int(round(height * (1 + 2 * BLEED_RATIO)))
    w_big = int(round(width  * (1 + 2 * BLEED_RATIO)))
    pad_h = (h_big - height) // 2
    pad_w = (w_big - width)  // 2
    coastline_big        = scale_matrix(randomized_matrix, h_big, w_big)
    coastline_height_map = coastline_big[pad_h:pad_h + height, pad_w:pad_w + width]

    state.randomized_matrix = randomized_matrix
    state.coastline_height_map = coastline_height_map.copy()
    state.wall_matrix = np.zeros((height, width), dtype=int)
    state.completed_step = max(state.completed_step, int(WizardStep.COASTLINE))

    if preview_cb:
        preview_cb("coastline_complete", coastline_height_map.copy())


# ---------------------------------------------------------------------------
# Step 3: Height/Ocean
# ---------------------------------------------------------------------------

# How far (in tiles) the wall influence extends
WALL_INFLUENCE_RADIUS = 12


def _bias_terrain_near_walls(height_map, wall_matrix, num_height_levels):
    """Boost land terrain levels near walls so terrain visually matches hills."""
    wall_mask = wall_matrix == 1
    dist = distance_transform_cdt(~wall_mask, metric='chessboard').astype(float)
    influence = np.clip(1.0 - dist / WALL_INFLUENCE_RADIUS, 0.0, 1.0)
    land_mask = height_map >= 1
    max_target = min(num_height_levels, 5)
    current = height_map[land_mask].astype(float)
    boosted = current + influence[land_mask] * (max_target - current)
    boosted = np.clip(np.round(boosted).astype(int), height_map[land_mask], max_target)
    result = height_map.copy()
    result[land_mask] = boosted
    return result


def run_height_ocean(state: WizardState, seed=None, preview_cb=None):
    """Perlin + height/ocean levels. Writes perlin_seed, perlin_map, height_map."""
    height, width = state.height, state.width

    if seed is None:
        seed = random.randint(0, 99999)
    state.perlin_seed = seed

    height_map = state.coastline_height_map.copy()
    num_height_levels = state.num_height_levels
    num_ocean_levels = state.num_ocean_levels
    region_scale = float(getattr(state, "height_region_scale", 1.0))
    region_scale = max(0.4, min(2.5, region_scale))

    land_mask = height_map >= 1
    sea_mask = ~land_mask

    # Smooth macro-noise + coast-distance fields produce broad, contiguous terrain bands.
    perlin_raw = perlin(height, width, octaves_num=5, seed=seed)
    sigma = max(1.2, min(4.8, (min(height, width) / 85.0) * region_scale))
    perlin_soft = _gaussian_filter(perlin_raw, sigma=sigma, mode='nearest')
    pmin, pmax = float(np.min(perlin_soft)), float(np.max(perlin_soft))
    noise01 = (perlin_soft - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(perlin_soft)
    state.perlin_map = noise01 - 0.5

    land_dist = distance_transform_cdt(land_mask, metric='chessboard').astype(float)
    sea_dist = distance_transform_cdt(sea_mask, metric='chessboard').astype(float)
    if np.any(land_mask):
        land_dist /= max(1.0, float(np.max(land_dist[land_mask])))
    if np.any(sea_mask):
        sea_dist /= max(1.0, float(np.max(sea_dist[sea_mask])))

    land_score = (0.60 * land_dist) + (0.40 * noise01)
    sea_score = (0.65 * sea_dist) + (0.35 * (1.0 - noise01))

    # Quantile binning creates smooth, layered regions (no point-like spikes).
    if np.any(land_mask):
        q = np.linspace(0.0, 1.0, num_height_levels + 1)
        bins = np.quantile(land_score[land_mask], q)
        idx = np.digitize(land_score[land_mask], bins[1:-1], right=False) + 1
        height_map[land_mask] = np.clip(idx, 1, num_height_levels)
        if preview_cb:
            preview_cb("height_quantized", height_map.copy())
    if np.any(sea_mask) and num_ocean_levels > 0:
        q = np.linspace(0.0, 1.0, num_ocean_levels + 1)
        bins = np.quantile(sea_score[sea_mask], q)
        idx = np.digitize(sea_score[sea_mask], bins[1:-1], right=False) + 1
        height_map[sea_mask] = -np.clip(idx, 1, num_ocean_levels)
        if preview_cb:
            preview_cb("ocean_quantized", height_map.copy())

    # Consolidation pass: median/weighted smoothing while preserving sign (land vs sea).
    for _ in range(max(2, int(round(2 + region_scale)))):
        med = _median_filter(height_map, size=3, mode='nearest')
        if np.any(land_mask):
            blended_land = np.round(0.72 * height_map[land_mask] + 0.28 * med[land_mask]).astype(int)
            height_map[land_mask] = np.clip(blended_land, 1, num_height_levels)
        if np.any(sea_mask) and num_ocean_levels > 0:
            sea_abs = np.abs(height_map[sea_mask]).astype(float)
            sea_med_abs = np.abs(med[sea_mask]).astype(float)
            blended_sea = np.round(0.72 * sea_abs + 0.28 * sea_med_abs).astype(int)
            height_map[sea_mask] = -np.clip(blended_sea, 1, num_ocean_levels)

    # Remove tiny fragmented spots for both land and ocean levels.
    min_comp = max(8, int((height * width) * (0.0010 * region_scale)))
    for lvl in range(num_height_levels, 1, -1):
        comp_mask = height_map == lvl
        if not np.any(comp_mask):
            continue
        labels, n = _label_segments(comp_mask)
        for lab in range(1, n + 1):
            comp = labels == lab
            if int(np.sum(comp)) < min_comp:
                height_map[comp] = lvl - 1
    for lvl in range(-num_ocean_levels, -1):
        comp_mask = height_map == lvl
        if not np.any(comp_mask):
            continue
        labels, n = _label_segments(comp_mask)
        for lab in range(1, n + 1):
            comp = labels == lab
            if int(np.sum(comp)) < min_comp:
                height_map[comp] = lvl + 1

    if state.wall_matrix is not None and np.any(state.wall_matrix == 1):
        height_map = _bias_terrain_near_walls(height_map, state.wall_matrix, num_height_levels)
        if preview_cb:
            preview_cb("wall_bias", height_map.copy())

    state.height_map = height_map
    state.completed_step = max(state.completed_step, int(WizardStep.HEIGHT_OCEAN))

    if preview_cb:
        preview_cb("height_ocean_complete", height_map.copy())


# ---------------------------------------------------------------------------
# Step 4: Command Centers
# ---------------------------------------------------------------------------

def run_place_cc_random(state: WizardState):
    """Random CC placement avoiding water + walls. Writes units_matrix, cc_positions, cc_groups."""
    randomized_matrix = state.randomized_matrix
    height_map = state.height_map
    mirroring = state.mirroring
    num_centers = state.num_command_centers

    if height_map is None:
        height_map = np.ones((state.height, state.width), dtype=int)
    if randomized_matrix is None:
        randomized_matrix = np.ones((5, 5), dtype=int)

    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros_like(height_map)

    units_matrix = add_command_centers(
        randomized_matrix, num_centers, mirroring, height_map.shape, items_matrix,
        wall_matrix=state.wall_matrix
    )

    cc_positions = [(int(r), int(c)) for r, c in zip(*np.where(units_matrix > 0))]
    state.units_matrix = units_matrix
    state.cc_positions = cc_positions
    state.cc_groups = [cc_positions[:]]
    state.completed_step = max(state.completed_step, int(WizardStep.COMMAND_CENTERS))


def run_place_cc_manual(state: WizardState, row, col, mirrored=True):
    """Place single CC + mirrors. Returns list of placed (row, col) positions."""
    height_map = state.height_map
    h, w = height_map.shape
    mirroring = state.mirroring if mirrored else "none"
    randomized_matrix = state.randomized_matrix
    rm_h, rm_w = randomized_matrix.shape

    rm_row = int(row * rm_h / h)
    rm_col = int(col * rm_w / w)

    if rm_row < 0 or rm_row >= rm_h or rm_col < 0 or rm_col >= rm_w:
        return [], 0
    if randomized_matrix[rm_row, rm_col] != 1:
        return [], 0
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return [], 0
    if height_map[row, col] <= 0:
        return [], 0
    if state.items_matrix is not None:
        cc_clearance = 2
        y_min = max(0, row - cc_clearance); y_max = min(h, row + cc_clearance + 1)
        x_min = max(0, col - cc_clearance); x_max = min(w, col + cc_clearance + 1)
        if np.any(state.items_matrix[y_min:y_max, x_min:x_max] != 0):
            return [], 0

    mirrored_rm = _get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring)
    scale_y = h / rm_h
    scale_x = w / rm_w
    placed = [
        (min(int(mr * scale_y), h - 1), min(int(mc * scale_x), w - 1))
        for mr, mc in mirrored_rm
    ]

    if state.units_matrix is None:
        state.units_matrix = np.zeros((h, w), dtype=int)

    # Normalize any legacy list-format groups to dict format before processing
    for i, group in enumerate(state.cc_groups):
        if isinstance(group, list):
            state.cc_groups[i] = {"id": 101 + i, "mirrored": True, "positions": group}

    # FIFO: if adding this group would exceed 10 CCs, remove oldest groups first
    max_ccs = 10
    evicted_count = 0
    while len(state.cc_positions) + len(placed) > max_ccs and state.cc_groups:
        oldest = state.cc_groups.pop(0)
        positions = oldest["positions"] if isinstance(oldest, dict) else oldest
        for r2, c2 in positions:
            state.units_matrix[r2, c2] = 0
            if (r2, c2) in state.cc_positions:
                state.cc_positions.remove((r2, c2))
        evicted_count += 1

    # Collect ALL GIDs already assigned
    all_used_gids = set()
    for g in state.cc_groups:
        if isinstance(g, dict):
            for j, _ in enumerate(g["positions"]):
                all_used_gids.add(g["id"] + (5 if j > 0 else 0))

    # Pairs: (teamA_gid, teamB_gid) — display numbers 1-10 in RW order
    GID_PAIRS = [(101,106),(102,107),(103,108),(104,109),(105,110)]

    team_a_id = 101
    if len(placed) <= 1:
        for a, b in GID_PAIRS:
            if a not in all_used_gids: team_a_id = a; break
            if b not in all_used_gids: team_a_id = b; break
    else:
        for a, b in GID_PAIRS:
            if a not in all_used_gids and b not in all_used_gids:
                team_a_id = a; break

    state.cc_positions.extend(placed)
    state.cc_groups.append({"id": team_a_id, "mirrored": mirrored, "positions": placed})
    _rebuild_cc_matrix(state)

    return placed, evicted_count


def run_remove_cc_manual(state: WizardState, row, col):
    if state.units_matrix is None or not state.cc_groups:
        return False

    closest_group_idx = -1
    min_dist = 999

    for i, group in enumerate(state.cc_groups):
        positions = group["positions"] if isinstance(group, dict) else group
        for pr, pc in positions:
            dist = max(abs(pr - row), abs(pc - col))
            if dist <= 3 and dist < min_dist:
                min_dist = dist
                closest_group_idx = i

    if closest_group_idx == -1:
        return False

    removed_group = state.cc_groups.pop(closest_group_idx)
    is_mirrored = removed_group.get("mirrored", True) if isinstance(removed_group, dict) else True
    positions = removed_group["positions"] if isinstance(removed_group, dict) else removed_group

    for r2, c2 in positions:
        state.units_matrix[r2, c2] = 0
        if (r2, c2) in state.cc_positions:
            state.cc_positions.remove((r2, c2))

    if is_mirrored:
        used_by_unmirrored = set()
        for g in state.cc_groups:
            if isinstance(g, dict) and not g.get("mirrored", True):
                used_by_unmirrored.add(g["id"])

        candidate = 101
        for group in state.cc_groups:
            if isinstance(group, dict) and group.get("mirrored", True):
                while candidate in used_by_unmirrored or (candidate + 5) in used_by_unmirrored:
                    candidate += 1
                group["id"] = candidate
                candidate += 1

    _rebuild_cc_matrix(state)
    return True


def _rebuild_cc_matrix(state):
    state.units_matrix[:] = 0
    for i, group in enumerate(state.cc_groups):
        if isinstance(group, list):
            state.cc_groups[i] = {"id": 101 + i, "mirrored": True, "positions": group}
            group = state.cc_groups[i]
        base = group["id"]
        pos  = group["positions"]
        n    = len(pos)
        if n == 1:
            state.units_matrix[pos[0][0], pos[0][1]] = base
        elif n == 2:
            state.units_matrix[pos[0][0], pos[0][1]] = base
            state.units_matrix[pos[1][0], pos[1][1]] = base + 5
        else:
            for j, (pr, pc) in enumerate(pos):
                state.units_matrix[pr, pc] = base if j % 2 == 0 else base + 5


def undo_last_cc(state: WizardState):
    """Remove last CC group, rebuild units_matrix."""
    if not state.cc_groups:
        return
    last_group = state.cc_groups.pop()
    positions = last_group.get("positions", last_group) if isinstance(last_group, dict) else last_group
    for pos in positions:
        r, c = int(pos[0]), int(pos[1])
        if state.units_matrix is not None:
            h, w = state.units_matrix.shape
            if 0 <= r < h and 0 <= c < w:
                state.units_matrix[r, c] = 0
        if (r, c) in state.cc_positions:
            state.cc_positions.remove((r, c))


def clear_all_cc(state: WizardState):
    """Remove all CCs."""
    h, w = _state_hw(state)
    state.units_matrix = np.zeros((h, w), dtype=int)
    state.cc_positions = []
    state.cc_groups = []


# ---------------------------------------------------------------------------
# Step 5: Resources
# ---------------------------------------------------------------------------

def run_place_resources_random(state: WizardState):
    """Random resource placement avoiding water + walls + CCs."""
    randomized_matrix = state.randomized_matrix
    height_map = state.height_map
    mirroring = state.mirroring
    num_resource_pulls = state.num_resource_pulls

    if height_map is None:
        height_map = np.ones((state.height, state.width), dtype=int)
    if randomized_matrix is None:
        randomized_matrix = np.ones((5, 5), dtype=int)

    items_matrix = np.zeros_like(height_map)

    height_map_copy, items_matrix, resource_positions = add_resource_pulls(
        randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix,
        wall_matrix=state.wall_matrix,
        units_matrix=state.units_matrix
    )

    state.items_matrix = items_matrix
    state.resource_positions = resource_positions
    state.resource_groups = [resource_positions[:]]
    state.completed_step = max(state.completed_step, int(WizardStep.RESOURCES))


def run_place_resource_manual(state: WizardState, row, col, mirrored=True):
    """Place single resource + mirrors. Returns list of placed (row, col) positions."""
    height_map = state.height_map
    h, w = height_map.shape
    mirroring = state.mirroring if mirrored else "none"
    randomized_matrix = state.randomized_matrix
    rm_h, rm_w = randomized_matrix.shape

    rm_row = int(row * rm_h / h)
    rm_col = int(col * rm_w / w)

    if rm_row < 0 or rm_row >= rm_h or rm_col < 0 or rm_col >= rm_w:
        return []
    if randomized_matrix[rm_row, rm_col] != 1:
        return []
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return []
    if height_map[row, col] <= 0:
        return []
    if state.units_matrix is not None:
        cc_clearance = 4
        y_min = max(0, row - cc_clearance); y_max = min(h, row + cc_clearance + 1)
        x_min = max(0, col - cc_clearance); x_max = min(w, col + cc_clearance + 1)
        if np.any(state.units_matrix[y_min:y_max, x_min:x_max] > 0):
            return []

    mirrored_rm = _get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring)
    scale_y = h / rm_h
    scale_x = w / rm_w

    placed = []
    for mr, mc in mirrored_rm:
        sr = min(int(mr * scale_y), h - 1)
        sc = min(int(mc * scale_x), w - 1)
        # Skip if too close to an already placed position in this mirror group
        if any(abs(pr - sr) <= 3 and abs(pc - sc) <= 3 for pr, pc in placed):
            continue
        placed.append((sr, sc))

    if state.items_matrix is None:
        state.items_matrix = np.zeros((h, w), dtype=int)

    for sr, sc in placed:
        if state.items_matrix[sr, sc] == 0:
            place_resource_pull(state.items_matrix, sr, sc)

    state.resource_positions.extend(placed)
    state.resource_groups.append(placed)

    return placed


def run_remove_resource_manual(state: WizardState, row, col):
    """Removes a resource pool if clicked nearby."""
    if state.items_matrix is None or not state.resource_groups:
        return False

    closest_group_idx = -1
    min_dist = 999

    for i, group in enumerate(state.resource_groups):
        for pr, pc in group:
            dist = max(abs(pr - row), abs(pc - col))
            if dist <= 3 and dist < min_dist:
                min_dist = dist
                closest_group_idx = i

    if closest_group_idx == -1:
        return False

    removed_group = state.resource_groups.pop(closest_group_idx)
    for r, c in removed_group:
        _clear_resource_pool(state.items_matrix, r, c)
        if (r, c) in state.resource_positions:
            state.resource_positions.remove((r, c))

    return True


def undo_last_resource(state: WizardState):
    """Remove last resource group, rebuild items_matrix."""
    if not state.resource_groups:
        return

    last_group = state.resource_groups.pop()
    for r, c in last_group:
        _clear_resource_pool(state.items_matrix, r, c)
        if (r, c) in state.resource_positions:
            state.resource_positions.remove((r, c))


def clear_all_resources(state: WizardState):
    """Remove all resources."""
    h, w = _state_hw(state)
    state.items_matrix = np.zeros((h, w), dtype=int)
    state.resource_positions = []
    state.resource_groups = []


# ---------------------------------------------------------------------------
# Step 6: Finalize & Export
# ---------------------------------------------------------------------------

def _adaptive_passes(h, w):
    px = h * w
    if px <= 100_000:   return 12
    elif px <= 250_000: return 10
    elif px <= 500_000: return 8
    else:               return 6


def run_finalize(state: WizardState, preview_cb=None):
    """Terrain smoothing + decoration. Writes id_matrix."""
    # ── Default AutoLight tile_sets ──────────────────────────────────────────
    _default_tile_sets = {
        "water_sand":       (31, 34, 6, 7, 8, 33, 35, 60, 61, 62, 87, 88, 114, 115),
        "sand_grass":       (34, 37, 9, 10, 11, 36, 38, 63, 64, 65, 90, 91, 117, 118),
        "grass_soil":       (37, 40, 12, 13, 14, 39, 41, 66, 67, 68, 93, 94, 120, 121),
        "soil_swamp":       (40, 43, 15, 16, 17, 42, 44, 69, 70, 71, 96, 97, 123, 124),
        "swamp_stone":      (43, 46, 18, 19, 20, 45, 47, 72, 73, 74, 99, 100, 126, 127),
        "stone_snow":       (46, 49, 21, 22, 23, 48, 50, 75, 76, 77, 102, 103, 129, 130),
        "snow_ice":         (49, 52, 24, 25, 26, 51, 53, 78, 79, 80, 105, 106, 132, 133),
        "deep_water_water": (28, 31, 3, 4, 5, 30, 32, 57, 58, 59, 84, 85, 111, 112),
        "ocean_deep_water": (83, 28, 0, 1, 2, 27, 29, 54, 55, 56, 81, 82, 108, 109),
    }
    _default_decoration_tiles = {
        1: (86, 89, 116),
        2: (110, 119),
        3: (95, 122),
        4: (98, 125),
        5: (101, 128),
        6: (104, 131),
        7: (107, 134),
    }

    # ── Merge custom mapping from state (if user imported a different tileset) ─
    ctm = getattr(state, "custom_terrain_mapping", None) or {}

    tile_sets = {}
    for key, default in _default_tile_sets.items():
        raw = ctm.get(key)
        tile_sets[key] = tuple(int(v) for v in raw) if raw and len(raw) == 14 else default

    decoration_tiles = {}
    dec_ctm = ctm.get("decoration", {})
    for lvl, default_ids in _default_decoration_tiles.items():
        override = dec_ctm.get(str(lvl)) or dec_ctm.get(lvl)
        decoration_tiles[lvl] = tuple(int(v) for v in override) if override else default_ids

    terrain_levels = [
        (-1, tile_sets["ocean_deep_water"]),
        (0, tile_sets["deep_water_water"]),
        (1, tile_sets["water_sand"]),
        (2, tile_sets["sand_grass"]),
        (3, tile_sets["grass_soil"]),
        (4, tile_sets["soil_swamp"]),
        (5, tile_sets["swamp_stone"]),
        (6, tile_sets["stone_snow"]),
        (7, tile_sets["snow_ice"]),
    ]

    height_map = state.height_map.copy()
    h, w = height_map.shape
    id_matrix = np.full((h, w), 83)

    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros((h, w), dtype=int)
    units_matrix = state.units_matrix if state.units_matrix is not None else np.zeros((h, w), dtype=int)

    smooth_passes = _adaptive_passes(*id_matrix.shape)
    for level, tile_set in reversed(terrain_levels):
        id_matrix = smooth_terrain_tiles(height_map, id_matrix, level, tile_set, passes=smooth_passes)
        if preview_cb:
            preview_cb(f"terrain_smooth_{level}", height_map.copy(), id_matrix.copy(),
                       items_matrix.copy(), units_matrix.copy())

    id_matrix = add_decoration_tiles(id_matrix, height_map, decoration_tiles, DECORATION_FREQUENCY)

    # Merge wall tiles into items layer
    if state.wall_matrix is not None and np.any(state.wall_matrix):
        wall_id_matrix = smooth_wall_tiles(state.wall_matrix)
        if wall_id_matrix is not None:
            wall_mask = wall_id_matrix > 0
            items_matrix[wall_mask] = wall_id_matrix[wall_mask]
            if preview_cb:
                preview_cb("wall_tiles", height_map.copy(), id_matrix.copy(),
                           items_matrix.copy(), units_matrix.copy())

    # ── Bridge tiles → Items layer ───────────────────────────────────────────
    bm = getattr(state, "bridge_matrix", None)
    if bm is not None and np.any(bm > 0):
        TILED_ROT_MASK = 0xA0000000
        H2, W2 = height_map.shape
        APPROACH_DEPTH = 3
        APPROACH_H, APPROACH_ID = 1, 34   # sand  (ramp)
        SPAN_H,     SPAN_ID     = 5, 46   # stone (span — matches minimap gray)

        ts_active = getattr(state, "bridge_tileset", None)
        fg        = int(ts_active.get("firstgid", 351)) if ts_active else 351
        ENDCAP_IDS = {0, 3, 6, 9, 2, 5, 8, 11}
        ROTFLAG64  = np.int64(TILED_ROT_MASK)
        bm_local   = np.where(bm > 0,
                               (bm.astype(np.int64) & ~ROTFLAG64) - fg,
                               np.int64(-1))

        labeled, n_segs = _label_segments((bm > 0).astype(np.int32))
        for seg_id in range(1, n_segs + 1):
            seg_mask = labeled == seg_id
            seg_rows, seg_cols = np.where(seg_mask)
            seg_r_min, seg_r_max    = int(seg_rows.min()), int(seg_rows.max())
            seg_c_left, seg_c_right = int(seg_cols.min()), int(seg_cols.max())
            is_vertical = (seg_r_max - seg_r_min) > (seg_c_right - seg_c_left)

            height_map[seg_mask & (height_map < SPAN_H)] = SPAN_H
            id_matrix [seg_mask & (id_matrix  < SPAN_ID)] = SPAN_ID

            endcap_mask = seg_mask & np.isin(bm_local, list(ENDCAP_IDS))
            ec_rows, ec_cols = np.where(endcap_mask)
            if len(ec_rows) == 0:
                continue

            _stamp_approach_ramp(
                is_vertical,
                int(ec_rows.min()), int(ec_rows.max()),
                int(ec_cols.min()), int(ec_cols.max()),
                H2, W2,
                height_map, height_map,
                APPROACH_H, SPAN_H, APPROACH_DEPTH,
                write_id=id_matrix, approach_id=APPROACH_ID,
            )

        # Visual tiles → items_matrix (overlay)
        bridge_mask = bm > 0
        free_mask   = bridge_mask & (items_matrix == 0)
        items_matrix[free_mask] = bm[free_mask].astype(items_matrix.dtype)

        # Write AutoLight stone tile (local_id=46, level 5 = gray/stone) to ground layer
        # under all bridge cells. This makes the minimap show gray (bridge-like color)
        # without any GID conflict with custom tilesets.
        # local_id 46 = stone center in AutoLight (firstgid=201, so GID=247).
        BRIDGE_GROUND_LOCAL = 46   # AutoLight stone/cliff center — gray color
        id_matrix[bridge_mask] = BRIDGE_GROUND_LOCAL

    state.id_matrix = id_matrix
    state.items_matrix = items_matrix
    state.completed_step = max(state.completed_step, int(WizardStep.FINALIZE))

    if preview_cb:
        preview_cb("terrain_complete", height_map.copy(), id_matrix.copy(),
                   items_matrix.copy(), units_matrix.copy())


def write_tmx(state: WizardState, blueprint_xml: str) -> bytes:
    """Encode matrices and write into blueprint XML."""
    import io as _io
    id_matrix = state.id_matrix
    h, w = id_matrix.shape
    items_matrix = state.items_matrix if state.items_matrix is not None else np.zeros((h,w), dtype=int)
    units_matrix = state.units_matrix if state.units_matrix is not None else np.zeros((h,w), dtype=int)

    def _enc(matrix, offset=0):
        """Vectorized GID encoder: NumPy uint32 → gzip → base64."""
        arr = (np.asarray(matrix).ravel().astype(np.int64) + offset).astype(np.uint32)
        return base64.b64encode(gzip.compress(arr.tobytes())).decode('ascii')

    b64_ground = _enc(id_matrix, TILE_ID_OFFSET)
    b64_items  = _enc(items_matrix, 0)
    b64_units  = _enc(units_matrix, 0)

    tree = ET.ElementTree(ET.fromstring(blueprint_xml))
    root = tree.getroot()
    root.set('width', str(w)); root.set('height', str(h))

    # Inject bridge tileset(s) with water-bridge tile properties
    bts_map = getattr(state, 'bridge_tilesets', {})
    bm      = getattr(state, 'bridge_matrix', None)
    TILED_ROT_MASK = 0xA0000000
    if bts_map and bm is not None and np.any(bm > 0):
        bm_clean = (bm.astype(np.int64) & ~TILED_ROT_MASK)
        for bts_name, bts in bts_map.items():
            fg   = int(bts.get('firstgid', 351))
            cols = int(bts.get('columns', 3))
            tc   = int(bts.get('tilecount', 12))
            tw   = int(bts.get('tilewidth', 20))
            th   = int(bts.get('tileheight', 20))
            tname = bts.get('name', bts_name)
            png_b64 = bts.get('png', '').strip()

            if not np.any((bm_clean >= fg) & (bm_clean < fg + tc)):
                continue

            if not png_b64:
                for r2, c2 in zip(*np.where((bm_clean >= fg) & (bm_clean < fg + tc))):
                    id_matrix[r2, c2] = 34  # sand center — safe fallback
                continue

            ts_el = ET.Element('tileset')
            ts_el.set('firstgid', str(fg)); ts_el.set('name', tname)
            ts_el.set('tilewidth', str(tw)); ts_el.set('tileheight', str(th))
            ts_el.set('columns', str(cols)); ts_el.set('tilecount', str(tc))

            pts = ET.SubElement(ts_el, 'properties')
            emb = ET.SubElement(pts, 'property')
            emb.set('name', 'embedded_png')
            emb.text = '\n' + png_b64 + '\n'

            active_bts_ref = bts_map.get(bts_name, {})
            tiles_map_ref  = active_bts_ref.get('tiles', {})
            span_locals = {
                tiles_map_ref.get('W_top', 3), tiles_map_ref.get('H_top', 4),
                tiles_map_ref.get('E_top', 5), tiles_map_ref.get('W_bot', 6),
                tiles_map_ref.get('H_bot', 7), tiles_map_ref.get('E_bot', 8),
            }
            for local_id in range(tc):
                tile_el = ET.SubElement(ts_el, 'tile')
                tile_el.set('id', str(local_id))
                if local_id in span_locals:
                    props_el = ET.SubElement(tile_el, 'properties')
                    prop_el  = ET.SubElement(props_el, 'property')
                    prop_el.set('name', 'water-bridge')
                    prop_el.set('type', 'string')
                    prop_el.set('value', '')

            first_layer = root.find('layer')
            if first_layer is not None:
                root.insert(list(root).index(first_layer), ts_el)
            else:
                root.append(ts_el)

    lmap = {'Ground': b64_ground, 'Items': b64_items, 'Units': b64_units}
    for layer in root.findall('layer'):
        name = layer.get('name')
        layer.set('width', str(w)); layer.set('height', str(h))
        if name in lmap:
            de = layer.find('data')
            if de is not None: de.text = lmap[name]

    # ── Inject active unit / items tilesets (custom sprite sheets) ───────────
    # These are NOT slot-replacements — they are new <tileset> elements with
    # their own firstgid, inserted before the first <layer>.
    _ROLE_TO_LAYER = {"unit": "Units", "items": "Items"}
    active_ts = getattr(state, "active_tilesets", {})
    first_layer = root.find("layer")
    for role in ("items", "unit"):   # items first so unit is closer to layers
        ts_info = active_ts.get(role)
        if not ts_info or not ts_info.get("png"):
            continue
        ts_name = ts_info["name"]
        # Skip if already injected (e.g. bridge loop above already added it)
        if any(t.get("name") == ts_name for t in root.findall("tileset")):
            continue
        ts_el = ET.Element("tileset")
        ts_el.set("firstgid",   str(ts_info["firstgid"]))
        ts_el.set("name",       ts_name)
        ts_el.set("tilewidth",  str(ts_info.get("tilewidth",  20)))
        ts_el.set("tileheight", str(ts_info.get("tileheight", 20)))
        ts_el.set("columns",    str(ts_info["columns"]))
        ts_el.set("tilecount",  str(ts_info["tilecount"]))
        pts = ET.SubElement(ts_el, "properties")
        emb = ET.SubElement(pts, "property")
        emb.set("name", "embedded_png")
        emb.text = "\n" + ts_info["png"] + "\n"
        if first_layer is not None:
            root.insert(list(root).index(first_layer), ts_el)
        else:
            root.append(ts_el)

    buf = _io.BytesIO()
    tree.write(buf, encoding='UTF-8', xml_declaration=True)
    return buf.getvalue()
