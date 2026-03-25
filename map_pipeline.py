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
    generate_level,  # backward-compat for older height pipelines
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


def _in_bounds(row, col, h, w):
    return 0 <= int(row) < h and 0 <= int(col) < w


def _scale_coord(src_idx, src_size, dst_size):
    """Scale index between matrices using cell-center mapping (less edge bias)."""
    if src_size <= 0 or dst_size <= 0:
        return 0
    ratio = (int(src_idx) + 0.5) * (dst_size / src_size) - 0.5
    return int(np.clip(np.rint(ratio), 0, dst_size - 1))


def _mirrored_canvas_positions(row, col, map_shape, seed_shape, mirroring):
    """Map a click on canvas grid into mirrored map coordinates."""
    h, w = map_shape
    rm_h, rm_w = seed_shape
    if not _in_bounds(row, col, h, w):
        return [], (-1, -1)

    rm_row = _scale_coord(row, h, rm_h)
    rm_col = _scale_coord(col, w, rm_w)
    mirrored_rm = _get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring)

    placed = []
    seen = set()
    for mr, mc in mirrored_rm:
        sr = _scale_coord(mr, rm_h, h)
        sc = _scale_coord(mc, rm_w, w)
        key = (sr, sc)
        if key in seen:
            continue
        seen.add(key)
        placed.append(key)

    return placed, (rm_row, rm_col)


def _local_region_penalty(state, r, c):
    """Return a local penalty factor (>=1) from slope + region scale."""
    slope_map = getattr(state, "region_slope_map", None)
    base = float(np.clip(getattr(state, "height_region_scale", 1.0), 0.1, 3.0))
    if slope_map is None:
        return 1.0 + (base - 1.0) * 0.2
    h, w = slope_map.shape
    if not _in_bounds(r, c, h, w):
        return 1.0
    sv = float(np.clip(slope_map[int(r), int(c)], 0.0, 2.0))
    return 1.0 + (base - 1.0) * 0.35 + sv * 0.55


def _region_penalty(state, r, c):
    """Backward-safe wrapper in case helper symbol is missing in hot-reload envs."""
    fn = globals().get("_local_region_penalty")
    if callable(fn):
        try:
            return float(fn(state, r, c))
        except Exception:
            pass
    base = float(np.clip(getattr(state, "height_region_scale", 1.0), 0.1, 3.0))
    return 1.0 + (base - 1.0) * 0.2


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
    # region_scale: slider 1-10 → 0.1-3.0 (sent by JS, bridge passes through).
    # Behaviour:
    #   Small (0.1) → tight bands, levels change quickly from coast inward (steep)
    #   Large (3.0) → wide bands, levels spread slowly over large distances (gradual)
    region_scale = float(getattr(state, "height_region_scale", 1.0))
    region_scale = max(0.1, min(3.0, region_scale))

    land_mask = height_map >= 1
    sea_mask = ~land_mask

    # ── Perlin noise (texture) ────────────────────────────────────────────────
    # Sigma scales with region so large regions also have large-scale noise blobs.
    base_sigma = min(height, width) / 80.0
    perlin_sigma = max(0.5, base_sigma * region_scale * 2.5)
    perlin_raw = perlin(height, width, octaves_num=5, seed=seed)
    perlin_soft = _gaussian_filter(perlin_raw, sigma=perlin_sigma, mode='nearest')
    pmin, pmax = float(np.min(perlin_soft)), float(np.max(perlin_soft))
    noise01 = (perlin_soft - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(perlin_soft)
    state.perlin_map = noise01 - 0.5

    # ── Distance fields ───────────────────────────────────────────────────────
    # Key idea: clip land_dist to `band_tiles` before normalizing.
    # band_tiles = how many tiles wide each level band is.
    # Small region → few tiles per band → steep, cramped transitions.
    # Large region → many tiles per band → gradual, wide transitions.
    land_dist_raw = distance_transform_cdt(land_mask, metric='chessboard').astype(float)
    sea_dist_raw  = distance_transform_cdt(sea_mask,  metric='chessboard').astype(float)

    # band_tiles: at region=1.0 ≈ 8 tiles per level; scales linearly.
    band_tiles = max(2.0, 8.0 * region_scale)

    if np.any(land_mask):
        # Clip then normalize so the score 0→1 spans exactly band_tiles*num_levels tiles.
        clip_dist = band_tiles * num_height_levels
        land_dist = np.clip(land_dist_raw, 0, clip_dist) / clip_dist
    else:
        land_dist = land_dist_raw

    if np.any(sea_mask):
        clip_dist = band_tiles * num_ocean_levels
        sea_dist = np.clip(sea_dist_raw, 0, max(clip_dist, 1.0)) / max(clip_dist, 1.0)
    else:
        sea_dist = sea_dist_raw

    # Region parameter must feel impactful: increase distance dominance at higher region,
    # and increase noise dominance at lower region.
    region_alpha = float(np.clip((region_scale - 0.1) / (3.0 - 0.1), 0.0, 1.0))
    land_dist_w = 0.45 + 0.45 * region_alpha
    sea_dist_w  = 0.50 + 0.40 * region_alpha
    land_noise_w = 1.0 - land_dist_w
    sea_noise_w  = 1.0 - sea_dist_w

    # Noise adds organic variation; distance drives broad level structure.
    land_score = (land_dist_w * land_dist) + (land_noise_w * noise01)
    sea_score  = (sea_dist_w * sea_dist)  + (sea_noise_w * (1.0 - noise01))

    # Smooth land score so band edges are not pixelated.
    scalar_sigma = max(0.5, perlin_sigma * 0.4)
    land_score = _gaussian_filter(land_score, sigma=scalar_sigma, mode='nearest')

    # Sea score: coastal zone only gets light smoothing; interior keeps raw perlin variation.
    if np.any(sea_mask):
        coast_radius = max(3, int(round(scalar_sigma * 1.5)))
        coast_dist = distance_transform_cdt(sea_mask, metric='chessboard').astype(float)
        coastal_zone = sea_mask & (coast_dist <= coast_radius)
        if np.any(coastal_zone):
            coast_sigma = max(0.6, scalar_sigma * 0.5)
            sea_score_smooth = _gaussian_filter(sea_score, sigma=coast_sigma, mode='nearest')
            blend = np.clip(1.0 - coast_dist / (coast_radius + 1), 0.0, 1.0)
            sea_score = np.where(coastal_zone,
                                 blend * sea_score_smooth + (1.0 - blend) * sea_score,
                                 sea_score)

    def _weighted_thresholds(vals, levels, tail_weight=0.45):
        if levels <= 1:
            return np.array([])
        # Keep high levels present (not collapsing to tiny points) with controlled decay.
        w = np.linspace(1.0, max(0.30, tail_weight), levels)
        p = w / np.sum(w)
        c = np.cumsum(p)[:-1]
        return np.quantile(vals, c)

    # Weighted quantization keeps level spread more even and avoids single-point apexes.
    if np.any(land_mask):
        thr = _weighted_thresholds(land_score[land_mask], num_height_levels, tail_weight=0.52)
        idx = np.digitize(land_score[land_mask], thr, right=False) + 1
        height_map[land_mask] = np.clip(idx, 1, num_height_levels)
        if preview_cb:
            preview_cb("height_quantized", height_map.copy())
    if np.any(sea_mask) and num_ocean_levels > 0:
        # Sea level numbering (matches terrain_levels tile mapping):
        #   0  = shallowest water  (deep_water_water tileset)
        #  -1  = deep water        (ocean_deep_water tileset)
        # idx=1 (shallowest) → 0,  idx=2 → -1,  idx=3 → -2 …
        thr = _weighted_thresholds(sea_score[sea_mask], num_ocean_levels, tail_weight=0.58)
        idx = np.digitize(sea_score[sea_mask], thr, right=False) + 1
        height_map[sea_mask] = -np.clip(idx - 1, 0, num_ocean_levels - 1)
        if preview_cb:
            preview_cb("ocean_quantized", height_map.copy())

    # Consolidation: median filter on land only, kernel size scales with region_scale.
    # Small region → kernel 3 (keeps detail).  Large region → kernel up to 9 (merges patches).
    # Water tiles are never touched here.
    if np.any(land_mask):
        kernel = int(round(3 + region_scale * 3.0))   # 3 … 9
        kernel = kernel if kernel % 2 == 1 else kernel + 1  # must be odd
        land_only = np.where(land_mask, height_map, 0)
        med = _median_filter(land_only, size=kernel, mode='nearest')
        blend_w = min(0.55, 0.15 + region_scale * 0.20)   # 0.15 … 0.55
        height_map[land_mask] = np.clip(
            np.round((1 - blend_w) * height_map[land_mask] + blend_w * med[land_mask]).astype(int),
            1, num_height_levels
        )

    # Remove tiny land fragments; threshold also scales with region_scale.
    min_comp = max(5, int((height * width) * (0.00015 + 0.00025 * region_scale)))
    for lvl in range(num_height_levels, 1, -1):
        comp_mask = height_map == lvl
        if not np.any(comp_mask):
            continue
        labels, n = _label_segments(comp_mask)
        for lab in range(1, n + 1):
            comp = labels == lab
            if int(np.sum(comp)) < min_comp:
                height_map[comp] = lvl - 1

    # ── Natural coastline enforcement (gradient, multi-tile) ────────────────────
    # Both sides of the coastline are enforced with a smooth gradient so levels
    # rise/fall gradually away from the waterline — no sudden jumps.
    #
    # Land side gradient (distance from water edge):
    #   dist 1 → capped at 1,  dist 2 → capped at 2, … up to num_height_levels
    # Sea side gradient (distance from land edge):
    #   dist 1 → capped at 0,  dist 2 → capped at -1, … down to -num_ocean_levels+1
    #
    # The gradient depth scales with the number of levels so a map with more
    # height/ocean levels still gets a proportionally wide transition zone.

    # Distance of every land tile from the nearest water tile, and vice-versa.
    land_dist_from_sea  = distance_transform_cdt(land_mask,  metric='chessboard').astype(int)
    sea_dist_from_land  = distance_transform_cdt(sea_mask,   metric='chessboard').astype(int)

    # Land gradient: tile at dist d from water must be <= d (capped at num_height_levels)
    if np.any(land_mask) and num_height_levels > 1:
        grad_zone = land_mask & (land_dist_from_sea <= num_height_levels)
        max_allowed = np.clip(land_dist_from_sea[grad_zone], 1, num_height_levels)
        height_map[grad_zone] = np.minimum(height_map[grad_zone], max_allowed)

    # Sea gradient: tile at dist d from land must be >= -(d-1)  i.e. 0, -1, -2 …
    if np.any(sea_mask) and num_ocean_levels > 1:
        grad_zone = sea_mask & (sea_dist_from_land <= num_ocean_levels)
        min_allowed = -np.clip(sea_dist_from_land[grad_zone] - 1, 0, num_ocean_levels - 1)
        height_map[grad_zone] = np.maximum(height_map[grad_zone], min_allowed)

    if state.wall_matrix is not None and np.any(state.wall_matrix == 1):
        height_map = _bias_terrain_near_walls(height_map, state.wall_matrix, num_height_levels)
        if preview_cb:
            preview_cb("wall_bias", height_map.copy())

    # Region-derived slope map + metrics (used by placement + UI diagnostics).
    gy, gx = np.gradient(height_map.astype(float))
    slope_map = np.clip(np.hypot(gx, gy), 0.0, 2.0)
    state.region_slope_map = slope_map.astype(np.float32)
    land_vals = slope_map[land_mask] if np.any(land_mask) else slope_map.ravel()
    state.region_metrics = {
        "region_scale": float(region_scale),
        "mean_slope": float(np.mean(land_vals) if land_vals.size else 0.0),
        "p95_slope": float(np.quantile(land_vals, 0.95) if land_vals.size else 0.0),
        "land_dist_weight": float(land_dist_w),
        "band_tiles": float(band_tiles),
    }

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

    row, col = int(row), int(col)
    if not _in_bounds(row, col, h, w):
        return [], 0
    placed, (rm_row, rm_col) = _mirrored_canvas_positions(
        row, col, (h, w), (rm_h, rm_w), mirroring
    )
    if randomized_matrix[rm_row, rm_col] != 1:
        return [], 0
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return [], 0
    if height_map[row, col] <= 0:
        return [], 0
    if state.units_matrix is not None and state.units_matrix[row, col] > 0:
        return [], 0
    if state.items_matrix is not None:
        cc_clearance = int(np.clip(np.ceil(2 * _region_penalty(state, row, col)), 2, 5))
        y_min = max(0, row - cc_clearance); y_max = min(h, row + cc_clearance + 1)
        x_min = max(0, col - cc_clearance); x_max = min(w, col + cc_clearance + 1)
        if np.any(state.items_matrix[y_min:y_max, x_min:x_max] != 0):
            return [], 0
    for pr, pc in placed:
        if height_map[pr, pc] <= 0:
            return [], 0
        if state.wall_matrix is not None and state.wall_matrix[pr, pc] == 1:
            return [], 0
        if state.units_matrix is not None and state.units_matrix[pr, pc] > 0:
            return [], 0

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

    row, col = int(row), int(col)
    if not _in_bounds(row, col, h, w):
        return []
    mirrored_positions, (rm_row, rm_col) = _mirrored_canvas_positions(
        row, col, (h, w), (rm_h, rm_w), mirroring
    )
    if randomized_matrix[rm_row, rm_col] != 1:
        return []
    if state.wall_matrix is not None and state.wall_matrix[row, col] == 1:
        return []
    if height_map[row, col] <= 0:
        return []
    if state.units_matrix is not None:
        cc_clearance = int(np.clip(np.ceil(4 * _region_penalty(state, row, col)), 4, 8))
        y_min = max(0, row - cc_clearance); y_max = min(h, row + cc_clearance + 1)
        x_min = max(0, col - cc_clearance); x_max = min(w, col + cc_clearance + 1)
        if np.any(state.units_matrix[y_min:y_max, x_min:x_max] > 0):
            return []

    placed = []
    for sr, sc in mirrored_positions:
        if height_map[sr, sc] <= 0:
            continue
        if state.wall_matrix is not None and state.wall_matrix[sr, sc] == 1:
            continue
        # Skip if too close to an already placed position in this mirror group
        if any(abs(pr - sr) <= 3 and abs(pc - sc) <= 3 for pr, pc in placed):
            continue
        y_min = max(0, sr - 1); y_max = min(h, sr + 2)
        x_min = max(0, sc - 1); x_max = min(w, sc + 2)
        if state.items_matrix is not None and np.any(state.items_matrix[y_min:y_max, x_min:x_max] != 0):
            continue
        if state.units_matrix is not None:
            cclr = int(np.clip(np.ceil(4 * _region_penalty(state, sr, sc)), 4, 8))
            y_min2 = max(0, sr - cclr); y_max2 = min(h, sr + cclr + 1)
            x_min2 = max(0, sc - cclr); x_max2 = min(w, sc + cclr + 1)
            if np.any(state.units_matrix[y_min2:y_max2, x_min2:x_max2] > 0):
                continue
        placed.append((sr, sc))

    if state.items_matrix is None:
        state.items_matrix = np.zeros((h, w), dtype=int)

    if not placed:
        return []

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
