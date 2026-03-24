import logging
import numpy as np
import random
from perlin_noise import PerlinNoise
from scipy.ndimage import distance_transform_cdt, uniform_filter
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# --- Constants ---
BORDER_MARGIN_RATIO     = 0.08
CENTER_FORBIDDEN_RATIO  = 0.06
CC_MARGIN_RATIO         = 0.07
CC_MIN_DISTANCE_RATIO   = 0.1
DECORATION_FREQUENCY    = 0.05

RESOURCE_PULL_TILES = {
    (-1, -1): 1,  (-1, 0): 2,  (-1, 1): 3,
    ( 0, -1): 11, ( 0, 0): 12, ( 0, 1): 13,
    ( 1, -1): 21, ( 1, 0): 22, ( 1, 1): 23,
}
VALID_MIRROR_MODES = {"none", "horizontal", "vertical", "diagonal1", "diagonal2", "both"}


# ---------------------------------------------------------------------------
# Core terrain generation
# ---------------------------------------------------------------------------

def subdivide(matrix):
    arr = np.asarray(matrix)
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)


def randomize(matrix, smoothness=0.0):
    arr = np.asarray(matrix)
    rows, cols = arr.shape
    padded = np.pad(arr, 1, mode='edge')
    neighbor_count = np.zeros((rows, cols), dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor_count += (padded[1+di:rows+1+di, 1+dj:cols+1+dj] != arr).astype(int)
    base_prob = 0.2 + 0.1 * (neighbor_count - 3)
    prob = np.clip(base_prob * (1.0 - smoothness * 0.92), 0, 1)
    flip_mask = (neighbor_count >= 3) & (np.random.random((rows, cols)) < prob)
    result = arr.copy()
    result[flip_mask] = 1 - result[flip_mask]
    return result


def cleanup_coastline(matrix, min_feature_size=3, smooth_passes=1):
    """Remove tiny isolated land pixels/water holes; optionally round coastline.

    smooth_passes (0-4):
        0 = no extra smoothing  1 = light  2 = medium  3 = heavy  4 = very heavy
    """
    from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening
    arr    = (np.asarray(matrix) > 0).astype(np.uint8)
    struct = np.ones((min_feature_size, min_feature_size), dtype=bool)

    eroded  = binary_erosion(arr, structure=struct)
    arr     = np.where(arr & ~binary_dilation(eroded, structure=struct), 0, arr)

    water   = 1 - arr
    eroded_w = binary_erosion(water, structure=struct)
    arr     = np.where(water & ~binary_dilation(eroded_w, structure=struct), 1, arr)

    if smooth_passes > 0:
        radius = smooth_passes + 1
        y, x   = np.ogrid[-radius:radius+1, -radius:radius+1]
        disk   = (x*x + y*y <= radius*radius)
        for _ in range(smooth_passes):
            arr = binary_opening(
                binary_closing(arr.astype(bool), structure=disk).astype(np.uint8),
                structure=disk
            ).astype(np.uint8)

    return arr.astype(matrix.dtype if hasattr(matrix, 'dtype') else int)


def generate_level(map_matrix, perlin_noise, level_type, level, min_perlin_value,
                   min_distance_to_prev_level=3, min_distance_to_next_level=4):
    rows, cols = map_matrix.shape
    new_map    = map_matrix.copy()

    if level_type == 'height':
        candidate_mask = map_matrix == (level - 1)
        forbidden_mask = map_matrix == (level - 2)
        min_distance   = min_distance_to_prev_level
    else:
        candidate_mask = map_matrix == (level + 1)
        forbidden_mask = map_matrix == (level + 2)
        min_distance   = min_distance_to_next_level

    perlin_mask = perlin_noise >= min_perlin_value

    if forbidden_mask.any():
        distance_ok = distance_transform_cdt(~forbidden_mask, metric='chessboard') > min_distance
    else:
        distance_ok = np.ones((rows, cols), dtype=bool)

    new_map[candidate_mask & perlin_mask & distance_ok] = level
    return new_map


def mirror(matrix, mirroring):
    arr = np.asarray(matrix)
    if mirroring not in VALID_MIRROR_MODES:
        logger.warning("Mirroring option was defined incorrectly")
        return arr

    if mirroring == "none":
        return arr
    elif mirroring == "horizontal":
        mid = arr.shape[0] // 2
        arr[arr.shape[0] - mid:] = arr[:mid][::-1]
    elif mirroring == "vertical":
        mid = arr.shape[1] // 2
        arr[:, arr.shape[1] - mid:] = arr[:, :mid][:, ::-1]
    elif mirroring == "diagonal1":
        n = arr.shape[0]
        for i in range(n):
            arr[i+1:, i] = arr[i, i+1:n]
    elif mirroring == "diagonal2":
        n = arr.shape[0]
        source = arr.copy()
        for i in range(n):
            for j in range(n):
                if j + i >= n:
                    arr[j][i] = source[n - 1 - i][n - 1 - j]
    elif mirroring == "both":
        mid_r = arr.shape[0] // 2
        arr[arr.shape[0] - mid_r:] = arr[:mid_r][::-1]
        mid_c = arr.shape[1] // 2
        arr[:, arr.shape[1] - mid_c:] = arr[:, :mid_c][:, ::-1]
    return arr


def get_all_neighbors(matrix, x, y):
    rows, cols = matrix.shape
    neighbors = [
        (x-1, y-1), (x-1, y), (x-1, y+1),
        (x,   y-1), (x,   y), (x,   y+1),
        (x+1, y-1), (x+1, y), (x+1, y+1),
    ]
    result = []
    for nx, ny in neighbors:
        if 0 <= nx < rows and 0 <= ny < cols:
            result.append(int(matrix[nx, ny] < matrix[x, y]))
        else:
            result.append(0)
    return tuple(result[i:i+3] for i in range(0, 9, 3))


# ---------------------------------------------------------------------------
# Terrain smoothing (vectorised)
# ---------------------------------------------------------------------------

def _remove_isolated_tiles(map_matrix, height_level, passes):
    """Vectorized removal of thin/isolated tiles."""
    for _ in range(passes):
        mask = map_matrix == height_level
        if not np.any(mask):
            break
        pad    = np.pad(map_matrix, 1, mode='edge')
        top    = (pad[:-2, 1:-1] < height_level).astype(np.uint8)
        right  = (pad[1:-1, 2:]  < height_level).astype(np.uint8)
        bottom = (pad[2:,  1:-1] < height_level).astype(np.uint8)
        left   = (pad[1:-1, :-2] < height_level).astype(np.uint8)
        total  = top + right + bottom + left
        bad = mask & (
            (top.astype(bool) & bottom.astype(bool)) |
            (left.astype(bool) & right.astype(bool)) |
            (total >= 3)
        )
        if not np.any(bad):
            break
        map_matrix[bad] = height_level - 1


def _assign_edge_tiles(map_matrix, id_matrix, height_level, tile_set):
    """Vectorized edge tile assignment using cardinal bitmask.

    tile_set layout:
      [0]=flat_below [1]=center  [2]=NW_corner [3]=N_edge [4]=NE_corner
      [5]=W_edge     [6]=E_edge  [7]=SW_corner [8]=S_edge [9]=SE_corner
      [10]=innerTL   [11]=innerTR [12]=innerBL [13]=innerBR
    """
    pad      = np.pad(map_matrix, 1, mode='edge')
    at_level = (map_matrix == height_level)
    n = pad[:-2, 1:-1] < height_level
    e = pad[1:-1, 2:]  < height_level
    s = pad[2:,  1:-1] < height_level
    w = pad[1:-1, :-2] < height_level
    ts = tile_set

    id_matrix[at_level] = ts[1]
    id_matrix[at_level & n & ~e & ~s & ~w] = ts[3]
    id_matrix[at_level & e & ~n & ~s & ~w] = ts[6]
    id_matrix[at_level & s & ~n & ~e & ~w] = ts[8]
    id_matrix[at_level & w & ~n & ~e & ~s] = ts[5]
    id_matrix[at_level & n & e & ~s & ~w]  = ts[4]
    id_matrix[at_level & n & w & ~e & ~s]  = ts[2]
    id_matrix[at_level & s & e & ~n & ~w]  = ts[9]
    id_matrix[at_level & s & w & ~n & ~e]  = ts[7]

    three_plus = (n.astype(np.uint8) + e.astype(np.uint8) +
                  s.astype(np.uint8) + w.astype(np.uint8)) >= 3
    id_matrix[at_level & three_plus] = ts[0]


def _assign_corner_tiles(map_matrix, id_matrix, height_level, tile_set):
    """Vectorized inner corner tile assignment using 8-direction bitmask.

    tile_set[10]=innerTL(SE lower), [11]=innerTR(SW lower),
    tile_set[12]=innerBL(NE lower), [13]=innerBR(NW lower)
    """
    pad      = np.pad(map_matrix, 1, mode='edge')
    at_level = (map_matrix == height_level)
    n  = pad[:-2, 1:-1] < height_level
    e  = pad[1:-1, 2:]  < height_level
    s  = pad[2:,  1:-1] < height_level
    w  = pad[1:-1, :-2] < height_level
    interior = at_level & ~n & ~e & ~s & ~w

    nw = pad[:-2, :-2] < height_level
    ne = pad[:-2, 2:]  < height_level
    sw = pad[2:,  :-2] < height_level
    se = pad[2:,  2:]  < height_level

    ts = tile_set
    id_matrix[interior & se] = ts[10]
    id_matrix[interior & sw] = ts[11]
    id_matrix[interior & ne] = ts[12]
    id_matrix[interior & nw] = ts[13]


def smooth_terrain_tiles(map_matrix, id_matrix, height_level, tile_set, passes=None):
    """Terrain smoothing pipeline (fully vectorized)."""
    p_iso  = max(2, passes or 5)
    p_diag = max(1, (passes or 5) // 2)

    _remove_isolated_tiles(map_matrix, height_level, passes=p_iso)

    for _ in range(p_diag):
        pad      = np.pad(map_matrix, 1, mode='edge')
        at_level = (map_matrix == height_level)
        no_cardinal = (
            at_level
            & ~(pad[:-2, 1:-1] == height_level)
            & ~(pad[1:-1, 2:]  == height_level)
            & ~(pad[2:,  1:-1] == height_level)
            & ~(pad[1:-1, :-2] == height_level)
        )
        if not np.any(no_cardinal):
            break
        map_matrix[no_cardinal] = height_level - 1

    _assign_edge_tiles(map_matrix, id_matrix, height_level, tile_set)
    _assign_corner_tiles(map_matrix, id_matrix, height_level, tile_set)
    return id_matrix


def perlin(x, y, octaves_num, seed=0):
    noise = PerlinNoise(octaves=octaves_num, seed=seed)
    return np.array([[noise([i / x, j / x]) for j in range(y)] for i in range(x)])


def scale_matrix(matrix, target_height, target_width):
    arr     = np.asarray(matrix)
    src_h, src_w = arr.shape
    row_idx = (np.arange(target_height) * src_h // target_height).astype(int)
    col_idx = (np.arange(target_width)  * src_w // target_width).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def place_resource_pull(items_matrix, i, j):
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(items_matrix) and 0 <= nj < len(items_matrix[0]):
                items_matrix[ni][nj] = RESOURCE_PULL_TILES[(di, dj)]


# ---------------------------------------------------------------------------
# Forbidden zone and resource position helpers
# ---------------------------------------------------------------------------

def _get_forbidden_zones(rows, cols, mirroring):
    """Return a boolean numpy mask where resource placement is forbidden."""
    mask   = np.zeros((rows, cols), dtype=bool)
    border = int(min(rows, cols) * BORDER_MARGIN_RATIO)
    half   = int(min(rows, cols) * CENTER_FORBIDDEN_RATIO) // 2

    mask[:border, :] = True
    mask[rows-border:, :] = True
    mask[:, :border] = True
    mask[:, cols-border:] = True

    if mirroring == "horizontal":
        c = rows // 2
        mask[c-half:c+half, :] = True
    elif mirroring == "vertical":
        c = cols // 2
        mask[:, c-half:c+half] = True
    elif mirroring == "diagonal1":
        for i in range(rows):
            mask[i, max(0, i-half):min(cols, i+half)] = True
    elif mirroring == "diagonal2":
        for i in range(rows):
            j = cols - 1 - i
            mask[i, max(0, j-half):min(cols, j+half)] = True
    elif mirroring == "both":
        cr, cc = rows // 2, cols // 2
        mask[cr-half:cr+half, :] = True
        mask[:, cc-half:cc+half] = True

    return mask


def _find_valid_resource_positions(randomized_matrix, forbidden_mask):
    """Find land tiles where the 5x5 neighbourhood is all land and not forbidden.

    Vectorised via scipy uniform_filter (replaces the O(N^2 x 25) Python loop).
    forbidden_mask must be a numpy boolean array.
    """
    land = (randomized_matrix == 1).astype(np.float32)
    avg  = uniform_filter(land, size=5, mode='nearest')
    candidate = (avg > 0.999) & ~forbidden_mask
    # Match original range(2, rows-2) / range(2, cols-2) border guard
    candidate[:2, :]  = False
    candidate[-2:, :] = False
    candidate[:, :2]  = False
    candidate[:, -2:] = False
    ys, xs = np.where(candidate)
    return list(zip(ys.tolist(), xs.tolist()))


def _get_mirrored_positions(i, j, rows, cols, mirroring):
    """Return all positions (including original) that a point maps to after mirroring."""
    positions = [(i, j)]
    if mirroring == "horizontal":
        positions.append((rows - 1 - i, j))
    elif mirroring == "vertical":
        positions.append((i, cols - 1 - j))
    elif mirroring == "diagonal1":
        positions.append((j, i))
    elif mirroring == "diagonal2":
        positions.append((rows - 1 - j, cols - 1 - i))
    elif mirroring == "both":
        positions += [(rows-1-i, j), (i, cols-1-j), (rows-1-i, cols-1-j)]
    seen, result = set(), []
    for p in positions:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _is_valid_pool_position(scaled_i, scaled_j, height_map, placed_positions,
                            min_pool_distance=4, wall_matrix=None,
                            units_matrix=None, cc_clearance=4):
    """Check if a 3x3 pool can be placed at the given height_map position."""
    h_rows, h_cols = height_map.shape
    if scaled_i - 1 < 0 or scaled_i + 1 >= h_rows or scaled_j - 1 < 0 or scaled_j + 1 >= h_cols:
        return False
    if np.any(height_map[scaled_i-1:scaled_i+2, scaled_j-1:scaled_j+2] <= 0):
        return False
    if wall_matrix is not None and np.any(
        wall_matrix[scaled_i-1:scaled_i+2, scaled_j-1:scaled_j+2] == 1
    ):
        return False
    if units_matrix is not None:
        y_min = max(0, scaled_i - cc_clearance); y_max = min(h_rows, scaled_i + cc_clearance + 1)
        x_min = max(0, scaled_j - cc_clearance); x_max = min(h_cols, scaled_j + cc_clearance + 1)
        if np.any(units_matrix[y_min:y_max, x_min:x_max] > 0):
            return False
    for pi, pj in placed_positions:
        if abs(scaled_i - pi) < min_pool_distance and abs(scaled_j - pj) < min_pool_distance:
            return False
    return True


def _find_mirror_axis_positions(randomized_matrix, mirroring):
    """Find valid positions that lie exactly on the mirror axis."""
    rows, cols = randomized_matrix.shape

    if mirroring == "horizontal":
        axis_tiles = [(rows // 2, j) for j in range(2, cols - 2)]
    elif mirroring == "vertical":
        axis_tiles = [(i, cols // 2) for i in range(2, rows - 2)]
    elif mirroring == "diagonal1":
        axis_tiles = [(i, i) for i in range(2, min(rows, cols) - 2)]
    elif mirroring == "diagonal2":
        axis_tiles = [(i, cols - 1 - i) for i in range(2, min(rows, cols) - 2)]
    elif mirroring == "both":
        axis_tiles = [(rows // 2, cols // 2)]
    else:
        return []

    valid = []
    for i, j in axis_tiles:
        if not (2 <= i < rows - 2 and 2 <= j < cols - 2):
            continue
        if randomized_matrix[i][j] == 1 and np.all(randomized_matrix[i-2:i+3, j-2:j+3] == 1):
            valid.append((i, j))
    return valid


def add_resource_pulls(randomized_matrix, num_resource_pulls, mirroring, height_map,
                       items_matrix, wall_matrix=None, units_matrix=None):
    rows, cols     = randomized_matrix.shape
    forbidden_mask = _get_forbidden_zones(rows, cols, mirroring)
    available_tiles = _find_valid_resource_positions(randomized_matrix, forbidden_mask)

    if not available_tiles:
        logger.warning("No valid positions for resource pulls")
        return height_map, items_matrix, []

    scale_factor_x = height_map.shape[1] / cols
    scale_factor_y = height_map.shape[0] / rows
    random.shuffle(available_tiles)

    placed_positions = []
    for ci, cj in available_tiles:
        if len(placed_positions) >= num_resource_pulls:
            break
        remaining = num_resource_pulls - len(placed_positions)
        mirrored  = _get_mirrored_positions(ci, cj, rows, cols, mirroring)
        scaled    = [(int(mi * scale_factor_y), int(mj * scale_factor_x)) for mi, mj in mirrored]

        if len(scaled) > remaining:
            continue
        if not all(
            _is_valid_pool_position(si, sj, height_map, placed_positions,
                                    wall_matrix=wall_matrix, units_matrix=units_matrix)
            for si, sj in scaled
        ):
            continue
        for si, sj in scaled:
            placed_positions.append((si, sj))
            place_resource_pull(items_matrix, si, sj)

    # Second pass: fill remaining slots with on-axis single pools
    if len(placed_positions) < num_resource_pulls and mirroring != "none":
        axis_tiles = _find_mirror_axis_positions(randomized_matrix, mirroring)
        random.shuffle(axis_tiles)
        for ci, cj in axis_tiles:
            if len(placed_positions) >= num_resource_pulls:
                break
            si = int(ci * scale_factor_y)
            sj = int(cj * scale_factor_x)
            if _is_valid_pool_position(si, sj, height_map, placed_positions,
                                       wall_matrix=wall_matrix, units_matrix=units_matrix):
                placed_positions.append((si, sj))
                place_resource_pull(items_matrix, si, sj)

    if len(placed_positions) < num_resource_pulls:
        logger.warning(
            f"Could only place {len(placed_positions)} of {num_resource_pulls} resource pulls"
        )
    return height_map, items_matrix, placed_positions


# ---------------------------------------------------------------------------
# Command centre placement helpers
# ---------------------------------------------------------------------------

def _find_valid_cc_positions(randomized_matrix, mirroring, margin):
    height, width = randomized_matrix.shape

    if mirroring == 'horizontal':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:height // 2 - margin, margin:-margin] = True
        preferred_area = valid_area.copy()
        preferred_area[margin:margin * 2, :] = True
    elif mirroring == 'vertical':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:-margin, margin:width // 2 - margin] = True
        preferred_area = valid_area.copy()
        preferred_area[:, margin:margin * 2] = True
    elif mirroring == 'diagonal1':
        valid_area = np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin)
        valid_area[-margin:, :] = False
        valid_area[:, -margin:] = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, -width // 4:-margin] = True
    elif mirroring == 'diagonal2':
        valid_area = np.fliplr(np.triu(np.ones_like(randomized_matrix, dtype=bool), k=margin))
        valid_area[-margin:, :] = False
        valid_area[:, :margin]  = False
        preferred_area = np.zeros_like(randomized_matrix, dtype=bool)
        preferred_area[margin:height // 4, margin:width // 4] = True
    elif mirroring == 'both':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:height // 2 - margin, margin:width // 2 - margin] = True
        preferred_area = valid_area.copy()
        preferred_area[margin:margin * 2, margin:margin * 2] = True
    elif mirroring == 'none':
        valid_area = np.zeros_like(randomized_matrix, dtype=bool)
        valid_area[margin:-margin, margin:-margin] = True
        preferred_area = valid_area.copy()
    else:
        return [], None

    # Vectorised 5x5 all-land check (replaces is_valid_position closure loop)
    land = (randomized_matrix == 1).astype(np.float32)
    avg  = uniform_filter(land, size=5, mode='nearest')
    candidate = (avg > 0.999) & valid_area & (randomized_matrix == 1)
    valid_positions = list(zip(*np.where(candidate)))
    return valid_positions, preferred_area


def _mirror_command_centers(selected_positions, mirroring, randomized_matrix, height_map_shape):
    height, width   = randomized_matrix.shape
    scale_y = height_map_shape[0] / height
    scale_x = height_map_shape[1] / width
    scaled_units_matrix = np.zeros(height_map_shape, dtype=int)

    for i, (y, x) in enumerate(selected_positions):
        scaled_units_matrix[int(y * scale_y), int(x * scale_x)] = 101 + i

    for y, x in selected_positions:
        team1_val = scaled_units_matrix[int(y * scale_y), int(x * scale_x)]
        if mirroring == 'horizontal':
            mirrors = [(height - 1 - y, x)]
        elif mirroring == 'vertical':
            mirrors = [(y, width - 1 - x)]
        elif mirroring == 'diagonal1':
            mirrors = [(x, y)]
        elif mirroring == 'diagonal2':
            mirrors = [(width - 1 - x, height - 1 - y)]
        elif mirroring == 'both':
            mirrors = [(height-1-y, x), (y, width-1-x), (height-1-y, width-1-x)]
        else:
            continue
        for my, mx in mirrors:
            scaled_units_matrix[int(my * scale_y), int(mx * scale_x)] = team1_val + 5

    return scaled_units_matrix


def add_command_centers(randomized_matrix, num_centers, mirroring, height_map_shape,
                        items_matrix=None, wall_matrix=None):
    if mirroring not in ('horizontal', 'vertical', 'diagonal1', 'diagonal2', 'both', 'none'):
        logger.warning(f"Unsupported mirroring mode for command centers: {mirroring}")
        return np.zeros(height_map_shape, dtype=int)

    if mirroring == 'both':
        num_centers //= 4
    elif mirroring != 'none':
        num_centers //= 2

    height = randomized_matrix.shape[0]
    margin = int(CC_MARGIN_RATIO * height)
    valid_positions, preferred_area = _find_valid_cc_positions(randomized_matrix, mirroring, margin)

    scale_y = height_map_shape[0] / randomized_matrix.shape[0]
    scale_x = height_map_shape[1] / randomized_matrix.shape[1]

    if items_matrix is not None:
        cc_clearance = 2
        def _overlaps_resource(pos):
            sy = int(pos[0] * scale_y); sx = int(pos[1] * scale_x)
            return np.any(items_matrix[
                max(0, sy-cc_clearance):min(items_matrix.shape[0], sy+cc_clearance+1),
                max(0, sx-cc_clearance):min(items_matrix.shape[1], sx+cc_clearance+1)
            ] != 0)
        valid_positions = [p for p in valid_positions if not _overlaps_resource(p)]

    if wall_matrix is not None:
        def _overlaps_wall(pos):
            sy = int(pos[0] * scale_y); sx = int(pos[1] * scale_x)
            return np.any(wall_matrix[
                max(0, sy-1):min(wall_matrix.shape[0], sy+2),
                max(0, sx-1):min(wall_matrix.shape[1], sx+2)
            ] == 1)
        valid_positions = [p for p in valid_positions if not _overlaps_wall(p)]

    preferred_positions = [pos for pos in valid_positions if preferred_area[pos]]

    if len(valid_positions) < num_centers:
        raise ValueError("Not enough valid positions for command centers")

    selected_positions = []
    while len(selected_positions) < num_centers:
        pool = (preferred_positions
                if preferred_positions and np.random.random() < 0.7
                else valid_positions)
        pos  = pool.pop(np.random.randint(len(pool)))
        selected_positions.append(pos)
        min_dist = height * CC_MIN_DISTANCE_RATIO
        valid_positions     = [p for p in valid_positions     if cdist([pos], [p])[0][0] > min_dist]
        preferred_positions = [p for p in preferred_positions if cdist([pos], [p])[0][0] > min_dist]

    return _mirror_command_centers(selected_positions, mirroring, randomized_matrix, height_map_shape)


# ---------------------------------------------------------------------------
# Decoration tiles
# ---------------------------------------------------------------------------

def add_decoration_tiles(id_matrix, map_matrix, dec_tiles, freq):
    """Place decorations on interior flat land cells (vectorised candidate selection).

    A cell qualifies if it is land (> 0) and no 8-neighbour has a lower value
    (i.e. get_all_neighbors returns all-zero, meaning it's at the lowest level
    in its neighbourhood -- same logic as the original loop).
    """
    pad = np.pad(map_matrix, 1, mode='edge')
    # Minimum of the 8 neighbours (not the cell itself)
    neighbor_min = np.minimum.reduce([
        pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:],
        pad[1:-1, :-2],                pad[1:-1, 2:],
        pad[2:,  :-2],  pad[2:, 1:-1], pad[2:,  2:],
    ])
    candidate = (map_matrix > 0) & (neighbor_min >= map_matrix)
    ys, xs    = np.where(candidate)
    rand_vals = np.random.random(len(ys))

    for idx in range(len(ys)):
        if freq > rand_vals[idx]:
            level = int(map_matrix[ys[idx], xs[idx]])
            if level in dec_tiles:
                id_matrix[ys[idx], xs[idx]] = random.choice(dec_tiles[level])
    return id_matrix


# ---------------------------------------------------------------------------
# Legacy full-pipeline entry point (kept for backward compatibility)
# ---------------------------------------------------------------------------

def create_map_matrix(initial_matrix, height, width, mirroring, num_resource_pulls,
                      num_command_centers, num_height_levels, num_ocean_levels,
                      shoreline_smoothness=0.0, preview_callback=None):
    if not isinstance(initial_matrix, list):
        raise ValueError("Initial matrix was defined incorrectly")
    if mirroring not in ("none", "horizontal", "vertical", "diagonal1", "diagonal2", "both"):
        raise ValueError("Mirroring option was defined incorrectly")
    if num_command_centers % 2 != 0 or num_command_centers > 10:
        raise ValueError("Number of command centers must be an even number up to 10")
    if not (1 <= num_height_levels <= 7):
        raise ValueError("Number of height levels must be from 1 to 7")
    if not (1 <= num_ocean_levels <= 3):
        raise ValueError("Number of ocean levels must be from 1 to 3")

    upscales = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    num_upscales = sum(1 for u in upscales if min(height, width) >= u)

    randomized_matrix = np.array(initial_matrix)
    if preview_callback:
        preview_callback("initial_matrix", randomized_matrix.copy(), None, None, None)

    for i in range(num_upscales):
        subdivided_matrix = subdivide(randomized_matrix)
        randomized_matrix = mirror(
            randomize(subdivided_matrix, smoothness=shoreline_smoothness), mirroring
        )
        if preview_callback:
            preview_callback(f"upscale_{i+1}/{num_upscales}", randomized_matrix.copy(), None, None, None)

    height_map = scale_matrix(randomized_matrix, height, width)
    logger.info("Basic matrix created")

    perlin_map = perlin(height, width, octaves_num=9, seed=random.randint(0, 99999))
    if preview_callback:
        preview_callback("scaled_matrix", height_map.copy(), None, None, None)
    logger.info("Perlin matrix generated")

    perlin_change = 1 / num_height_levels
    perlin_value  = -0.5
    for level in range(2, num_height_levels + 1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "height",
                                    level=level, min_perlin_value=perlin_value)
        logger.info(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"height_level_{level}", height_map.copy(), None, None, None)

    perlin_change = 1 / num_ocean_levels
    perlin_value  = -0.5
    for level in range(-1, -num_ocean_levels - 1, -1):
        perlin_value += perlin_change
        height_map = generate_level(height_map, perlin_map, "ocean",
                                    level=level, min_perlin_value=perlin_value)
        logger.info(f"Level {level} generated")
        if preview_callback:
            preview_callback(f"ocean_level_{level}", height_map.copy(), None, None, None)

    items_matrix = np.zeros_like(height_map)
    # FIX: add_resource_pulls returns (height_map, items_matrix, placed_positions)
    height_map, items_matrix, _ = add_resource_pulls(
        randomized_matrix, num_resource_pulls, mirroring, height_map, items_matrix
    )
    logger.info("Resource pulls added")
    if preview_callback:
        preview_callback("resource_pulls", height_map.copy(), None, items_matrix.copy(), None)

    units_matrix = add_command_centers(
        randomized_matrix, num_command_centers, mirroring, height_map.shape, items_matrix
    )
    logger.info("Command centers added")
    if preview_callback:
        preview_callback("command_centers", height_map.copy(), None,
                         items_matrix.copy(), units_matrix.copy())

    return height_map, items_matrix, units_matrix
