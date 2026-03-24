"""
tile_analyzer.py
----------------
Fully vectorized tile sheet analysis using NumPy.
No Python-level per-tile loops anywhere.

Public API:
    analyze_sheet(png_b64, tilecount, columns, tilewidth, tileheight)
        → dict with per-tile metrics + bridge role hints

    extract_tiles_b64(png_b64, tilecount, columns, tilewidth, tileheight)
        → list of base64-encoded PNG strings, one per tile (for UI preview)

    detect_terrain_anchors(png_b64, tilecount, columns, tilewidth, tileheight,
                           water_local_ids, cliff_local_ids)
        → { "water_centroid": [R,G,B], "cliff_centroid": [...],
            "water_like": [...], "cliff_like": [...] }
"""

import base64
import io
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_png(png_b64: str) -> np.ndarray:
    """Decode base64 PNG → RGBA uint8 numpy array."""
    from PIL import Image
    raw  = base64.b64decode(png_b64)
    img  = Image.open(io.BytesIO(raw)).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def _extract_all_tiles(arr: np.ndarray, tilecount: int, columns: int,
                       tw: int, th: int) -> np.ndarray:
    """Reshape sheet array into (tilecount, th, tw, 4) — fully vectorized.

    Uses stride-based reshape: no per-tile loops, no copies beyond the final slice.
    """
    n_rows = (tilecount + columns - 1) // columns
    # Pad sheet if shorter than expected
    needed_h = n_rows * th
    needed_w = columns * tw
    if arr.shape[0] < needed_h or arr.shape[1] < needed_w:
        padded = np.zeros((needed_h, needed_w, 4), dtype=arr.dtype)
        h = min(arr.shape[0], needed_h)
        w = min(arr.shape[1], needed_w)
        padded[:h, :w] = arr[:h, :w]
        arr = padded

    tiles = (
        arr[:n_rows * th, :columns * tw, :]
        .reshape(n_rows, th, columns, tw, 4)
        .transpose(0, 2, 1, 3, 4)          # → (n_rows, columns, th, tw, 4)
        .reshape(n_rows * columns, th, tw, 4)
    )
    return tiles[:tilecount]               # (tilecount, th, tw, 4)


def _symmetry_metrics(tiles: np.ndarray) -> dict:
    """Compute h_sym, v_sym, d_sym, variance for all tiles simultaneously.

    tiles: (N, H, W, 4) uint8
    Returns dict of float32 arrays, each shape (N,).
    """
    N, H, W, _ = tiles.shape
    hh, hw = H // 2, W // 2

    rgb   = tiles[:, :, :, :3].astype(np.float32)   # (N, H, W, 3)
    alpha = tiles[:, :, :,  3].astype(np.float32)   # (N, H, W)
    vis   = alpha > 30                               # (N, H, W) bool mask

    def _sym(a_patch, b_patch, vis_patch):
        """Mean normalised L1 similarity between two patches, masked by visibility."""
        diff   = np.abs(a_patch - b_patch) * vis_patch[..., None]
        total  = vis_patch.sum(axis=(1, 2)).clip(min=1) * 3 * 255
        return 1.0 - diff.sum(axis=(1, 2, 3)) / total

    # Horizontal symmetry: left vs mirrored-right
    h_sym = _sym(
        rgb[:, :, :hw, :],
        rgb[:, :, hw:, :][:, :, ::-1, :],
        vis[:, :, :hw],
    )

    # Vertical symmetry: top vs mirrored-bottom
    v_sym = _sym(
        rgb[:, :hh, :, :],
        rgb[:, hh:, :, :][:, ::-1, :, :],
        vis[:, :hh, :],
    )

    # Diagonal symmetry TL↔BR (low = asymmetric corner)
    d_sym = _sym(
        rgb[:, :hh, :hw, :],
        rgb[:, hh:, hw:, :][:, ::-1, ::-1, :],
        vis[:, :hh, :hw],
    )

    # Per-tile color variance (high = transition/edge tile)
    vis_sum  = vis.sum(axis=(1, 2)).clip(min=1)                         # (N,)
    mean_rgb = (rgb * vis[..., None]).sum(axis=(1, 2)) / vis_sum[:, None]  # (N, 3)
    sq_diff  = ((rgb - mean_rgb[:, None, None, :]) ** 2) * vis[..., None]
    variance = sq_diff.sum(axis=(1, 2, 3)) / (vis_sum * 3)

    # Edge alpha (transparency at border rows/cols)
    top_alpha   = alpha[:, 0,  :].mean(axis=1) / 255.0
    bot_alpha   = alpha[:, -1, :].mean(axis=1) / 255.0
    left_alpha  = alpha[:, :,  0].mean(axis=1) / 255.0
    right_alpha = alpha[:, :, -1].mean(axis=1) / 255.0

    return {
        "mean_rgb":    mean_rgb.astype(np.float32),
        "h_sym":       h_sym.astype(np.float32),
        "v_sym":       v_sym.astype(np.float32),
        "d_sym":       d_sym.astype(np.float32),
        "variance":    variance.astype(np.float32),
        "top_alpha":   top_alpha.astype(np.float32),
        "bot_alpha":   bot_alpha.astype(np.float32),
        "left_alpha":  left_alpha.astype(np.float32),
        "right_alpha": right_alpha.astype(np.float32),
    }


def _bridge_hints(m: dict) -> list:
    """Score each tile for each bridge role. Returns list of hint dicts.

    Scores are in [0, 1] — higher = more likely that role.
    These are HINTS ONLY, not auto-assignments.

    Role heuristics:
        H_top / H_bot  – horizontal span: h_sym high, v_sym low
        W_top / W_bot  – vertical span:   v_sym high, h_sym low
        N / S          – border decor:    top/bot alpha low (transparent edge)
        NW/NE/SW/SE    – corner:          d_sym low (diagonal asymmetry)
        (center tiles are not a bridge role — filtered by low variance)
    """
    h_sym  = m["h_sym"]
    v_sym  = m["v_sym"]
    d_sym  = m["d_sym"]
    t_a    = m["top_alpha"]
    b_a    = m["bot_alpha"]
    l_a    = m["left_alpha"]
    r_a    = m["right_alpha"]
    var    = m["variance"]

    # Normalise variance to [0,1]
    var_max = var.max() if var.max() > 0 else 1.0
    var_n   = var / var_max

    scores = {
        "span_H":  (h_sym - v_sym).clip(0),
        "span_V":  (v_sym - h_sym).clip(0),
        "border_N": ((1 - t_a) * var_n).clip(0),
        "border_S": ((1 - b_a) * var_n).clip(0),
        "corner":  ((1 - d_sym) * var_n).clip(0),
    }

    hints = []
    for i in range(len(h_sym)):
        hints.append({
            "span_H":   round(float(scores["span_H"][i]),  3),
            "span_V":   round(float(scores["span_V"][i]),  3),
            "border_N": round(float(scores["border_N"][i]),3),
            "border_S": round(float(scores["border_S"][i]),3),
            "corner":   round(float(scores["corner"][i]),  3),
        })
    return hints


def _tile_metrics_payload(metrics: dict, hints: list, idx: int) -> dict:
    """Build serializable per-tile metrics payload."""
    return {
        "id": idx,
        "mean_rgb": [round(float(v)) for v in metrics["mean_rgb"][idx]],
        "h_sym": round(float(metrics["h_sym"][idx]), 3),
        "v_sym": round(float(metrics["v_sym"][idx]), 3),
        "d_sym": round(float(metrics["d_sym"][idx]), 3),
        "variance": round(float(metrics["variance"][idx]), 2),
        "top_alpha": round(float(metrics["top_alpha"][idx]), 2),
        "bot_alpha": round(float(metrics["bot_alpha"][idx]), 2),
        "bridge_hints": hints[idx],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sheet(png_b64: str, tilecount: int, columns: int,
                  tilewidth: int = 20, tileheight: int = 20) -> dict:
    """Fully vectorized sheet analysis.

    Returns:
    {
      "tilecount": N,
      "columns": cols,
      "tilewidth": tw, "tileheight": th,
      "tiles": [
        {
          "id": 0,
          "mean_rgb": [R, G, B],
          "h_sym": 0.97, "v_sym": 0.95, "d_sym": 0.91,
          "variance": 123.4,
          "top_alpha": 1.0, "bot_alpha": 1.0,
          "bridge_hints": { "span_H": 0.02, "span_V": 0.01,
                            "border_N": 0.0, "border_S": 0.0, "corner": 0.05 }
        }, ...
      ]
    }
    """
    arr   = _load_png(png_b64)
    tiles = _extract_all_tiles(arr, tilecount, columns, tilewidth, tileheight)
    m     = _symmetry_metrics(tiles)
    hints = _bridge_hints(m)

    result = [_tile_metrics_payload(m, hints, i) for i in range(tilecount)]

    return {
        "tilecount": tilecount,
        "columns":   columns,
        "tilewidth": tilewidth,
        "tileheight": tileheight,
        "tiles": result,
    }


def extract_tiles_b64(png_b64: str, tilecount: int, columns: int,
                       tilewidth: int = 20, tileheight: int = 20) -> list:
    """Return each tile as an individual base64 PNG for UI grid preview.

    Vectorized extraction; only the final encode loop is per-tile
    (PIL save to bytes cannot be vectorized).
    """
    from PIL import Image

    arr   = _load_png(png_b64)
    tiles = _extract_all_tiles(arr, tilecount, columns, tilewidth, tileheight)
    # tiles: (N, H, W, 4) uint8

    result = []
    for i in range(tilecount):
        buf = io.BytesIO()
        Image.fromarray(tiles[i], mode="RGBA").save(buf, format="PNG", optimize=False)
        result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return result


def detect_terrain_anchors(png_b64: str, tilecount: int, columns: int,
                            tilewidth: int = 20, tileheight: int = 20,
                            water_local_ids: Optional[list] = None,
                            cliff_local_ids: Optional[list] = None) -> dict:
    """Use confirmed property anchor IDs to find similar-colored tiles.

    Vectorized distance computation across all tiles.

    Returns:
    {
      "water_centroid": [R, G, B],
      "cliff_centroid": [R, G, B],
      "water_like":     [id, ...],   # tiles within color distance threshold
      "cliff_like":     [id, ...],
      "unclassified":   [id, ...],
    }
    """
    arr   = _load_png(png_b64)
    tiles = _extract_all_tiles(arr, tilecount, columns, tilewidth, tileheight)
    m     = _symmetry_metrics(tiles)
    rgb   = m["mean_rgb"]   # (N, 3) float32

    result = {
        "water_centroid": None,
        "cliff_centroid": None,
        "water_like":     [],
        "cliff_like":     [],
        "unclassified":   list(range(tilecount)),
    }

    classified = set()

    def _classify(anchor_ids: list, threshold: float = 20.0):
        if not anchor_ids:
            return None, []
        ids    = np.array(anchor_ids, dtype=int)
        ids    = ids[(ids >= 0) & (ids < tilecount)]
        if len(ids) == 0:
            return None, []
        centroid = rgb[ids].mean(axis=0)                              # (3,)
        dists    = np.linalg.norm(rgb - centroid, axis=1)            # (N,)
        similar  = np.where(dists < threshold)[0].tolist()
        return centroid.tolist(), similar

    if water_local_ids:
        centroid, similar = _classify(water_local_ids)
        result["water_centroid"] = [round(v) for v in centroid] if centroid else None
        result["water_like"]     = similar
        classified.update(similar)

    if cliff_local_ids:
        centroid, similar = _classify(cliff_local_ids)
        result["cliff_centroid"] = [round(v) for v in centroid] if centroid else None
        result["cliff_like"]     = similar
        classified.update(similar)

    result["unclassified"] = [i for i in range(tilecount) if i not in classified]
    return result
