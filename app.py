"""
Rusted Warfare Map Generator — Flask Server (Full Version)
Jalankan: python app.py
Buka di browser: http://localhost:5000
"""

import os, json, base64, tempfile
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from registry_routes import registry_bp

import bridge

app = Flask(__name__)
app.register_blueprint(registry_bp)

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def _json_body() -> dict:
    """Return request JSON payload as dict, defaulting to empty object."""
    return request.get_json(force=True, silent=True) or {}

# ── Undo / Redo stacks (in-process, per session) ──────────────
_brush_undo:   list = []
_brush_redo:   list = []
_polygon_undo: list = []
_polygon_redo: list = []
_bridge_undo:  list = []
_bridge_redo:  list = []

# ── Polygon list (mirrors client state) ──────────────────────
_polygons: list = []

# ── Custom tilesets for Ground/Wall/Items/Units PNG embed ─────
_custom_tilesets = {}   # slot -> {name, firstGid, columns, tileCount, tileWidth, tileHeight, png_b64}

# ── Blueprint cache — parsed once at startup, served from memory ──────────────
# Avoids re-reading and re-parsing large .tmx files on every /blueprints or
# /tilesets request. Files are static and never change at runtime.
_blueprint_bytes: dict = {}   # pattern (int) -> raw bytes
_tileset_cache:   dict = {}   # pattern (int) -> {ground/wall/items/units -> tileset dict}

def _load_blueprints():
    _TILESET_KEYS = {
        "AutoLight":        "ground",
        "large-rock":       "wall",
        "export_items":     "items",
        "50pCommandCenter": "units",
    }
    for i in range(1, 6):
        path = Path(f"generator_blueprint{i}.tmx")
        if not path.exists():
            continue
        raw = path.read_bytes()
        _blueprint_bytes[i] = raw
        root = ET.fromstring(raw)
        result = {}
        for ts_elem in root.findall("tileset"):
            name = ts_elem.get("name", "")
            key  = _TILESET_KEYS.get(name)
            if not key:
                continue
            firstgid  = int(ts_elem.get("firstgid", 1))
            columns   = int(ts_elem.get("columns", 1))
            tilecount = int(ts_elem.get("tilecount", 0))
            tw        = int(ts_elem.get("tilewidth", 20))
            th        = int(ts_elem.get("tileheight", 20))
            for prop in ts_elem.findall(".//property"):
                if prop.get("name") == "embedded_png":
                    raw_p   = prop.get("value") or prop.text or ""
                    png_b64 = "".join(raw_p.split())
                    result[key] = {
                        "name": name, "firstGid": firstgid,
                        "columns": columns, "tileCount": tilecount,
                        "tileWidth": tw, "tileHeight": th, "png": png_b64,
                    }
                    break
        _tileset_cache[i] = result

_load_blueprints()


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blueprints/<int:pattern>")
def blueprint(pattern):
    raw = _blueprint_bytes.get(pattern)
    if raw is None:
        return jsonify({"error": f"Blueprint {pattern} not found"}), 404
    import io as _io
    return send_file(_io.BytesIO(raw), mimetype="application/xml",
                     download_name=f"generator_blueprint{pattern}.tmx")


@app.route("/tilesets/<int:pattern>")
def get_tilesets(pattern):
    """Return embedded tileset PNGs for a blueprint (served from memory cache)."""
    result = _tileset_cache.get(pattern)
    if result is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(result)


# ─────────────────────────────────────────────────────────────
#  RPC dispatcher
# ─────────────────────────────────────────────────────────────

@app.route("/rpc/<method>", methods=["POST"])
def rpc(method):
    global _brush_undo, _brush_redo, _polygon_undo, _polygon_redo, _polygons
    global _bridge_undo, _bridge_redo
    try:
        params = _json_body()

        # ── Wall brush ────────────────────────────────────────
        if method == "draw_brush_walls":
            _push_undo(_brush_undo, _brush_redo, _cp(bridge.state.wall_matrix))
            result = bridge.rpc_call("draw_walls", params)

        elif method == "undo_brush":
            if _brush_undo:
                _push_undo(_brush_redo, None, _cp(bridge.state.wall_matrix))
                bridge.state.wall_matrix = _brush_undo.pop()
            result = _snap()

        elif method == "redo_brush":
            if _brush_redo:
                _push_undo(_brush_undo, None, _cp(bridge.state.wall_matrix))
                bridge.state.wall_matrix = _brush_redo.pop()
            result = _snap()

        elif method == "clear_brush_walls":
            _push_undo(_brush_undo, _brush_redo, _cp(bridge.state.wall_matrix))
            result = bridge.rpc_call("clear_walls", params)

        elif method == "set_hill_drawing_mode":
            result = _snap()

        # ── Polygon walls ─────────────────────────────────────
        elif method == "update_polygons":
            state = bridge.state
            h, w = int(state.height), int(state.width)
            _push_undo(_polygon_undo, _polygon_redo,
                       _cp(state.wall_matrix) if state.wall_matrix is not None
                       else np.zeros((h, w), dtype=int))
            _polygons = params.get("polygons", [])
            state.wall_matrix = _rasterize_all_polygons(_polygons, h, w)
            result = _snap()

        elif method == "clear_all_polygons":
            state = bridge.state
            h, w = int(state.height), int(state.width)
            _push_undo(_polygon_undo, _polygon_redo,
                       _cp(state.wall_matrix) if state.wall_matrix is not None
                       else np.zeros((h, w), dtype=int))
            _polygons = []
            state.wall_matrix = np.zeros((h, w), dtype=int)
            result = _snap()

        elif method == "undo_polygons":
            if _polygon_undo:
                _push_undo(_polygon_redo, None, _cp(bridge.state.wall_matrix))
                bridge.state.wall_matrix = _polygon_undo.pop()
            result = _snap()

        elif method == "redo_polygons":
            if _polygon_redo:
                _push_undo(_polygon_undo, None, _cp(bridge.state.wall_matrix))
                bridge.state.wall_matrix = _polygon_redo.pop()
            result = _snap()

        elif method == "toggle_edge_gap":
            poly_id  = params.get("polygon_id")
            edge_idx = params.get("edge_index", 0)
            for p in _polygons:
                if p.get("id") == poly_id:
                    gaps = p.get("edgeGaps", [])
                    if edge_idx < len(gaps):
                        gaps[edge_idx] = not gaps[edge_idx]
                    break
            state = bridge.state
            h, w = int(state.height), int(state.width)
            state.wall_matrix = _rasterize_all_polygons(_polygons, h, w)
            result = _snap()

        # ── Bridge drawing ────────────────────────────────────
        elif method == "draw_bridge":
            _push_undo(_bridge_undo, _bridge_redo,
                       _cp(getattr(bridge.state, "bridge_matrix", None)))
            result = bridge.rpc_call("draw_bridge", params)

        elif method == "erase_bridge":
            _push_undo(_bridge_undo, _bridge_redo,
                       _cp(getattr(bridge.state, "bridge_matrix", None)))
            params["erase"] = True
            result = bridge.rpc_call("draw_bridge", params)

        elif method == "undo_bridge":
            if _bridge_undo:
                _push_undo(_bridge_redo, None,
                           _cp(getattr(bridge.state, "bridge_matrix", None)))
                bridge.state.bridge_matrix = _bridge_undo.pop()
            result = _snap()

        elif method == "redo_bridge":
            if _bridge_redo:
                _push_undo(_bridge_undo, None,
                           _cp(getattr(bridge.state, "bridge_matrix", None)))
                bridge.state.bridge_matrix = _bridge_redo.pop()
            result = _snap()

        elif method == "clear_bridge":
            _push_undo(_bridge_undo, _bridge_redo,
                       _cp(getattr(bridge.state, "bridge_matrix", None)))
            result = bridge.rpc_call("clear_bridge", params)

        # ── Reset ─────────────────────────────────────────────
        elif method == "reset_state":
            _brush_undo.clear();   _brush_redo.clear()
            _polygon_undo.clear(); _polygon_redo.clear()
            _bridge_undo.clear();  _bridge_redo.clear()
            _polygons.clear()
            result = bridge.rpc_call("reset_state", params)

        else:
            result = bridge.rpc_call(method, params)

        if "tmx_bytes" in result and isinstance(result["tmx_bytes"], (bytes, bytearray)):
            result["tmx_bytes"] = base64.b64encode(result["tmx_bytes"]).decode("ascii")

        return jsonify({"ok": True, "result": result})

    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 400


# ─────────────────────────────────────────────────────────────
#  TILESET IMPORT — Ground / Wall / Items / Units PNG
# ─────────────────────────────────────────────────────────────
#  MAP UTILITY
# ─────────────────────────────────────────────────────────────

@app.route("/extract_seed", methods=["POST"])
def extract_seed():
    data = _json_body()
    tmx_text = data.get("tmx_text", "")
    grid_size = int(data.get("grid_size", 5))
    if not tmx_text:
        return jsonify({"error": "No tmx_text"}), 400
    try:
        tree = _parse_tmx_root(tmx_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    map_w = int(tree.get("width", 160)); map_h = int(tree.get("height", 160))
    water_gids = _collect_water_gids(tree)
    use_prop = len(water_gids) > 0
    try:
        tiles = _decode_layer_gids(tree, "Ground", map_w, map_h)
    except ValueError as exc:
        msg = str(exc).replace("Ground decode failed", "Decode failed")
        return jsonify({"error": msg}), 400
    cell_h = map_h // grid_size; cell_w = map_w // grid_size
    grid = []
    for gr in range(grid_size):
        row = []
        for gc in range(grid_size):
            land = 0; total = 0
            for r in range(gr*cell_h, min((gr+1)*cell_h, map_h)):
                for c in range(gc*cell_w, min((gc+1)*cell_w, map_w)):
                    t = tiles[r*map_w+c]
                    land += (0 if t in water_gids else 1) if use_prop else (1 if (t-201)>=31 else 0)
                    total += 1
            row.append(1 if total>0 and land/total>0.5 else 0)
        grid.append(row)
    return jsonify({"ok": True, "grid": grid, "width": map_w, "height": map_h})


@app.route("/import_map", methods=["POST"])
def import_map():
    data = _json_body()
    tmx_text = data.get("tmx_text", "")
    try:
        tree = _parse_tmx_root(tmx_text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    map_w = int(tree.get("width", 160)); map_h = int(tree.get("height", 160))
    TILE_ID_OFFSET = 201
    water_gids = _collect_water_gids(tree)
    use_prop = len(water_gids) > 0
    try:
        tiles = _decode_layer_gids(tree, "Ground", map_w, map_h)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    height_map = [
        [
            0 if (gid in water_gids if use_prop else (gid - TILE_ID_OFFSET) < 31) else 1
            for gid in tiles[r * map_w:(r + 1) * map_w]
        ]
        for r in range(map_h)
    ]
    tilesets_out = _extract_embedded_tilesets(tree)
    target_h = max(40, min(2000, int(data.get("target_h", map_h))))
    target_w = max(40, min(2000, int(data.get("target_w", map_w))))
    height_map = np.array(height_map, dtype=np.int32)
    if target_h != map_h or target_w != map_w:
        ri = (np.arange(target_h)*map_h//target_h).astype(int)
        ci = (np.arange(target_w)*map_w//target_w).astype(int)
        height_map = height_map[np.ix_(ri, ci)]
    out_h, out_w = height_map.shape
    bridge.state.height = out_h; bridge.state.width = out_w
    bridge.state.coastline_height_map = height_map
    bridge.state.height_map = height_map
    bridge.state.randomized_matrix = (height_map > 0).astype(int)
    bridge.state.wall_matrix   = np.zeros((out_h, out_w), dtype=int)
    bridge.state.bridge_matrix = np.zeros((out_h, out_w), dtype=int)
    bridge.state.items_matrix  = np.zeros((out_h, out_w), dtype=int)
    bridge.state.units_matrix  = np.zeros((out_h, out_w), dtype=int)
    bridge.state.completed_step = 2
    snap = bridge.rpc_call("get_state_snapshot")
    return jsonify({"ok": True, "width": out_w, "height": out_h,
                    "orig_width": map_w, "orig_height": map_h,
                    "snapshot": snap.get("snapshot"),
                    "tilesets": tilesets_out,
                    "has_tilesets": len(tilesets_out) > 0})


# ─────────────────────────────────────────────────────────────
#  DOWNLOAD — embed all custom + bridge tilesets into TMX
# ─────────────────────────────────────────────────────────────

@app.route("/download", methods=["POST"])
def download():
    import zipfile, io as _io
    data = _json_body()
    tmx_b64   = data.get("tmx_bytes", "")
    png_b64   = data.get("png_bytes", "")
    base_name = data.get("filename", "generated_map").removesuffix(".tmx")
    try: tmx_bytes = base64.b64decode(tmx_b64)
    except Exception: return jsonify({"error": "Invalid tmx_bytes"}), 400

    if _custom_tilesets:
        tmx_bytes = _embed_slot_tilesets(tmx_bytes, _custom_tilesets)
    # NOTE: bridge tilesets are already injected by write_tmx with correct
    # water-bridge tile properties. Do NOT embed again here.

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}.tmx", tmx_bytes)
        if png_b64:
            try: zf.writestr(f"{base_name}_map.png", base64.b64decode(png_b64))
            except Exception: pass
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(buf.getvalue()); tmp.close()
    return send_file(tmp.name, as_attachment=True,
                     download_name=f"{base_name}.zip", mimetype="application/zip")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _cp(m):
    if m is None: return None
    return m.copy() if isinstance(m, np.ndarray) else np.array(m, dtype=int)

def _snap(): return bridge.rpc_call("get_state_snapshot")

def _push_undo(stack, clear_stack, value, max_depth=50):
    stack.append(value)
    if len(stack) > max_depth: stack.pop(0)
    if clear_stack is not None: clear_stack.clear()

def _parse_tmx_root(tmx_text: str):
    if not tmx_text:
        raise ValueError("No tmx_text provided")
    try:
        return ET.fromstring(tmx_text)
    except Exception as exc:
        raise ValueError(f"Invalid XML: {exc}") from exc

def _collect_water_gids(root):
    water_gids = set()
    ts_firsts = sorted(int(t.get("firstgid", 0)) for t in root.findall("tileset"))
    for ts_el in root.findall("tileset"):
        first_gid = int(ts_el.get("firstgid", 0))
        ts_name = (ts_el.get("name") or "").lower()
        if "deep water" in ts_name or ts_name == "shallow water":
            nxt = next((g for g in ts_firsts if g > first_gid), first_gid + 100)
            water_gids.update(range(first_gid, nxt))
        for tile_el in ts_el.findall("tile"):
            tile_id = int(tile_el.get("id", 0))
            for prop in tile_el.findall(".//property"):
                if prop.get("name") == "water":
                    water_gids.add(first_gid + tile_id)
    return water_gids

def _decode_layer_gids(root, layer_name: str, map_w: int, map_h: int):
    import gzip as gz_mod
    import struct
    import zlib
    for layer in root.findall("layer"):
        if layer.get("name") != layer_name:
            continue
        data_el = layer.find("data")
        if data_el is None:
            continue
        raw = (data_el.text or "").strip().replace("\n", "").replace(" ", "")
        raw += "=" * (-len(raw) % 4)
        try:
            compressed = base64.b64decode(raw)
            if compressed[:2] == b"\x1f\x8b":
                decoded = gz_mod.decompress(compressed)
            else:
                decoded = zlib.decompress(compressed)
            return struct.unpack_from(f"<{map_w * map_h}I", decoded)
        except Exception as exc:
            raise ValueError(f"{layer_name} decode failed: {exc}") from exc
    raise ValueError(f"No {layer_name} layer found")

def _extract_embedded_tilesets(root):
    name_map = {
        "AutoLight": "ground",
        "large-rock": "wall",
        "export_items": "items",
        "50pCommandCenter": "units",
    }
    tilesets_out = {}
    for ts_elem in root.findall("tileset"):
        key = name_map.get(ts_elem.get("name", ""))
        if not key:
            continue
        for prop in ts_elem.findall(".//property"):
            if prop.get("name") != "embedded_png":
                continue
            tilesets_out[key] = {
                "name": ts_elem.get("name"),
                "firstGid": int(ts_elem.get("firstgid", 1)),
                "columns": int(ts_elem.get("columns", 1)),
                "tileCount": int(ts_elem.get("tilecount", 0)),
                "tileWidth": int(ts_elem.get("tilewidth", 20)),
                "tileHeight": int(ts_elem.get("tileheight", 20)),
                "png": "".join((prop.get("value") or prop.text or "").split()),
            }
            break
    return tilesets_out

def _embed_slot_tilesets(tmx_bytes: bytes, custom: dict) -> bytes:
    SLOT_TO_NAME = {"ground":"AutoLight","wall":"large-rock",
                    "items":"export_items","units":"50pCommandCenter"}
    try:
        import io as _io
        tree = ET.ElementTree(ET.fromstring(tmx_bytes.decode("utf-8")))
        root = tree.getroot()
        for slot, ts_data in custom.items():
            ts_name = SLOT_TO_NAME.get(slot)
            if not ts_name: continue
            for ts_el in root.findall("tileset"):
                if ts_el.get("name") == ts_name:
                    props = ts_el.find("properties")
                    if props is None: props = ET.SubElement(ts_el, "properties")
                    found = False
                    for prop in props.findall("property"):
                        if prop.get("name") == "embedded_png":
                            prop.text = "\n" + ts_data["png_b64"] + "\n"; found = True; break
                    if not found:
                        p = ET.SubElement(props, "property")
                        p.set("name", "embedded_png")
                        p.text = "\n" + ts_data["png_b64"] + "\n"
                    ts_el.set("columns",    str(ts_data["columns"]))
                    ts_el.set("tilecount",  str(ts_data["tileCount"]))
                    ts_el.set("tilewidth",  str(ts_data["tileWidth"]))
                    ts_el.set("tileheight", str(ts_data["tileHeight"]))
                    break
        buf = _io.BytesIO(); tree.write(buf, encoding="UTF-8", xml_declaration=True)
        return buf.getvalue()
    except Exception: return tmx_bytes

def _embed_bridge_png(tmx_bytes: bytes, bts: dict) -> bytes:
    """Append (or update) a <tileset> element for the bridge in the TMX."""
    try:
        import io as _io
        tree = ET.ElementTree(ET.fromstring(tmx_bytes.decode("utf-8")))
        root = tree.getroot()
        name = bts.get("name", "bridge")
        ts_el = next((t for t in root.findall("tileset") if t.get("name")==name), None)
        if ts_el is None:
            ts_el = ET.Element("tileset")
            first_layer = root.find("layer")
            if first_layer is not None: root.insert(list(root).index(first_layer), ts_el)
            else: root.append(ts_el)
        ts_el.set("firstgid",   str(bts["firstgid"]))
        ts_el.set("name",       name)
        ts_el.set("tilewidth",  str(bts.get("tilewidth", 20)))
        ts_el.set("tileheight", str(bts.get("tileheight", 20)))
        ts_el.set("columns",    str(bts.get("columns", 3)))
        ts_el.set("tilecount",  str(bts.get("tilecount", 12)))
        props = ts_el.find("properties")
        if props is None: props = ET.SubElement(ts_el, "properties")
        found = False
        for prop in props.findall("property"):
            if prop.get("name") == "embedded_png":
                prop.text = "\n" + bts["png"] + "\n"; found = True; break
        if not found:
            p = ET.SubElement(props, "property"); p.set("name","embedded_png")
            p.text = "\n" + bts["png"] + "\n"
        buf = _io.BytesIO(); tree.write(buf, encoding="UTF-8", xml_declaration=True)
        return buf.getvalue()
    except Exception: return tmx_bytes

def _rasterize_all_polygons(polygons, height, width):
    wall = np.zeros((height, width), dtype=int)
    for poly in polygons:
        verts = poly.get("vertices", [])
        if not poly.get("closed", False) or len(verts) < 3: continue
        mask = _rasterize_polygon(verts, height, width, poly.get("edgeGaps", []))
        np.maximum(wall, mask, out=wall)
    return wall

def _rasterize_polygon(vertices, height, width, edge_gaps=None):
    mask = np.zeros((height, width), dtype=np.uint8)
    n = len(vertices)
    if n < 3: return mask
    rows_v = [v[0] for v in vertices]
    r_min = max(0, int(min(rows_v))); r_max = min(height-1, int(max(rows_v)))
    for r in range(r_min, r_max+1):
        crossings = []
        for i in range(n):
            r1,c1=float(vertices[i][0]),float(vertices[i][1])
            r2,c2=float(vertices[(i+1)%n][0]),float(vertices[(i+1)%n][1])
            if (r1<=r<r2) or (r2<=r<r1): crossings.append(c1+(r-r1)/(r2-r1)*(c2-c1))
        crossings.sort()
        for k in range(0, len(crossings)-1, 2):
            c0=max(0,int(crossings[k])); c1=min(width-1,int(crossings[k+1]))
            mask[r, c0:c1+1] = 1
    for i in range(n):
        if edge_gaps and i < len(edge_gaps) and edge_gaps[i]: continue
        r1,c1=int(vertices[i][0]),int(vertices[i][1])
        r2,c2=int(vertices[(i+1)%n][0]),int(vertices[(i+1)%n][1])
        for pr,pc in _bresenham(r1,c1,r2,c2):
            if 0<=pr<height and 0<=pc<width: mask[pr,pc]=1
    return mask

def _bresenham(r1, c1, r2, c2):
    pts=[]; dr=abs(r2-r1); dc=abs(c2-c1)
    sr=1 if r1<r2 else -1; sc=1 if c1<c2 else -1; err=dr-dc
    while True:
        pts.append((r1,c1))
        if r1==r2 and c1==c2: break
        e2=2*err
        if e2>-dc: err-=dc; r1+=sr
        if e2< dr: err+=dr; c1+=sc
    return pts


# ─────────────────────────────────────────────────────────────
#  TILESET REGISTRY ROUTES
# ─────────────────────────────────────────────────────────────
import tileset_registry as _reg
import tile_analyzer as _ta

# Built-in tileset names that ship with every blueprint
_BUILTIN_NAMES = frozenset({
    "AutoLight", "large-rock", "export_items",
    "50pCommandCenter", "export_units",
})
_BUILTIN_RANGES = [(201, 335), (336, 349)]

def _is_builtin(name: str, firstgid: int) -> bool:
    if name in _BUILTIN_NAMES:
        return True
    return any(lo <= firstgid <= hi for lo, hi in _BUILTIN_RANGES)

def _guess_role(name: str, tilewidth: int, tileheight: int) -> str:
    nl = name.lower()
    if any(k in nl for k in ("bridge", "pont", "jembatan")):
        return "bridge"
    if any(k in nl for k in ("unit", "tank", "soldier", "mech", "vehicle")):
        return "unit"
    if any(k in nl for k in ("item", "resource", "struct", "building")):
        return "items"
    if tilewidth >= 32 or tileheight >= 32:
        return "unit"
    return "terrain"


@app.route("/bridge-mapper")
def bridge_mapper():
    return render_template("bridge_mapper.html")


@app.route("/tileset-wizard")
def tileset_wizard_page():
    return render_template("tileset_wizard.html")


# Registry routes → see registry_routes.py (registered as Blueprint)



# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  Rusted Warfare Map Generator — Full Version")
    print("  Buka di browser: http://localhost:5000")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False)
