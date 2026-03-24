"""
registry_routes.py — Flask Blueprint for all /registry/* endpoints.

Extracted from app.py to keep the main app file lean.
Register with: app.register_blueprint(registry_bp)
"""
from flask import Blueprint, request, jsonify
import bridge
import tileset_registry as _reg

registry_bp = Blueprint("registry", __name__)


def _json_body() -> dict:
    """Return request JSON body as dict (empty when missing/invalid)."""
    return request.get_json(force=True, silent=True) or {}


def _tile_dims(data: dict) -> tuple[int, int]:
    """Parse tile size fields from a request payload."""
    return int(data.get("tilewidth", 20)), int(data.get("tileheight", 20))


# ── Inline helpers ────────────────────────────────────────────────────────────

def _discover_tilesets_inline(tmx_xml: str) -> dict:
    """Parse TMX and return non-builtin tilesets with PNG previews."""
    import xml.etree.ElementTree as ET
    BUILTIN = {"AutoLight","large-rock","export_items","50pCommandCenter"}
    try:
        root = ET.fromstring(tmx_xml)
    except ET.ParseError as e:
        return {"discovered": [], "skipped": [], "error": str(e)}

    discovered, skipped = [], []
    for ts in root.findall("tileset"):
        name = ts.get("name","")
        if name in BUILTIN:
            skipped.append(name); continue
        firstgid   = int(ts.get("firstgid","1"))
        tilecount  = int(ts.get("tilecount","0"))
        columns    = int(ts.get("columns","1"))
        tilewidth  = int(ts.get("tilewidth","20"))
        tileheight = int(ts.get("tileheight","20"))
        # Try embedded PNG
        png = ""
        prop = ts.find(".//property[@name='embedded_png']")
        if prop is not None:
            raw = prop.get("value","") or prop.text or ""
            png = "".join(raw.split())
        if not png:
            skipped.append(name); continue
        already = _reg.get_tileset(name) is not None
        # Guess role
        nl = name.lower()
        role = "bridge" if any(x in nl for x in ["bridge","pont","jembatan"]) \
            else "unit"  if any(x in nl for x in ["unit","tank","soldier","mech"]) \
            else "items" if any(x in nl for x in ["item","resource","building"]) \
            else "terrain"
        discovered.append({
            "name": name, "firstgid": firstgid, "tilecount": tilecount,
            "columns": columns, "tilewidth": tilewidth, "tileheight": tileheight,
            "png": png, "has_embedded_png": True, "has_external_image": False,
            "suggested_role": role, "already_registered": already,
        })
    return {"discovered": discovered, "skipped": skipped}


# AutoLight constants (columns=27, firstgid=201)
_AL_COLS = 27
_AL_FIRSTGID = 201
_AL_CENTERS = {-2:83,-1:28,0:31,1:34,2:37,3:40,4:43,5:46,6:49,7:52}
_AL_LABELS  = {-2:"Deep Ocean",-1:"Deep Water",0:"Water",1:"Sand",
               2:"Grass",3:"Soil",4:"Swamp",5:"Stone",6:"Snow",7:"Ice"}

_AL_COLORS = {-2:"#04101f",-1:"#071a3a",0:"#0e3a6e",1:"#c9a84c",
              2:"#3d7a28",3:"#7c5230",4:"#4a5e30",5:"#6b6b6b",6:"#b0c8d8",7:"#a0c4e8"}

def _get_autolight_reference_inline() -> dict:
    """Return AutoLight center GIDs per terrain level for reference display."""
    centers = []
    for level, local_id in sorted(_AL_CENTERS.items()):
        gid = _AL_FIRSTGID + local_id
        centers.append({
            "level": level, "local_id": local_id, "gid": gid,
            "label": _AL_LABELS.get(level, f"Level {level}"),
            "color": _AL_COLORS.get(level, "#888"),
        })
    return {"ok": True, "centers": centers,
            "columns": _AL_COLS, "firstgid": _AL_FIRSTGID}


# ── Terrain suggestion / preview ─────────────────────────────────────────────

@registry_bp.route("/registry/suggest-terrain", methods=["POST"])
def registry_suggest_terrain():
    data = _json_body()
    png_b64 = data.get("png", "")
    tilecount = int(data.get("tilecount", 0))
    columns   = int(data.get("columns", 1))
    tilewidth, tileheight = _tile_dims(data)
    if not png_b64 or not tilecount:
        return jsonify({"error": "Missing png or tilecount"}), 400
    try:
        import tile_analyzer as _ta
        suggestions = _ta.suggest_terrain_centers(
            png_b64, tilecount, columns, tilewidth, tileheight)
        return jsonify({"ok": True, "suggestions": suggestions})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@registry_bp.route("/registry/preview-terrain", methods=["POST"])
def registry_preview_terrain():
    data = _json_body()
    try:
        import tile_analyzer as _ta
        result = _ta.build_terrain_preview(
            data["png"], int(data["tilecount"]),
            int(data["columns"]), int(data.get("tilewidth", 20)),
            int(data.get("tileheight", 20)), data.get("mapping", {}))
        return jsonify({"ok": True, "previews": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@registry_bp.route("/registry/preview-terrain-example", methods=["GET"])
def registry_preview_terrain_example():
    """Return AutoLight reference data for the terrain visual panel."""
    try:
        result = _get_autolight_reference_inline()
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Tile extraction / analysis ────────────────────────────────────────────────

@registry_bp.route("/registry/analyze", methods=["POST"])
def registry_analyze():
    """Analyze a tileset PNG — return per-tile color metrics + bridge hints."""
    data = _json_body()
    try:
        import tile_analyzer as _ta
        png       = data.get("png", "")
        tilecount = int(data.get("tilecount", 0))
        columns   = int(data.get("columns", 1))
        tilewidth, tileheight = _tile_dims(data)
        if not png or not tilecount:
            return jsonify({"error": "png and tilecount required"}), 400
        tiles = _ta.analyze_sheet(png, tilecount, columns, tilewidth, tileheight).get("tiles", [])
        return jsonify({"ok": True, "tiles": tiles})
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@registry_bp.route("/registry/tiles", methods=["POST"])
def registry_tiles():
    """Extract individual tile PNGs (base64) from a tileset sheet."""
    data = _json_body()
    try:
        import tile_analyzer as _ta
        png       = data.get("png", "")
        tilecount = int(data.get("tilecount", 0))
        columns   = int(data.get("columns", 1))
        tilewidth, tileheight = _tile_dims(data)
        if not png or not tilecount:
            return jsonify({"error": "png and tilecount required"}), 400
        tiles_b64 = _ta.extract_tiles_b64(png, tilecount, columns, tilewidth, tileheight)
        return jsonify({"ok": True, "tiles": tiles_b64})
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@registry_bp.route("/registry/active-tiles", methods=["GET"])
def registry_active_tiles():
    role = request.args.get("role", "unit")
    try:
        result = bridge.rpc_call("list_active_tiles", {"role": role})
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@registry_bp.route("/registry/discover", methods=["POST"])
def registry_discover():
    """Parse uploaded TMX and return all non-builtin tilesets."""
    try:
        if request.content_type and "multipart" in request.content_type:
            f   = request.files.get("file")
            xml = f.read().decode("utf-8") if f else ""
        else:
            xml = _json_body().get("tmx_xml", "")
        if not xml:
            return jsonify({"error": "No TMX provided"}), 400
        result = _discover_tilesets_inline(xml)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Activate / deactivate ─────────────────────────────────────────────────────

@registry_bp.route("/registry/activate", methods=["POST"])
def registry_activate():
    data = _json_body()
    try:
        result = bridge.rpc_call("activate_tileset", data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@registry_bp.route("/registry/deactivate", methods=["POST"])
def registry_deactivate():
    data = _json_body()
    try:
        result = bridge.rpc_call("deactivate_tileset", data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Register / import / export ────────────────────────────────────────────────

@registry_bp.route("/registry/register", methods=["POST"])
def registry_register():
    data = _json_body()
    try:
        result = bridge.rpc_call("register_tileset", {**data, "activate": True})
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@registry_bp.route("/registry/import", methods=["POST"])
def registry_import():
    """Bulk-import tilesets from a registry JSON export."""
    data = _json_body()
    tilesets = data.get("tilesets", {})
    if not tilesets:
        return jsonify({"error": "No tilesets in payload"}), 400
    imported, errors = [], []
    for name, ts in tilesets.items():
        try:
            bridge.rpc_call("register_tileset", {
                "name": name, "type": ts.get("type","terrain"),
                "png":  ts.get("png",""),
                "tilecount": ts.get("tilecount", 1),
                "columns":   ts.get("columns", 1),
                "tilewidth":  ts.get("tilewidth",  20),
                "tileheight": ts.get("tileheight", 20),
                "tiles":       ts.get("tiles"),
                "bridge_variant": ts.get("bridge_variant","12"),
                "activate": False,
            })
            imported.append(name)
        except Exception as e:
            errors.append({"name": name, "error": str(e)})
    return jsonify({"ok": True, "imported": imported, "errors": errors})


@registry_bp.route("/registry/export", methods=["GET"])
def registry_export():
    """Export entire registry as JSON (includes PNGs)."""
    try:
        reg = _reg.get_all()
        return jsonify({"tilesets": reg})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── List / get / delete ───────────────────────────────────────────────────────

@registry_bp.route("/registry", methods=["GET"])
def registry_list():
    ts_type = request.args.get("type")
    try:
        tilesets = _reg.get_all(ts_type)
        # Strip PNG from list response (large payload)
        slim = {k: {ek: ev for ek, ev in v.items() if ek != "png"}
                for k, v in tilesets.items()}
        return jsonify({"tilesets": slim})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@registry_bp.route("/registry/<path:name>/png", methods=["GET"])
def registry_get_png(name):
    entry = _reg.get_tileset(name)
    if not entry:
        return jsonify({"error": f"'{name}' not found"}), 404
    png = entry.get("png", "")
    if not png:
        return jsonify({"error": "No PNG stored"}), 404
    return jsonify({"ok": True, "name": name, "png": png})


@registry_bp.route("/registry/<path:name>", methods=["DELETE"])
def registry_delete(name):
    try:
        removed = _reg.remove_tileset(name)
        if not removed:
            return jsonify({"error": f"'{name}' not found"}), 404
        return jsonify({"ok": True, "deleted": name})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
