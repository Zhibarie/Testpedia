# Mappedia — Rusted Warfare Map Generator

Mappedia is a **Flask-based Rusted Warfare map generator** with a web UI to create, edit, preview, and export `.tmx` maps.

## Program Overview

Main capabilities:

- Generate coastline, height/ocean layers, command centers, resources, and final tile IDs.
- Import TMX and extract seed/height data.
- Manage custom tilesets with a persistent registry (terrain/unit/items/bridge).
- Edit walls (brush/polygon), draw bridges, and place custom units/items.
- Preview output and export TMX.

---

## Requirements

### 1) Software

- Python **3.10+** (3.10 or 3.11 recommended)
- `pip`

### 2) Python Dependencies

Core dependencies:

- `flask`
- `numpy`
- `perlin-noise`
- `scipy`

All dependencies are listed in `requirements.txt`.

---

## Installation

From the project root:

```bash
cd /workspace/Testpedia
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

---

## Run

Start the Flask server:

```bash
python app.py
```

Open:

- `http://localhost:5000`

This loads the main map generator UI.

---

## Quick Usage Flow

1. Set map size (height/width), mirroring, player count, and resource count.
2. Generate coastline from the seed grid.
3. Generate height/ocean layers (Perlin-based).
4. Place command centers (manual/random).
5. Place resources (manual/random).
6. Finalize map tile IDs.
7. Export/download TMX.

Additional tools:

- Wall editor (brush/polygon)
- Bridge drawing tool
- Tileset registry for custom tilesets

---

## Important Endpoints (Summary)

- `GET /` → main UI.
- `POST /rpc/<method>` → main stateful backend RPC entrypoint.
- `GET /registry` and `/registry/*` → tileset registry management.
- `POST /extract_seed` → extract seed/height data from TMX.
- `POST /import_map` → import TMX into app state.

---

## Core Files

- `app.py` → Flask app and primary routes.
- `bridge.py` → RPC dispatcher and state orchestration.
- `map_pipeline.py` → map generation pipeline.
- `registry_routes.py` → tileset registry routes.
- `tileset_registry.py` → persistent tileset registry (`tileset_registry.json`).
- `templates/index.html` → main web UI.

---

## Troubleshooting

### 1) Port 5000 already in use

Run on another port:

```bash
python -m flask --app app run -p 5001
```

### 2) Missing module/import errors

Ensure virtual environment is active and dependencies are installed:

```bash
pip install -r requirements.txt
```

### 3) Preview/canvas not rendering

- Hard refresh the browser (`Ctrl+F5`).
- Check browser console for JavaScript errors.
- Verify `/rpc/*` responses return HTTP 200 with JSON `ok: true`.

---

## Notes

- `tileset_registry.json` stores persistent tileset registry data.
- `generator_blueprint*.tmx` files are used as base templates for map layers/tilesets.
- The project runs well on Linux/macOS and also works on Windows with equivalent venv commands.
