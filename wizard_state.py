from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import numpy as np

class WizardStep(IntEnum):
    COASTLINE = 0
    HILLS = 1
    HEIGHT_OCEAN = 2
    COMMAND_CENTERS = 3
    RESOURCES = 4
    FINALIZE = 5

@dataclass
class WizardState:
    initial_matrix: Optional[list] = None
    height: int = 160
    width: int = 160
    mirroring: str = "vertical"
    pattern: int = 1
    num_height_levels: int = 7
    num_ocean_levels: int = 3
    num_command_centers: int = 4
    num_resource_pulls: int = 12
    output_path: str = ""
    randomized_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    coastline_height_map: Optional[np.ndarray] = field(default=None, repr=False)
    wall_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    perlin_seed: Optional[int] = None
    perlin_map: Optional[np.ndarray] = field(default=None, repr=False)
    height_map: Optional[np.ndarray] = field(default=None, repr=False)
    units_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    cc_positions: List[Tuple[int,int]] = field(default_factory=list)
    cc_groups: List[Any] = field(default_factory=list)
    items_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    resource_positions: List[Tuple[int,int]] = field(default_factory=list)
    resource_groups: List[Any] = field(default_factory=list)
    id_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    current_step: WizardStep = WizardStep.COASTLINE
    completed_step: int = -1
    hill_drawing_mode: str = "brush"
    _smoothness: float = 0.0
    coast_smooth_passes: int = 1          # 0=none, 1=light, 2=medium, 3=heavy, 4=very heavy
    bridge_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    # Multiple bridge tilesets keyed by name (e.g. "Forest", "Jungle", "Lava")
    # Each entry: {png, name, firstgid, columns, tilecount, tilewidth, tileheight, tiles}
    bridge_tilesets: dict = field(default_factory=dict)
    active_bridge_name: str = ""   # which tileset is active for drawing

    @property
    def bridge_tileset(self):
        """Backwards-compat: return active tileset dict."""
        return self.bridge_tilesets.get(self.active_bridge_name)
    # Custom terrain tile mapping — overrides hardcoded AutoLight layout in run_finalize.
    # Format: {level_pair_name: [flat_below, center, NW, N, NE, W, E, SW, S, SE, iTL, iTR, iBL, iBR]}
    # Level pair names: "water_sand","sand_grass","grass_soil","soil_swamp",
    #                   "swamp_stone","stone_snow","snow_ice","deep_water_water","ocean_deep_water"
    # Also accepts "decoration" key: {height_level: [tile_id, ...]}
    custom_terrain_mapping: Optional[dict] = field(default=None)

    # Active non-bridge tilesets loaded from registry, keyed by role:
    #   "terrain" – replaces AutoLight PNG (slot replacement, firstgid stays 201)
    #   "unit"    – injected as new <tileset> in TMX with its own firstgid
    #   "items"   – injected as new <tileset> in TMX with its own firstgid
    # Each value: {name, firstgid, tilecount, columns, tilewidth, tileheight, png}
    active_tilesets: dict = field(default_factory=dict)

    # Custom placements — separate from CC/resource placements
    # Each entry: (row, col, gid)
    # _custom_units  → written to units_matrix
    # _custom_items  → written to items_matrix (non-resource)
    # histories stored as list-of-snapshots for undo
    _custom_units:        list = field(default_factory=list, repr=False)
    _custom_unit_history: list = field(default_factory=list, repr=False)
    _custom_items:        list = field(default_factory=list, repr=False)
    _custom_item_history: list = field(default_factory=list, repr=False)

    def invalidate_from(self, step: WizardStep):
        if step <= WizardStep.COASTLINE:
            self.randomized_matrix = None; self.coastline_height_map = None
        if step <= WizardStep.HILLS:
            self.wall_matrix = None
            self.bridge_matrix = None
        # bridge_tilesets, active_bridge_name, coast_smooth_passes, custom_terrain_mapping
        # are user preferences — they survive invalidation.
        if step <= WizardStep.HEIGHT_OCEAN:
            self.perlin_seed = None; self.perlin_map = None; self.height_map = None
        if step <= WizardStep.COMMAND_CENTERS:
            self.units_matrix = None; self.cc_positions = []; self.cc_groups = []
        if step <= WizardStep.RESOURCES:
            self.items_matrix = None; self.resource_positions = []; self.resource_groups = []
        if step <= WizardStep.FINALIZE: self.id_matrix = None
        self.completed_step = min(self.completed_step, int(step) - 1)
        # coast_smooth_passes and custom_terrain_mapping are user preferences
        # — they survive invalidation so re-generating uses the same settings.
