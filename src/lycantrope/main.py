"""Lycantrope entry point: window, fixed-step loop, Rowan in third person (or a free-fly camera).

Controls (play mode)
--------------------
Click            capture the mouse  ·  Esc: pause menu (Resume, Save, Quit; the world waits)
WASD             move (camera-relative)     Shift: sprint  ·  Ctrl: walk  ·  wheel: camera distance
Left click       light attack (click again during a swing to chain up to three)
Right mouse      hold to block              R: heavy attack  ·  Q: cast the Ward (needs a magic weapon)
Space            jump (light attack while dropping: plunge)  ·  C: landing roll (press just before landing)
E / G            pick up loot in reach, rest at a lit Waystone (then fast travel), otherwise drink from the
                 Lifewater flask  ·  G: eat an apple
I                inventory (pause menu: equip, use, drop)  ·  L: level-up screen (out of combat)
M                world map (wheel or +/- zoom, WASD pan, C centre on Rowan, M or Esc close)
H / K            debug: hit reaction / fall dead (the death screen, then the last save loads)
F4 free-fly camera · F7 helmet on/off · F8 grass on/off · F9 music on/off · F10 sound effects on/off ·
F11 NPC voices on/off · N noon/midnight · [ ] an hour back/forward · Ctrl + Up/Down zoom in/out (hold)

Free-fly mode: WASD / QE move, Shift fast, Ctrl slow, wheel base speed.
Always: T turbo (4x speed, walking and flying) · F1 wireframe · F2 fog · F3 debug overlay ·
        F4 fly / back to Rowan · F5 reload shaders · F6 terrain debug view · F12 screenshot

Command line
------------
python -m lycantrope [--cam X,Z,YAW,PITCH[,HEIGHT]] [--screenshot FILE] [--bench N] [--size WxH]
    --cam         start in free-fly mode at X/Z (meters, z = south), yaw/pitch in degrees,
                  HEIGHT above the ground (default 40)
    --screenshot  render a few frames, save a PNG and quit
    --bench       render N frames, print timing and quit
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from datetime import datetime

from pyglm import glm
import moderngl
import numpy as np
import pygame
from PIL import Image

from lycantrope.audio.music import start_music
from lycantrope.audio.sfx import start_sfx
from lycantrope.core.config import Settings
from lycantrope.core.paths import SCREENSHOTS
import lycantrope.core.save as save_sys
from lycantrope.ui import title as title_screen
from lycantrope.core.timing import FixedStep
from lycantrope.render.camera import Camera
from lycantrope.render.characters import CharacterSystem
from lycantrope.render.environment import Sky, Water, default_environment
from lycantrope.render.grass import Grass
from lycantrope.render.props import PropRenderer
from lycantrope.render.shaders import ShaderLibrary
from lycantrope.render.terrain import TerrainRenderer
from lycantrope.ui.hud import DeathScreen, Hud, LevelPanel
from lycantrope.render.pixelbuffer import PixelBuffer
from lycantrope.render.hiz import HiZBuffer
from lycantrope.ui.loading import LoadingScreen
from lycantrope.ui.navigation import Compass, RegionBanner, WorldMap
from lycantrope.ui.loot import LootView
from lycantrope.ui.pause import PauseMenu
from lycantrope.ui.waystone import WaystoneMenu, travel_spot
from lycantrope.world.inventory import Drop, GroundLoot, LootTables, load_items_file
from lycantrope.ui.overlay import LockReticle, Subtitles, TextOverlay
from lycantrope.world.lockon import LockOn
from lycantrope.world.actors import TURBO, RowanInput
from lycantrope.world.heightfield import Heightfield
from lycantrope.world.props import load_placements, paint_ground, paint_paths

CAM_PIVOT_HEIGHT = 1.55


def fit_to_desktop(size: tuple[int, int], margin: float = 0.9) -> tuple[int, int]:
    """Shrink a window size (keeping its aspect ratio) so it fits on the screen.

    With Windows display scaling (e.g. 125%), a 1920×1080 screen is only
    1536×864 to the game, so a 1600×900 window would hang off the edges.
    """
    try:
        dw, dh = pygame.display.get_desktop_sizes()[0]
    except (pygame.error, IndexError):
        return size
    w, h = size
    scale = min(1.0, dw * margin / w, dh * margin / h)
    return max(320, int(w * scale)), max(180, int(h * scale))


class App:
    def __init__(self, settings: Settings, args: argparse.Namespace,
                 save_slot: int = 0, save_data: dict | None = None) -> None:
        self.settings = settings
        self.args = args
        self.save_slot = save_slot
        pygame.mixer.pre_init(44100, -16, 2, 2048)       # a roomy buffer: music without crackles
        pygame.init()
        for attr, val in ((pygame.GL_CONTEXT_MAJOR_VERSION, 4), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                          (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE),
                          (pygame.GL_DEPTH_SIZE, 24), (pygame.GL_DOUBLEBUFFER, 1)):
            pygame.display.gl_set_attribute(attr, val)
        size = (settings.width, settings.height)
        if args.size:
            size = tuple(int(v) for v in args.size.lower().split("x"))
        if not settings.fullscreen:
            size = fit_to_desktop(size)
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if settings.fullscreen else pygame.RESIZABLE)
        vsync = 0 if (args.bench or not settings.vsync) else 1
        pygame.display.set_mode(size, flags, vsync=vsync)
        pygame.display.set_caption("Lycantrope")
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        t0 = time.perf_counter()
        w, h = pygame.display.get_window_size()
        loading = LoadingScreen(self.ctx, w, h)
        loading.draw(0.0)
        self.shaders = ShaderLibrary(self.ctx)
        loading.draw(0.03, "Shaders")
        self.world = Heightfield()
        placements = load_placements()
        paint_paths(self.world, placements)          # before the terrain uploads the road map
        paint_ground(self.world, placements)         # loose hay and other painted patches
        loading.draw(0.07, "World")
        self.terrain = TerrainRenderer(self.ctx, self.shaders, self.world)
        self.props = PropRenderer(self.ctx, self.shaders, self.world, placements)
        self.grass = Grass(self.ctx, self.shaders, self.world, self.terrain)
        loading.draw(0.10, "Terrain")

        def _char_prog(done: int, total: int, key: str) -> None:
            loading.draw(0.10 + done / total * 0.82, key.replace("_", " ").title())

        self.chars = CharacterSystem(self.ctx, self.shaders, self.world, on_progress=_char_prog)
        self.chars.set_colliders(self.props.colliders)   # houses, trunks, fences and rocks are solid
        loading.draw(0.95, "Sky & water")
        self.sky = Sky(self.ctx, self.shaders)
        self.water = Water(self.ctx, self.shaders, self.world, self.terrain.tex_height, self.terrain.tex_water)
        loading.draw(0.98, "UI")
        self.overlay = TextOverlay(self.ctx, self.shaders)
        self.subtitles = Subtitles(TextOverlay(self.ctx, self.shaders, size=18, mono=False))
        self.lock = LockOn()                              # lock-on targeting (CMB-02)
        self.reticle = LockReticle(self.ctx, self.shaders)
        self.hud = Hud(self.ctx, self.shaders)              # bars, XP, flask, hit arc (UIX-15..22)
        self.levels = LevelPanel(self.ctx, self.shaders)    # level-up screen (UIX-06)
        self.death = DeathScreen(self.ctx, self.shaders)    # UIX-09
        self.compass = Compass(self.ctx, self.shaders, self.world)      # UIX-02
        self.region = RegionBanner(self.ctx, self.shaders)
        self.map = WorldMap(self.ctx, self.shaders, self.world)
        self.rarity_colors = load_items_file()["rarity_colors"]
        self.loot_tables = LootTables()                     # WLD-17
        self.ground = GroundLoot()                          # drops and chests in the world (UIX-26)
        self.loot_rng = random.Random()
        self.loot_view = LootView(self.ctx, self.shaders, self.chars.progress.inv.catalog, self.rarity_colors)
        self.pause = PauseMenu(self.ctx, self.shaders, self)   # UIX-07
        self.waystones = WaystoneMenu(self.ctx, self.shaders, self)   # rest and fast travel (WLD-09)
        self.fade = 0.0                                     # seconds of fade-in from black left (after travel)
        self.quit_requested = False
        loading.draw(1.0)
        loading.release()
        self.pixel_buf = PixelBuffer(self.ctx, settings.pixel_scale, settings.pixel_smooth) if settings.pixel_scale >= 2 else None
        self.hiz = HiZBuffer(self.ctx) if self.pixel_buf is not None else None
        self.load_seconds = time.perf_counter() - t0
        # no music for benchmarks and screenshots (timing and silence)
        quiet = args.bench or args.screenshot or settings.music_volume <= 0
        self.music = None if quiet else start_music(settings.music_volume)
        no_sfx = args.bench or args.screenshot or settings.sfx_volume <= 0
        self.sfx = None if no_sfx else start_sfx(settings.sfx_volume)
        if self.sfx is not None:                          # impacts for hits that play no hit animation
            self.chars.cue_sound = self.sfx.director.cue

        if save_data is not None:
            self.chars.apply_save(save_data)
        self.place_chests(placements)
        self.play_time_s_offset = float((save_data or {}).get("play_time_s", 0.0))

        self.camera = Camera(fov_deg=settings.fov_deg, far=settings.view_distance_m)
        self.play = self.chars.rowan is not None and not args.cam
        self.orbit_dist = 2.8
        self.orbit_pitch = -0.22
        self.orbit_yaw = math.pi - self.chars.rowan.a.heading if self.chars.rowan else 0.0
        self.place_camera()
        self.clock = FixedStep(60)
        self.base_speed = 25.0
        self.fog = True
        from lycantrope.render.environment import sky_config
        from lycantrope.world.daytime import DayClock
        self.day = DayClock(sky_config().get("day"))               # day and night, on play time (WLD-11)
        if args.time is not None:
            self.day.set(float(args.time))
        self.turbo = False
        self.overlay_mode = 2                          # 0 full  1 minimal  2 off  3 controls  (F3 cycles)
        self.captured = False
        self.time_s = 0.0
        self.frame_times: list[float] = []
        self.overlay_timer = 0.0
        self.pressed: set[str] = set()                # one-shot actions since the last tick
        self.ticks = 0
        if args.orbit and self.chars.rowan:
            y, p, d = (float(v) for v in args.orbit.split(","))
            self.orbit_yaw = math.pi - self.chars.rowan.a.heading + math.radians(y)
            self.orbit_pitch, self.orbit_dist = math.radians(p), d
        if args.act and args.act.startswith("teleport:") and self.chars.rowan:
            x, z = (float(v) for v in args.act.split(":")[1].split(","))
            a = self.chars.rowan.a
            a.x, a.z = x, z
            a.y = self.world.height_at(x, z)
            a.save_prev()

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        """Write the current slot to disk and sync the in-memory death-rollback."""
        if not self.chars.rowan:
            return
        self.chars.update_spawn()
        sp = self.chars.spawn_point
        if self.args.bench or self.args.screenshot:      # test runs never touch the player's save slots
            self.chars.progress.save()
            return
        save_sys.save(
            self.save_slot,
            self.chars.progress.snapshot(),
            {"x": sp[0], "z": sp[1], "heading": sp[2]},
            self.play_time_s_offset + self.time_s,
        )
        self.chars.progress.save()          # keep _saved in sync with the disk
        self.hud.state.note("Game saved")

    # ------------------------------------------------------------------ setup
    def place_camera(self) -> None:
        cam = self.camera
        if self.args.cam:
            vals = [float(v) for v in self.args.cam.split(",")]
            x, z, yaw, pitch = vals[:4]
            above = vals[4] if len(vals) > 4 else 40.0
            cam.position = glm.vec3(x, self.world.height_at(x, z) + above, z)
            cam.yaw, cam.pitch = math.radians(yaw), math.radians(pitch)
        elif self.chars.rowan is not None:
            self.follow_camera(1.0)
        else:                                   # above Brackenvale Village, Rowan's home
            v = next((p for p in self.world.landmarks if p.name == "Brackenvale Village"),
                     self.world.waystones[0])
            cam.position = glm.vec3(v.x + 55.0, v.height + 22.0, v.z + 75.0)
            cam.look_at((v.x - 5.0, v.height + 2.0, v.z + 5.0))
        cam.prev_position = glm.vec3(cam.position)

    def follow_camera(self, alpha: float) -> None:
        """Third-person orbit around Rowan, kept above the ground."""
        a = self.chars.rowan.a
        cam = self.camera
        pivot = glm.vec3(a.px + (a.x - a.px) * alpha, a.py + (a.y - a.py) * alpha + CAM_PIVOT_HEIGHT,
                         a.pz + (a.z - a.pz) * alpha)
        cam.yaw, cam.pitch = self.orbit_yaw, self.orbit_pitch
        pos = pivot - cam.forward() * self.orbit_dist
        # pull the camera in rather than let it end up inside a house or a trunk
        t = self.props.colliders.clear_distance((pivot.x, pivot.y, pivot.z), (pos.x, pos.y, pos.z))
        if t < 1.0:
            pos = pivot - cam.forward() * max(0.6, self.orbit_dist * t)
        floor = max(self.world.height_at(pos.x, pos.z) + 0.35, self.world.water_at(pos.x, pos.z) + 0.3)
        if pos.y < floor:                         # stay above the ground and the water
            pos.y = floor
            d = pivot - pos
            cam.yaw = math.atan2(d.x, -d.z)
            cam.pitch = math.atan2(d.y, math.hypot(d.x, d.z))
        cam.position = pos
        cam.prev_position = glm.vec3(pos)

    # ------------------------------------------------------------------ input
    def set_capture(self, on: bool) -> None:
        self.captured = on
        if hasattr(pygame.mouse, "set_relative_mode"):
            pygame.mouse.set_relative_mode(on)
        else:
            pygame.event.set_grab(on)
            pygame.mouse.set_visible(not on)
        pygame.mouse.get_rel()

    def mouse_look(self, rel) -> None:
        s = self.settings.mouse_sensitivity
        dy = rel[1] * s * (1 if self.settings.invert_y else -1)
        if self.play and self.lock.locked and self.chars.rowan:
            # locked on: the camera follows the target; a sideways flick switches target
            p = self.camera.position
            self.lock.mouse(rel[0], self.chars.chars, self.chars.rowan.a, p.x, p.z)
            self.orbit_pitch = max(-1.2, min(0.6, self.orbit_pitch + dy))
        elif self.play:
            self.orbit_yaw = (self.orbit_yaw + rel[0] * s) % math.tau
            self.orbit_pitch = max(-1.2, min(0.6, self.orbit_pitch + dy))
        else:
            self.camera.look(rel[0] * s, dy)

    def latest_mouse(self) -> None:
        """Mouse movement that arrived during the world update, so the frame shows it now."""
        for ev in pygame.event.get(pygame.MOUSEMOTION):
            if self.captured:
                self.mouse_look(ev.rel)

    def handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if self.pause.open:                          # the pause menu has the keys and the mouse
                if ev.type == pygame.KEYDOWN:
                    self.pause.key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
                    self.pause.click(ev.pos, ev.button)
                if not self.pause.open:
                    self.set_capture(True)
                continue
            if self.waystones.open:                      # the Waystone travel list
                if ev.type == pygame.KEYDOWN:
                    self.waystones.key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.waystones.click(ev.pos)
                if not self.waystones.open:
                    self.set_capture(True)
                continue
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and not self.captured:
                self.set_capture(True)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.play:
                self.pressed.add("light")
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 2 and self.play:
                self.toggle_lock()
            elif ev.type == pygame.MOUSEMOTION and self.captured:
                self.mouse_look(ev.rel)
            elif ev.type == pygame.MOUSEWHEEL:
                if self.map.open:
                    self.map.zoom(ev.y)
                elif self.play:
                    self.orbit_dist = min(7.0, max(1.8, self.orbit_dist * (0.9 ** ev.y)))
                else:
                    self.base_speed = min(400.0, max(2.0, self.base_speed * (1.25 ** ev.y)))
            elif ev.type == pygame.KEYDOWN and self.map.open:
                x, z, _ = self.nav_position()
                self.map.key(ev.key, (x, z))
            elif ev.type == pygame.KEYDOWN and self.levels.open:
                if self.levels.key(ev.key, self.chars.progress) == "apply":
                    if self.chars.level_up(self.levels.plan):
                        self.levels.plan = {}
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_i):        # pause menu (UIX-07); I opens the inventory
                    self.pause.show(0 if k == pygame.K_i else 3)
                    self.set_capture(False)
                elif k == pygame.K_t:
                    self.turbo = not self.turbo
                    self.overlay_timer = 0.0                 # show the change straight away
                elif k == pygame.K_F1:
                    self.ctx.wireframe = not self.ctx.wireframe
                elif k == pygame.K_F2:
                    self.fog = not self.fog
                elif k == pygame.K_F3:
                    self.overlay_mode = (self.overlay_mode + 1) % 4
                    self.overlay_timer = 0.0
                elif k == pygame.K_F4 and self.chars.rowan is not None:
                    self.play = not self.play
                    if not self.play:
                        self.camera.prev_position = glm.vec3(self.camera.position)
                elif k == pygame.K_F5:
                    if self.play:
                        self.save()
                    else:
                        errors = self.shaders.reload()
                        print("shaders reloaded" if not errors else "shader errors:\n" + "\n".join(errors))
                elif k == pygame.K_F6:
                    self.terrain.debug_view = (self.terrain.debug_view + 1) % 3
                elif k == pygame.K_F7:
                    self.chars.helmet = not self.chars.helmet
                elif k == pygame.K_F8:
                    self.grass.enabled = not self.grass.enabled
                elif k == pygame.K_F9 and self.music is not None and not pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    on = self.music.layer_on                             # all music (Shift+F9 is kept free)
                    on["explore"] = on["combat"] = not (on["explore"] or on["combat"])
                elif k == pygame.K_F11 and self.sfx is not None:
                    self.sfx.voices = not self.sfx.voices
                elif k == pygame.K_F10 and self.sfx is not None:
                    self.sfx.enabled = not self.sfx.enabled
                elif k == pygame.K_TAB and self.play:      # lock on / let go
                    self.toggle_lock()
                elif k == pygame.K_n:                      # debug: jump between noon and midnight
                    self.day.set(0.0 if 6.0 <= self.day.hour < 18.0 else 12.0)
                elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):   # debug: an hour back / forward
                    self.day.set(self.day.hour + (1.0 if k == pygame.K_RIGHTBRACKET else -1.0))
                elif k == pygame.K_F12:
                    SCREENSHOTS.mkdir(exist_ok=True)
                    self.screenshot(SCREENSHOTS / f"lycantrope_{datetime.now():%Y%m%d_%H%M%S}.png")
                elif k == pygame.K_l and self.play:         # level-up screen, out of combat (STA-02)
                    if self.chars.chased(30.0):
                        self.hud.state.note("Can't level up while enemies are after you")
                    elif self.chars.rowan and self.chars.rowan.a.state != "dead":
                        self.levels.toggle()
                elif k == pygame.K_m:                       # world map (UIX-07)
                    if self.play and self.chars.chased(30.0):
                        self.hud.state.note("Can't read the map while enemies are after you")
                    else:
                        x, z, _ = self.nav_position()
                        self.map.toggle((x, z))
                elif k == pygame.K_e and self.play:
                    if not self.take_loot() and not self.use_waystone():   # loot, then a Waystone, else drink
                        self.chars.drink()
                elif k == pygame.K_g and self.play:
                    if not self.chars.eat("apple"):
                        self.hud.state.note("No apples left")
                elif self.play:
                    name = {pygame.K_r: "heavy", pygame.K_q: "cast", pygame.K_h: "hit", pygame.K_k: "die",
                            pygame.K_SPACE: "jump", pygame.K_f: "sheathe", pygame.K_c: "roll"}.get(k)
                    problem = self.chars.progress.cast_problem() if name == "cast" else None
                    if problem:                                        # WPR-01: needs a catalyst
                        self.hud.state.note(problem)
                    elif name == "cast" and self.chars.rite_cooldown > 0:   # CMB-34
                        self.hud.state.note(f"{self.chars.progress.rite()['name']} ready in "
                                            f"{math.ceil(self.chars.rite_cooldown)} s")
                    elif name:
                        self.pressed.add(name)
        return not self.quit_requested

    # ------------------------------------------------------------- simulation
    def update(self, dt: float) -> None:
        if self.pause.open or self.waystones.open:        # the world waits while a menu is up
            return
        self.fade = max(0.0, self.fade - dt)
        keys = pygame.key.get_pressed()
        self.key_zoom(keys, dt)
        inp = RowanInput()
        if self.map.open:
            self.map.update(dt, keys)
        if self.play and (self.levels.open or self.map.open):
            self.pressed.clear()                          # the level-up screen or the map has the keys
        elif self.play:
            inp.move_x = keys[pygame.K_d] - keys[pygame.K_a]
            inp.move_y = keys[pygame.K_w] - keys[pygame.K_s]
            inp.sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            inp.walk = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            inp.turbo = self.turbo
            inp.block = bool(self.captured and pygame.mouse.get_pressed()[2])
            inp.light, inp.heavy = "light" in self.pressed, "heavy" in self.pressed
            inp.cast, inp.hit, inp.die = "cast" in self.pressed, "hit" in self.pressed, "die" in self.pressed
            inp.jump = "jump" in self.pressed
            if self.lock.locked:
                t = self.lock.target.actor
                inp.face = (t.x, t.z)
            inp.sheathe = "sheathe" in self.pressed
            inp.roll = "roll" in self.pressed
            self.pressed.clear()
            act = self.args.act or ""
            if act.startswith("teleport:"):              # teleport:X,Z[:action]
                parts = act.split(":")
                act = parts[2] if len(parts) > 2 else ""
            if act:                                      # --act: scripted input for screenshots
                inp.move_y = max(inp.move_y, 1.0 if act in ("run", "sprint", "walk") else 0.0)
                inp.sprint |= act == "sprint"
                inp.walk |= act == "walk"
                inp.block |= act == "block"
                first = self.ticks == 1
                inp.light |= (act == "light" and first) or (act == "combo" and self.ticks % 20 == 1)
                inp.heavy |= act == "heavy" and first
                inp.cast |= act in ("cast", "ward") and first
                inp.hit |= act == "hit" and first
                inp.die |= act == "die" and first
                inp.jump |= act == "jump" and self.ticks % 50 == 1
                if act == "fight" and self.ticks == 1:        # testing: start beside the nearest enemy
                    ra = self.chars.rowan.a
                    near = self.lock.candidates(self.chars.chars, ra, range_m=5000.0)
                    if near:
                        e = min(near, key=lambda cd: cd[1])[0].actor
                        ra.x, ra.z = e.x + 2.0, e.z + 2.0
                        ra.y = self.world.height_at(ra.x, ra.z)
                        ra.heading = math.atan2(e.x - ra.x, e.z - ra.z)
                        self.orbit_yaw = math.pi - ra.heading
                        ra.save_prev()
                        self.chars.rowan.sheathed = False
                if act in ("lock", "fight") and self.ticks == 3 and not self.lock.locked:   # testing: lock on
                    near = self.lock.candidates(self.chars.chars, self.chars.rowan.a)
                    self.lock.target = min(near, key=lambda cd: cd[1])[0] if near else None
                if act == "fight" and self.lock.locked:      # testing: swing at the locked enemy
                    inp.light |= self.ticks % 40 == 5
                if act == "hurt" and self.ticks == 3:        # testing: a hit from the front-left, XP and a level
                    rc = self.chars.rowan_char
                    ra = rc.actor
                    src = (ra.x + 3 * math.sin(ra.heading + 0.8), ra.z + 3 * math.cos(ra.heading + 0.8))
                    rc.fighter.hurt(rc.fighter.max_hp * 0.4, 0.0, src)
                    rc.fighter.stamina *= 0.5
                    self.chars.events.append(("hurt", 0.0, src, "melee"))
                    self.chars.progress.gain(120)
                    self.chars.events.append(("xp", 120, "Goblin Cutthroat"))
                if act in ("inventory", "loot") and self.ticks == 3:   # testing: some things to carry
                    inv = self.chars.progress.inv
                    for iid, n in (("iron_shortsword", 1), ("flanged_mace", 1), ("trs_04", 1), ("hed_13", 1),
                                   ("wooden_round_shield", 1), ("bread", 2), ("wolf_pelt", 3), ("copper", 42)):
                        inv.add(iid, n)
                    if act == "inventory":
                        self.pause.show(0)
                        self.pause.sel_bag = 2
                    else:
                        ra = self.chars.rowan.a
                        x, z = ra.x + math.sin(ra.heading) * 1.4, ra.z + math.cos(ra.heading) * 1.4
                        self.ground.spawn(x, self.world.height_at(x, z) + 0.1, z,
                                          [["hed_05", 1], ["copper", 9]], "Bandit")
                if act == "waystone" and self.ticks == 3:        # testing: light them all, rest at the crossroads
                    self.chars.progress.discovered.update(p.name for p in self.world.waystones)
                    stone = next(p for p in self.world.waystones if p.name == "Brackenvale Crossroads")
                    self.chars.teleport(*travel_spot(self.world, stone))
                    self.use_waystone()
                if act == "map" and self.ticks == 3:
                    x, z, _ = self.nav_position()
                    self.map.toggle((x, z))
                if act == "levelup" and self.ticks == 3:
                    self.chars.progress.gain(400)
                    self.levels.toggle()
                    self.levels.plan = {"vigor": 2}
            self.ticks += 1
        elif not self.map.open:
            self.fly(dt, keys)
        self.chars.night = self.day.is_night()           # night groups come out (WLD-11)
        self.chars.update(inp, self.orbit_yaw, dt)
        if self.music is not None:
            self.music.update(dt, self.chars.chased(self.music.cfg.get("combat_range_m", 45.0)), self.music_region())
        if self.sfx is not None:
            cam = self.camera
            p, r = cam.position, cam.right()
            talking = self.sfx.update(self.chars.chars, self.chars.rowan.a if self.chars.rowan else None,
                                      self.chars.spells.items, (p.x, p.y, p.z), (r.x, r.y, r.z), dt,
                                      self.ground_sound)
            for speaker, envelope, text in talking:    # they stop, face Rowan and move their jaw
                self.chars.talk(speaker, envelope)
                if text and self.settings.subtitles:
                    self.subtitles.say(self.chars.name_of(speaker), text, len(envelope) / 30.0 + 1.0)
        self.subtitles.update(dt)
        for ev in self.chars.events:                       # hits, XP, levels for the HUD
            self.hud.state.event(ev)
            if ev[0] == "killed":                       # WLD-17: the dead drop their table's loot
                _, kind, (x, y, z), name = ev
                self.ground.spawn(x, y + 0.1, z, self.loot_tables.roll(kind, self.loot_rng), name)
            if ev[0] == "reload":
                self.lock.target = None
                self.levels.open = False
                self.map.open = False
                self.ground.clear_dropped()
                self.reset_chests()
        self.chars.events.clear()
        x, z, _ = self.nav_position()
        self.region.update(dt, self.world.biome_at(x, z))
        rc = self.chars.rowan_char
        if rc is not None and rc.fighter is not None:
            f = rc.fighter
            full = f.hp >= f.max_hp and f.stamina >= f.max_stamina and f.trail <= f.hp and f.ward_t <= 0
            self.hud.state.update(dt, full, self.chars.chased(30.0))
        self.time_s += dt
        self.day.update(dt)
        if self.play and self.chars.rowan:
            ra = self.chars.rowan.a
            self.lock.update(dt, ra)
            yaw = self.lock.yaw_to_target(ra)
            if yaw is not None:                           # swing the camera round to keep the target in view
                d = (yaw - self.orbit_yaw + math.pi) % math.tau - math.pi
                self.orbit_yaw = (self.orbit_yaw + d * min(1.0, dt * 7.0)) % math.tau

    def key_zoom(self, keys, dt: float) -> None:
        """Ctrl + Up zooms in, Ctrl + Down zooms out, for as long as they're held:
        the orbit distance behind Rowan, or the field of view in fly mode."""
        if not (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]):
            return
        way = keys[pygame.K_UP] - keys[pygame.K_DOWN]      # Ctrl + Left/Right are kept free
        if not way:
            return
        k = math.exp(-1.1 * dt * way)                    # about 3x per second held
        if self.play:
            self.orbit_dist = min(7.0, max(1.8, self.orbit_dist * k))
        else:
            self.camera.fov_deg = min(100.0, max(10.0, self.camera.fov_deg * k))

    def toggle_lock(self) -> None:
        if not self.chars.rowan:
            return
        cam = self.camera
        f = cam.forward()
        self.lock.toggle(self.chars.chars, self.chars.rowan.a, cam.position.x, cam.position.z, f.x, f.z)

    def draw_reticle(self, viewproj, w: int, h: int, alpha: float) -> None:
        """The lock-on ring on the target's chest, and its name above it."""
        c = self.lock.target if self.play else None
        if c is None:
            return
        a = c.actor
        x = a.px + (a.x - a.px) * alpha
        y = a.py + (a.y - a.py) * alpha + c.t.height * c.size * 0.62
        z = a.pz + (a.z - a.pz) * alpha
        clip = viewproj * glm.vec4(x, y, z, 1.0)
        if clip.w <= 0.05:
            return
        sx = (clip.x / clip.w * 0.5 + 0.5) * w
        sy = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * h
        f = c.fighter
        if f is not None:
            self.reticle.render(w, h, sx, sy, c.t.name, f.hp / f.max_hp, f.trail / f.max_hp)
        else:
            self.reticle.render(w, h, sx, sy, c.t.name)

    # ------------------------------------------------------------------ loot and equipment
    def place_chests(self, placements: dict) -> None:
        """Chests are placed props with a "chest": {"id", "table", "name"} entry (WLD-17). One whose model
        isn't built yet is left out, like any other missing prop."""
        anchors = {p.name: p for p in self.world.landmarks + self.world.waystones}
        for group in placements.get("groups", []):
            a = anchors.get(group.get("anchor"))
            for it in group.get("items", []):
                c = it.get("chest")
                if a is None or c is None or it.get("id") not in self.props.catalog:
                    continue
                x, z = a.x + it["at"][0], a.z - it["at"][1]
                self.ground.drops.append(Drop(x, self.world.height_at(x, z), z, [], c.get("name", "Chest"),
                                              chest_id=c["id"], table=c["table"]))
        self.reset_chests()

    def reset_chests(self) -> None:
        """Chests opened in the current save stay open and empty; the rest are closed again."""
        for d in self.ground.drops:
            if d.is_chest:
                d.opened = d.chest_id in self.chars.progress.opened
                d.items = []

    def loot_in_reach(self) -> Drop | None:
        if not (self.play and self.chars.rowan) or self.pause.open:
            return None
        a = self.chars.rowan.a
        return self.ground.nearest(a.x, a.z)

    def take_loot(self) -> bool:
        """UIX-26: E takes everything from the drop or chest in reach. False if there's nothing in reach."""
        d = self.loot_in_reach()
        if d is None:
            return False
        prog = self.chars.progress
        was_closed = d.is_chest and not d.opened
        taken = self.ground.take(d, prog.inv, self.loot_tables, self.loot_rng)
        if was_closed:
            prog.opened.add(d.chest_id)
            if not taken and not d.items:
                self.hud.state.note(f"{d.name}: empty")
        for iid, n in taken:
            name = prog.inv.catalog[iid]["name"]
            self.hud.state.note(f"+{n} {name}")
        if d.items:
            self.hud.state.note("Backpack full")
        return True

    def waystone_in_reach(self):
        if not (self.play and self.chars.rowan) or self.pause.open or self.waystones.open:
            return None
        a = self.chars.rowan.a
        return self.waystones.in_reach(a.x, a.z)

    def use_waystone(self) -> bool:
        """WLD-09: E at a lit Waystone rests and saves, then offers travel. False if none is in reach."""
        stone = self.waystone_in_reach()
        if stone is None:
            return False
        if self.chars.chased(30.0):
            self.hud.state.note("Can't rest with enemies nearby")
            return True
        self.chars.rest()
        self.save()
        self.hud.state.note(f"Rested at {stone.name}")
        self.waystones.show(stone)
        self.set_capture(False)
        return True

    def fast_travel(self, dest) -> None:
        """Arrive beside another lit Waystone, facing it, and save there."""
        x, z, heading = travel_spot(self.world, dest)
        self.chars.teleport(x, z, heading)
        self.lock.target = None
        self.map.open = False
        self.orbit_yaw = math.pi - heading
        self.follow_camera(1.0)
        self.camera.prev_position = glm.vec3(self.camera.position)
        self.save()
        self.hud.state.note(f"Travelled to {dest.name}")
        self.fade = 0.9

    def drop_item(self, iid: str, n: int) -> None:
        """Put something from the backpack on the ground at Rowan's feet (the pause menu's X)."""
        x, z, heading = self.nav_position()
        x, z = x + math.sin(heading) * 0.8, z + math.cos(heading) * 0.8
        self.ground.spawn(x, self.world.height_at(x, z) + 0.1, z, [[iid, n]])
        self.hud.state.note(f"Dropped {self.chars.progress.inv.catalog[iid]['name']}")

    def apply_settings(self, field: str) -> None:
        """A setting changed in the pause menu: use it now and save it."""
        s = self.settings
        if field == "music_volume" and self.music is not None:
            self.music.volume = s.music_volume
        elif field == "sfx_volume" and self.sfx is not None:
            self.sfx.player.volume = s.sfx_volume
        elif field == "fov_deg":
            self.camera.fov_deg = s.fov_deg
        elif field in ("pixel_scale", "pixel_smooth"):
            self.pixel_buf = PixelBuffer(self.ctx, s.pixel_scale, s.pixel_smooth) if s.pixel_scale >= 2 else None
            self.hiz = HiZBuffer(self.ctx) if self.pixel_buf is not None else None
        s.save()

    def nav_position(self) -> tuple[float, float, float]:
        """Where the compass and map put you (x, z) and which way you face (heading, actor convention)."""
        if self.play and self.chars.rowan:
            a = self.chars.rowan.a
            return a.x, a.z, a.heading
        p = self.camera.position
        return p.x, p.z, math.pi - self.camera.yaw

    def music_region(self) -> str:
        """'town' near Brackenvale (with a margin so the edge doesn't flip back and forth), else 'wilds'."""
        t = self.music.cfg.get("town")
        if not t:
            return "wilds"
        an = next((p for p in self.world.landmarks + self.world.waystones if p.name == t["anchor"]), None)
        if an is None:
            return "wilds"
        if self.play and self.chars.rowan:
            x, z = self.chars.rowan.a.x, self.chars.rowan.a.z
        else:
            x, z = self.camera.position.x, self.camera.position.z
        d = math.hypot(x - an.x, z - an.z)
        in_town = d < (t["leave_m"] if (self.music.region or "").startswith("town") else t["enter_m"])
        if not in_town:
            return "wilds"
        times = self.music.cfg.get("town_times", {})          # the town's theme follows the hour

        def within(span) -> bool:
            a, b = span
            h = self.day.hour
            return a <= h < b if a < b else (h >= a or h < b)
        if "night" in times and within(times["night"]):
            return "town_night"
        if "day" in times and within(times["day"]):
            return "town_day"
        return "town"                                          # morning and sunset

    def ground_sound(self, x: float, z: float) -> str:
        """Footstep surface: 'water' in the shallows, 'dirt' on paths and lanes, 'grass' elsewhere."""
        w = self.world
        if w.water_depth(x, z) > 0.15:
            return "water"
        c, r = w._px(x, z)
        ci, ri = int(round(c)), int(round(r))
        on_path = 0 <= ci < w.size and 0 <= ri < w.size and w.road[ri, ci] >= 120
        return "dirt" if on_path else "grass"

    def fly(self, dt: float, keys) -> None:
        cam = self.camera
        cam.prev_position = glm.vec3(cam.position)
        fwd = (keys[pygame.K_w] - keys[pygame.K_s])
        side = (keys[pygame.K_d] - keys[pygame.K_a])
        up = (keys[pygame.K_e] - keys[pygame.K_q])
        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        slow = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        move = cam.forward() * fwd + cam.right() * side + glm.vec3(0, 1, 0) * up
        if glm.length(move) > 0:
            speed = self.base_speed * (5.0 if fast else 1.0) * (0.2 if slow else 1.0) * (TURBO if self.turbo else 1.0)
            cam.position = cam.position + glm.normalize(move) * speed * dt
        floor = self.world.height_at(cam.position.x, cam.position.z) + 0.6
        if cam.position.y < floor:
            cam.position.y = floor
        lim = self.world.half - 1.0
        cam.position.x = min(max(cam.position.x, -lim), lim)
        cam.position.z = min(max(cam.position.z, -lim), lim)

    # ---------------------------------------------------------------- render
    def render(self, alpha: float) -> None:
        w, h = pygame.display.get_window_size()
        if self.play:
            self.follow_camera(alpha)
        eye, _view, _proj, viewproj = self.camera.matrices(w / max(h, 1), alpha)
        env = default_environment(self.fog, self.day.hour, self.day)
        env["u_flat_shade"] = int(self.settings.flat_shade)
        wire = self.ctx.wireframe

        if self.pixel_buf is not None:
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
            self.pixel_buf.use(w, h)
        else:
            self.ctx.viewport = (0, 0, w, h)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        self.ctx.wireframe = False
        self.sky.render(eye, viewproj, env)
        self.ctx.wireframe = wire
        self.terrain.render(eye, viewproj, env, self.settings.lod_distances_m)
        self.props.render(eye, viewproj, env, hiz=self.hiz)
        self.grass.render(eye, viewproj, env, self.time_s)
        self.chars.render(eye, viewproj, env, alpha)
        self.ctx.wireframe = False
        self.water.render(eye, viewproj, env, self.time_s)

        if self.pixel_buf is not None:
            if self.hiz is not None:
                pw, ph = self.pixel_buf.size
                self.hiz.build(self.pixel_buf._depth, pw, ph)
            self.pixel_buf.blit(w, h)

        if self.overlay_mode < 2:
            ow, oh = self.overlay.size
            self.overlay.render(w, h, x=w - ow - 10, y=h - oh - 10)
        elif self.overlay_mode == 3:
            self.overlay.render(w, h, x=10, y=10)
        self.subtitles.render(w, h)
        self.draw_reticle(viewproj, w, h, alpha)
        rc = self.chars.rowan_char
        if self.play and rc is not None:
            self.hud.render(w, h, rc.fighter, self.chars.progress, self.orbit_yaw, (rc.actor.x, rc.actor.z),
                            self.time_s, self.chars.rite_cooldown)
            self.levels.render(w, h, self.chars.progress)
            from lycantrope.render.characters import DEATH_RELOAD_S
            self.death.render(w, h, self.chars.down_t, DEATH_RELOAD_S)
        x, z, heading = self.nav_position()
        found = self.chars.progress.discovered
        if not self.map.open:
            self.compass.render(w, h, self.camera.yaw, (x, z), found)
            self.region.render(w, h)
        if not self.map.open and not self.pause.open and not self.waystones.open:
            near = self.loot_in_reach()
            self.loot_view.render(w, h, viewproj, self.ground.drops, near, eye, self.time_s)
            stone = self.waystone_in_reach() if near is None else None
            if stone is not None:
                self.waystones.render_prompt(w, h, stone)
        self.map.render(w, h, (x, z), heading, found, self.region.tracker.current or "")
        if self.waystones.open:
            self.pause.dim.render(w, h, 0, 0, alpha=0.5, size=(w, h))
        self.waystones.render(w, h)
        self.pause.render(w, h)
        if self.fade > 0:
            self.pause.dim.render(w, h, 0, 0, alpha=min(1.0, self.fade / 0.6), size=(w, h))
        self.ctx.wireframe = wire

    def controls_lines(self) -> list[str]:
        return [
            "─── Play ───────────────────────────────────────",
            "WASD             move  (Shift sprint  ·  Ctrl walk)",
            "Left click       light attack  (chain ×3)",
            "Right mouse      block",
            "R                heavy attack",
            "Q                cast the Ward  (absorbs damage)",
            "Space            jump  (attack mid-air = plunge)",
            "C                roll  (press just before landing)",
            "E                pick up loot  /  rest & travel at a Waystone  /  flask",
            "G                apple",
            "I                inventory  (Esc: pause menu)",
            "L                level-up  (out of combat)",
            "M                world map  (wheel zoom · WASD pan · C centre)",
            "Tab / mid btn    lock-on  /  release",
            "Scroll wheel     camera distance",
            "─── Debug ──────────────────────────────────────",
            "H / K            hit reaction  /  fall dead",
            "T                turbo  ×4",
            "N                noon  ↔  midnight",
            "[ / ]            −1 / +1 hour",
            "Ctrl + ↑/↓       zoom in / out  (hold)",
            "F1               wireframe",
            "F2               fog toggle",
            "F3               overlay: full / minimal / off / controls",
            "F4               free-fly  /  back to Rowan",
            "F5               save  (play)  ·  reload shaders  (fly)",
            "F6               terrain debug view",
            "F7               helmet on/off",
            "F8               grass on/off",
            "F9               music on/off",
            "F10              sound effects on/off",
            "F11              NPC voices on/off",
            "F12              screenshot",
            "Esc              pause menu  (resume, save, quit)",
        ]

    def overlay_lines(self) -> list[str]:
        if self.overlay_mode == 3:
            return self.controls_lines()
        ft = self.frame_times[-60:] or [0.0]
        ms = 1000 * sum(ft) / len(ft)
        p = self.camera.position
        fps_str = f"{1000 / max(ms, 1e-3):5.0f} fps  {ms:4.1f} ms"
        if self.overlay_mode == 1:                    # minimal: fps + position only
            if self.play and self.chars.rowan:
                a = self.chars.rowan.a
                return [fps_str, f"x {a.x:7.1f}  z {a.z:7.1f}"]
            return [fps_str, f"x {p.x:7.1f}  z {p.z:7.1f}  y {p.y:5.1f}"]
        # full mode
        st = self.terrain.stats
        counts = self.chars.counts()
        lines = [f"LYCANTROPE   {fps_str}"
                 + f"   {self.day.label()}{' night' if self.day.is_night() else ''}"
                 + (f"   TURBO x{TURBO:g}" if self.turbo else "") + ("" if self.fog else "   fog off")
                 + self.sound_flags()]
        if self.play and self.chars.rowan:
            a = self.chars.rowan.a
            lines.append(f"ROWAN  x {a.x:7.1f}  z {a.z:7.1f}   {a.state:<7} {a.anim.clip:<7} {a.speed:4.1f} m/s"
                         f"   biome {self.world.biome_at(a.x, a.z)}")
            rc, pr = self.chars.rowan_char, self.chars.progress
            if rc is not None and rc.fighter is not None:
                f = rc.fighter
                lines.append(f"HP {f.hp:4.0f}/{f.max_hp:.0f}  ST {f.stamina:3.0f}/{f.max_stamina:.0f}  "
                             f"Lv {pr.level}  XP {pr.xp}/{pr.xp_needed()}  flask {pr.flask}")
        else:
            lines.append(f"FLY {self.base_speed:.0f} m/s   x {p.x:7.1f}  z {p.z:7.1f}  y {p.y:6.1f}   (north = -z)")
        lines += [
            f"characters {self.chars.drawn} drawn of {sum(counts.values())}   " + self.count_summary(counts),
            f"terrain {st.triangles / 1000:.0f}k tris   props {self.props.drawn}/{self.props.count} "
            f"({self.props.triangles / 1000:.0f}k tris)",
        ]
        return lines

    def count_summary(self, counts: dict[str, int]) -> str:
        """Short totals for the overlay: enemies, townsfolk and animals (the full list got too wide)."""
        groups = {"enemies": 0, "townsfolk": 0, "animals": 0}
        for kind, n in counts.items():
            t = self.chars.types.get(kind)
            if kind == "rowan":
                continue
            if t is not None and t.ai is not None and t.ai.aggro_m > 0:
                groups["enemies"] += n
            elif t is not None and t.skel is not None and "Hips" in t.skel.names:
                groups["townsfolk"] += n                     # people (biped rigs) who don't attack
            else:
                groups["animals"] += n
        return "   ".join(f"{k} {v}" for k, v in groups.items())

    def sound_flags(self) -> str:
        """What's switched off, for the overlay's top line."""
        off = []
        if self.music is not None:
            off += [name for layer, name in (("explore", "overworld music"), ("combat", "battle music"))
                    if not self.music.layer_on[layer]]
        if self.sfx is not None:
            off += [] if self.sfx.enabled else ["sfx"]
            off += [] if self.sfx.voices else ["voices"]
        return f"   off: {', '.join(off)}" if off else ""

    def screenshot(self, path) -> None:
        w, h = pygame.display.get_window_size()
        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
        Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM).save(path)
        print(f"screenshot saved: {path}")

    # ------------------------------------------------------------------ loop
    def run(self) -> int:
        print(f"world loaded in {self.load_seconds:.1f}s  ({self.world.size}² heightmap, "
              f"{self.terrain.stats.chunks_total} chunks)")
        frames = 0
        target = self.args.bench or (int(self.args.frames) if self.args.screenshot else 0)
        last = time.perf_counter()
        running = True
        while running:
            now = time.perf_counter()
            real_dt = now - last
            last = now
            # screenshots step the world exactly 1/60 s per frame so they're repeatable
            dt = 1 / 60 if self.args.screenshot else real_dt
            running = self.handle_events()
            for _ in range(self.clock.advance(dt)):
                self.update(self.clock.dt)
            self.latest_mouse()
            self.render(self.clock.alpha)
            pygame.display.flip()
            if self.settings.low_latency and not self.args.bench:
                self.ctx.finish()        # don't let the driver queue frames ahead: the camera answers sooner
            frames += 1
            if frames > 2:
                self.frame_times.append(real_dt)
                del self.frame_times[:-600]
            self.overlay_timer -= dt
            if self.overlay_timer <= 0:
                self.overlay.set_lines(self.overlay_lines())
                self.overlay_timer = 0.25
            if target and frames >= target:
                if self.args.screenshot:
                    self.overlay.set_lines(self.overlay_lines())
                    self.render(1.0)
                    self.screenshot(self.args.screenshot)
                if self.args.bench:
                    ft = np.array(self.frame_times) * 1000
                    print(f"bench: {len(ft)} frames  mean {ft.mean():.2f} ms  p95 {np.percentile(ft, 95):.2f} ms  "
                          f"tris {self.terrain.stats.triangles}  LOD {self.terrain.stats.per_lod}")
                running = False
        pygame.quit()
        return 0


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lycantrope", description="Lycantrope")
    ap.add_argument("--cam", help="free-fly start: X,Z,YAW,PITCH[,HEIGHT_ABOVE_GROUND]")
    ap.add_argument("--screenshot", help="save a PNG after a few frames and quit")
    ap.add_argument("--frames", default="5", help="frames to run before --screenshot (default 5)")
    ap.add_argument("--bench", type=int, default=0, help="render N frames, print timing, quit")
    ap.add_argument("--size", help="window size WxH")
    ap.add_argument("--clean", action="store_true", help="hide the debug overlay (for screenshots)")
    ap.add_argument("--time", type=float, help="start at this hour of the day (0-24), e.g. 21.5")
    ap.add_argument("--orbit", help="testing: camera around Rowan as YAW,PITCH,DIST (yaw 0 = behind him)")
    ap.add_argument("--act", help="testing: what Rowan does from the start: run, sprint, walk, light, "
                                  "combo, heavy, block, cast, hit, die, lock, fight, hurt, levelup; "
                                  "or 'teleport:X,Z[:action]' to start elsewhere")
    ap.add_argument("--slot", type=int, default=None, help="bypass title screen and load slot N (0-2)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings.load()

    save_slot = 0
    save_data = None

    # bypass the title screen for headless modes and explicit --slot
    headless = bool(args.bench or args.screenshot)
    if args.slot is not None:
        save_slot = args.slot
        save_data = save_sys.load(save_slot)
    elif not headless:
        pygame.display.init()
        pygame.font.init()
        action, save_slot = title_screen.run()
        if action == "quit":
            pygame.display.quit()
            return 0
        if action == "load":
            save_data = save_sys.load(save_slot)

    return App(settings, args, save_slot=save_slot, save_data=save_data).run()


if __name__ == "__main__":
    sys.exit(run())
