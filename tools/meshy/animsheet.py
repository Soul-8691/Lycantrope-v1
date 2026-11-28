r"""Render preview strips of a rigged, animated GLB: a few frames of every clip, textured.

    .venv\Scripts\python.exe tools/meshy/animsheet.py assets/meshy/previews/HER-09/v1/rig/animations_1.glb [more.glb]
    .venv\Scripts\python.exe tools/meshy/animsheet.py HER-09                 # every rig/*.glb of that character's v1
    .venv\Scripts\python.exe tools/meshy/animsheet.py HER-09 --variant 2 --frames 8 --gif

Writes <folder>/sheets/<clip>.png (one row of frames, time left to right) and an
overview.png with every clip stacked. --gif also writes an animated <clip>.gif.
Skinning is done on the CPU with numpy; drawing uses an offscreen moderngl context.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import moderngl
import numpy as np
import pygltflib
from PIL import Image, ImageDraw
from pyglm import glm

ROOT = Path(__file__).resolve().parents[2]
PREVIEWS = ROOT / "assets" / "meshy" / "previews"
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NORM = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}

VERT = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
out vec3 v_nrm; out vec2 v_uv;
void main() { v_nrm = in_nrm; v_uv = in_uv; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
FRAG = """
#version 330
uniform sampler2D u_tex;
uniform int u_has_tex;
uniform vec3 u_eye_dir;
in vec3 v_nrm; in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_nrm);
    if (dot(n, u_eye_dir) < 0.0) n = -n;
    vec3 albedo = u_has_tex == 1 ? texture(u_tex, v_uv).rgb : vec3(0.75);
    vec3 key = normalize(vec3(0.4, 0.8, 0.6));
    float light = 0.45 + 0.55 * max(dot(n, key), 0.0) + 0.15 * max(dot(n, u_eye_dir), 0.0);
    f_color = vec4(albedo * light, 1.0);
}
"""


# ---------------------------------------------------------------------------- #
class Gltf:
    def __init__(self, path: Path) -> None:
        self.g = pygltflib.GLTF2().load(str(path))
        self.blob = self.g.binary_blob()

    def acc(self, i: int) -> np.ndarray:
        a = self.g.accessors[i]
        view = self.g.bufferViews[a.bufferView]
        comps = _COMPONENTS[a.type]
        dt = np.dtype(_DTYPES[a.componentType])
        start = (view.byteOffset or 0) + (a.byteOffset or 0)
        stride = view.byteStride or comps * dt.itemsize
        raw = np.frombuffer(self.blob, np.uint8, stride * (a.count - 1) + comps * dt.itemsize, start)
        rows = np.lib.stride_tricks.as_strided(raw, (a.count, comps * dt.itemsize), (stride, 1))
        out = np.ascontiguousarray(rows).view(dt).reshape(a.count, comps)
        if a.normalized and a.componentType in _NORM:
            out = out.astype(np.float32) / _NORM[a.componentType]
        return out

    def image(self, index: int) -> Image.Image:
        img = self.g.images[index]
        view = self.g.bufferViews[img.bufferView]
        data = self.blob[(view.byteOffset or 0):(view.byteOffset or 0) + view.byteLength]
        return Image.open(io.BytesIO(data)).convert("RGB")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


class Character:
    """One skinned mesh (all primitives merged) plus its node hierarchy and clips."""

    def __init__(self, path: Path) -> None:
        f = self.f = Gltf(path)
        g = f.g
        self.mesh_node = next(i for i, n in enumerate(g.nodes) if n.mesh is not None and n.skin is not None)
        node = g.nodes[self.mesh_node]
        pos, nrm, uv, joints, weights, idx, base = [], [], [], [], [], [], 0
        self.texture = None
        for prim in g.meshes[node.mesh].primitives:
            at = prim.attributes
            p = f.acc(at.POSITION).astype(np.float32)
            pos.append(p)
            nrm.append(f.acc(at.NORMAL).astype(np.float32) if at.NORMAL is not None else np.zeros_like(p))
            uv.append(f.acc(at.TEXCOORD_0).astype(np.float32) if at.TEXCOORD_0 is not None
                      else np.zeros((len(p), 2), np.float32))
            joints.append(f.acc(at.JOINTS_0).astype(np.int64))
            weights.append(f.acc(at.WEIGHTS_0).astype(np.float32))
            ix = f.acc(prim.indices).ravel() if prim.indices is not None else np.arange(len(p))
            idx.append(ix.astype(np.uint32) + base)
            base += len(p)
            if self.texture is None and prim.material is not None:
                pbr = g.materials[prim.material].pbrMetallicRoughness
                if pbr and pbr.baseColorTexture is not None:
                    self.texture = f.image(g.textures[pbr.baseColorTexture.index].source)
        self.pos, self.nrm, self.uv = np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv)
        self.joints, self.idx = np.concatenate(joints), np.concatenate(idx)
        w = np.concatenate(weights)
        self.weights = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

        skin = g.skins[node.skin]
        self.joint_nodes = list(skin.joints)
        self.ibm = f.acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)   # column-major
        self.parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
        self.rest = []
        for n in g.nodes:
            self.rest.append((np.array(n.translation or [0, 0, 0], float),
                              np.array(n.rotation or [0, 0, 0, 1], float),
                              np.array(n.scale or [1, 1, 1], float),
                              np.array(n.matrix, float).reshape(4, 4).T if n.matrix else None))

    def clips(self) -> list[tuple[str, float]]:
        out = []
        for a in self.f.g.animations:
            length = max(float(self.f.acc(s.input).max()) for s in a.samplers)
            out.append((a.name or "clip", length))
        return out

    def pose(self, clip: int | None, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Skinned positions and normals at time t of a clip (None = bind pose)."""
        g = self.f.g
        trs = [list(r) for r in self.rest]
        if clip is not None:
            anim = g.animations[clip]
            for ch in anim.channels:
                s = anim.samplers[ch.sampler]
                times = self.f.acc(s.input).ravel()
                vals = self.f.acc(s.output)
                if s.interpolation == "CUBICSPLINE":
                    vals = vals.reshape(len(times), 3, -1)[:, 1]
                k = int(np.searchsorted(times, t, side="right")) - 1
                k = min(max(k, 0), len(times) - 1)
                k1 = min(k + 1, len(times) - 1)
                u = 0.0 if k1 == k or s.interpolation == "STEP" else (t - times[k]) / (times[k1] - times[k])
                u = min(max(u, 0.0), 1.0)
                slot = {"translation": 0, "rotation": 1, "scale": 2}.get(ch.target.path)
                if slot is None:
                    continue
                v = slerp(vals[k], vals[k1], u) if slot == 1 else vals[k] + u * (vals[k1] - vals[k])
                trs[ch.target.node][slot] = np.asarray(v, float)
                trs[ch.target.node][3] = None
        world: dict[int, np.ndarray] = {}

        def world_of(i: int) -> np.ndarray:
            if i in world:
                return world[i]
            tr, rot, sc, mat = trs[i]
            if mat is None:
                mat = np.eye(4)
                mat[:3, :3] = quat_to_mat(rot) * sc
                mat[:3, 3] = tr
            m = world_of(self.parent[i]) @ mat if i in self.parent else mat
            world[i] = m
            return m

        inv_mesh = np.linalg.inv(world_of(self.mesh_node))
        jm = np.stack([inv_mesh @ world_of(j) @ self.ibm[k] for k, j in enumerate(self.joint_nodes)])
        skin = np.einsum("vi,vijk->vjk", self.weights, jm[self.joints])          # (V, 4, 4)
        p = np.einsum("vjk,vk->vj", skin[:, :3, :3], self.pos) + skin[:, :3, 3]
        n = np.einsum("vjk,vk->vj", skin[:, :3, :3], self.nrm)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        # bring into the mesh node's world space (usually identity)
        mw = world_of(self.mesh_node)
        p = p @ mw[:3, :3].T + mw[:3, 3]
        return p.astype(np.float32), n.astype(np.float32)


# ---------------------------------------------------------------------------- #
class Renderer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        ss = size * 2
        self.fbo = self.ctx.framebuffer(self.ctx.renderbuffer((ss, ss)), self.ctx.depth_renderbuffer((ss, ss)))
        self.tex = None

    def set_character(self, ch: Character) -> None:
        if self.tex is not None:
            self.tex.release()
            self.tex = None
        if ch.texture is not None:
            img = ch.texture
            self.tex = self.ctx.texture(img.size, 3, img.tobytes())
            self.tex.build_mipmaps()
            self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.ibo = self.ctx.buffer(ch.idx.tobytes())
        self.uv = self.ctx.buffer(np.ascontiguousarray(ch.uv).tobytes())

    def frame(self, pos: np.ndarray, nrm: np.ndarray, center: np.ndarray, radius: float) -> Image.Image:
        ctx = self.ctx
        vbo = ctx.buffer(np.hstack([pos, nrm]).astype("f4").tobytes())
        vao = ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_nrm"), (self.uv, "2f", "in_uv")],
                               index_buffer=self.ibo, index_element_size=4)
        eye_dir = glm.normalize(glm.vec3(0.55, 0.25, 1.0))           # front, a little right and above
        fov = 30.0
        dist = radius / np.sin(np.radians(fov / 2)) * 1.02
        c = glm.vec3(*center)
        view = glm.lookAt(c + eye_dir * dist, c, glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(fov), 1.0, dist * 0.05, dist * 4)
        self.prog["u_mvp"].write(proj * view)
        self.prog["u_eye_dir"].value = tuple(eye_dir)
        self.prog["u_has_tex"].value = 1 if self.tex else 0
        if self.tex:
            self.tex.use(0)
            self.prog["u_tex"].value = 0
        self.fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        self.fbo.clear(0.93, 0.93, 0.91, 1.0)
        vao.render()
        ss = self.size * 2
        img = Image.frombytes("RGB", (ss, ss), self.fbo.read(components=3))
        vao.release()
        vbo.release()
        return img.transpose(Image.FLIP_TOP_BOTTOM).resize((self.size, self.size), Image.LANCZOS)


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "clip"


def render_file(path: Path, r: Renderer, frames: int, gif: bool) -> list[Path]:
    ch = Character(path)
    r.set_character(ch)
    out_dir = path.parent / "sheets"
    out_dir.mkdir(exist_ok=True)
    written = []
    for ci, (name, length) in enumerate(ch.clips()):
        if length < 0.05:                       # Meshy's one-frame bind clip in rigged.glb
            continue
        n = frames
        times = [length * i / n for i in range(n)]       # loop clips: last frame = first, so skip it
        if gif:
            gif_times = list(np.arange(0.0, length, 1 / 15))
        poses = [ch.pose(ci, t) for t in times]

        # Clips with root motion (sprint, lunges) travel; the camera follows the character
        # sideways but keeps one height and zoom for the whole clip, so poses stay comparable.
        def framing(pose_list):
            allp = np.concatenate([p for p, _ in pose_list])
            y_lo, y_hi = allp[:, 1].min(), allp[:, 1].max()
            boxes = [(p.min(axis=0), p.max(axis=0)) for p, _ in pose_list]
            ext = np.max([hi - lo for lo, hi in boxes], axis=0)
            ext[1] = y_hi - y_lo
            radius = float(np.linalg.norm(ext)) / 2
            centers = [np.array([(lo[0] + hi[0]) / 2, (y_lo + y_hi) / 2, (lo[2] + hi[2]) / 2]) for lo, hi in boxes]
            return centers, radius

        centers, radius = framing(poses)
        imgs = [r.frame(p, nrm, c, radius) for (p, nrm), c in zip(poses, centers)]
        S = r.size
        strip = Image.new("RGB", (S * n, S + 22), (255, 255, 255))
        d = ImageDraw.Draw(strip)
        for i, (img, t) in enumerate(zip(imgs, times)):
            strip.paste(img, (i * S, 22))
            d.text((i * S + 6, 26), f"{t:.2f}s", fill=(90, 90, 90))
        if name == "retarget_clip" or name.startswith("Armature|"):   # Meshy's generic clip names
            name = path.stem
        label = f"{name}   ({length:.2f}s, {path.name})"
        d.text((6, 5), label, fill=(0, 0, 0))
        out = out_dir / f"{safe(name)}.png"
        strip.save(out)
        written.append(out)
        if gif:
            gposes = [ch.pose(ci, t) for t in gif_times]
            gcenters, gradius = framing(gposes)
            gframes = [r.frame(p, nrm, c, gradius) for (p, nrm), c in zip(gposes, gcenters)]
            gframes[0].save(out_dir / f"{safe(name)}.gif", save_all=True, append_images=gframes[1:],
                            duration=int(1000 / 15), loop=0)
        print(f"  {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Render animation preview strips")
    ap.add_argument("targets", nargs="+", help="GLB files, or character IDs such as HER-09")
    ap.add_argument("--variant", type=int, help="which preview (default: the newest one with a rig)")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--gif", action="store_true", help="also write animated GIFs")
    args = ap.parse_args()

    files: list[Path] = []
    for t in args.targets:
        p = Path(t)
        if p.suffix.lower() == ".glb":
            files.append(p.resolve())
        else:
            if args.variant:
                v = args.variant
            else:
                rigged = sorted((int(d.parent.name[1:]) for d in (PREVIEWS / t).glob("v*/rig")
                                 if d.parent.name[1:].isdigit()), reverse=True)
                if not rigged:
                    sys.exit(f"{t} has no rigged preview yet.")
                v = rigged[0]
                print(f"  {t}: using v{v} (newest rigged version; --variant picks another)")
            files += sorted((PREVIEWS / t / f"v{v}" / "rig").glob("*.glb"))
    if not files:
        sys.exit("No rigged GLBs found.")
    r = Renderer(args.size)
    sheets: list[Path] = []
    for f in files:
        sheets += render_file(f, r, args.frames, args.gif)
    if not sheets:
        sys.exit("Those files have no animation clips to draw (a rig on its own has only the bind pose).")
    for folder in sorted({s.parent for s in sheets}):
        # every strip in the folder, not just this run's, so the overview stays complete
        strips = sorted((p for p in folder.glob("*.png") if p.name != "overview.png"),
                        key=lambda p: p.stat().st_mtime)
        imgs = [Image.open(s) for s in strips]
        w = max(i.width for i in imgs)
        overview = Image.new("RGB", (w, sum(i.height for i in imgs)), (255, 255, 255))
        y = 0
        for i in imgs:
            overview.paste(i, (0, y))
            y += i.height
        out = folder / "overview.png"
        overview.save(out)
        print(f"  {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
