import argparse
import ctypes
import math
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

# If you need to force a backend, set PYOPENGL_PLATFORM before running:
#   PYOPENGL_PLATFORM=egl ...  (recommended on WSLg/Wayland)
#   PYOPENGL_PLATFORM=glx ...  (X11/GLX)
try:
    import glfw
    from OpenGL import GL
except ImportError as exc:
    raise SystemExit(
        "PyOpenGL and glfw are required for the viewer. Install them with `pip install PyOpenGL glfw`."
    ) from exc

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_renderer import GaussianModel, render_surfel
from scene import Scene
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import getProjectionMatrix


def _glfw_error_callback(code, desc) -> None:
    try:
        message = desc.decode("utf-8")
    except Exception:
        message = str(desc)
    print(f"[GLFW] ({code}) {message}")


VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 uv;
void main() {
    uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 uv;
out vec4 frag_color;
uniform sampler2D frame_tex;
void main() {
    frag_color = texture(frame_tex, uv);
}
"""


def _compile_shader(src: str, shader_type) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)
    success = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not success:
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed: {log}")
    return shader


def _create_program(vertex_src: str, fragment_src: str) -> int:
    if glfw.get_current_context() is None:
        raise RuntimeError("No current OpenGL context; call glfw.make_context_current(window) before creating shaders.")
    program = GL.glCreateProgram()
    vertex_shader = _compile_shader(vertex_src, GL.GL_VERTEX_SHADER)
    fragment_shader = _compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)
    success = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not success:
        log = GL.glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {log}")
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    return program


class GLTextureQuad:
    """Minimal textured quad to present frames in an OpenGL window."""

    def __init__(self, window) -> None:
        # Make sure the desired window context is current before touching GL.
        if glfw.get_current_context() != window:
            glfw.make_context_current(window)
        if glfw.get_current_context() is None:
            raise RuntimeError("OpenGL context is not current; ensure a GLFW window is created before the quad.")
        self.program = _create_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
        self.tex_id = GL.glGenTextures(1)
        self.tex_w = 0
        self.tex_h = 0
        self.format = GL.GL_RGB

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        vertices = np.array(
            [
                # x, y, u, v
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
        )

        stride = 4 * 4  # 4 floats per vertex
        GL.glVertexAttribPointer(
            0, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0)
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(8)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glUseProgram(self.program)
        tex_loc = GL.glGetUniformLocation(self.program, "frame_tex")
        GL.glUniform1i(tex_loc, 0)

    def update_texture(self, frame: np.ndarray) -> None:
        """Upload the rendered frame to the GPU texture."""
        h, w, channels = frame.shape
        if channels not in (3, 4):
            raise ValueError("Frame must be RGB or RGBA.")
        fmt = GL.GL_RGBA if channels == 4 else GL.GL_RGB

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        if w != self.tex_w or h != self.tex_h or fmt != self.format:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                fmt,
                w,
                h,
                0,
                fmt,
                GL.GL_UNSIGNED_BYTE,
                frame,
            )
            self.tex_w, self.tex_h, self.format = w, h, fmt
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, GL.GL_UNSIGNED_BYTE, frame
            )

    def draw(self) -> None:
        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex_id)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)


class FreeFlyCamera:
    """Simple free-fly camera with yaw/pitch look and WASD + QE movement."""

    def __init__(self, position, yaw, pitch, fov_y, znear=0.01, zfar=2000.0, base_speed=1.5, roll=0.0):
        self.pos = np.array(position, dtype=np.float32)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)
        self.fov_y = float(fov_y)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.base_speed = float(base_speed)

    def _basis(self):
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        # Forward points toward -Z when yaw=0,pitch=0 (standard view space)
        forward = np.array([sy * cp, sp, -cy * cp], dtype=np.float32)
        forward /= np.linalg.norm(forward) + 1e-8
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8
        if abs(self.roll) > 1e-6:
            cos_r = math.cos(self.roll)
            sin_r = math.sin(self.roll)
            right_rot = right * cos_r + up * sin_r
            up_rot = up * cos_r - right * sin_r
            right, up = right_rot, up_rot
        return right, up, forward

    def move(self, dir_right, dir_up, dir_forward, dt, speed_mult=1.0):
        right, up, forward = self._basis()
        vel = (right * dir_right + up * dir_up + forward * dir_forward)
        if np.linalg.norm(vel) > 1e-5:
            vel /= np.linalg.norm(vel)
        self.pos += vel * self.base_speed * speed_mult * dt

    def look(self, dx, dy, sensitivity=0.0025):
        self.yaw -= dx * sensitivity
        self.pitch = np.clip(self.pitch + dy * sensitivity, -math.pi / 2 + 1e-3, math.pi / 2 - 1e-3)

    def roll_by(self, delta):
        self.roll += delta

    def view_matrix(self) -> np.ndarray:
        right, up, forward = self._basis()
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.dot(view[:3, :3], self.pos)
        return view

    def make_camera(self, width: int, height: int) -> MiniCam:
        width = max(int(width), 1)
        height = max(int(height), 1)
        aspect = width / float(height)
        fovx = 2 * math.atan(math.tan(self.fov_y * 0.5) * aspect)
        view = self.view_matrix()
        world_view = torch.tensor(view, dtype=torch.float32, device="cuda").transpose(0, 1)
        proj = getProjectionMatrix(self.znear, self.zfar, fovx, self.fov_y).to(device="cuda").transpose(0, 1)
        full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        return MiniCam(width, height, self.fov_y, fovx, self.znear, self.zfar, world_view, full_proj)


class GaussianViewer:
    """OpenGL viewer that reuses the eval rendering pipeline."""

    def __init__(self, dataset, pipeline, opt, args) -> None:
        self.dataset = dataset
        self.pipeline = pipeline
        self.opt = opt
        self.args = args

        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=args.iteration, shuffle=False)

        iteration = args.iteration
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        self.iteration = iteration
        if args.indirect:
            self.opt.indirect = 1
            self.gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.camera_sets: Dict[str, Sequence] = {
            "train": self.scene.getTrainCameras(),
            "test": self.scene.getTestCameras(),
        }
        self.available_splits: List[str] = [
            name for name, cams in self.camera_sets.items() if len(cams)
        ]
        if not self.available_splits:
            raise RuntimeError("No cameras found in the loaded scene.")

        if args.split == "both" and self.available_splits:
            self.current_split = self.available_splits[0]
        elif args.split in self.available_splits:
            self.current_split = args.split
        else:
            self.current_split = self.available_splits[0]
        self.indices = {name: 0 for name in self.available_splits}

        first_cam = self._current_camera()
        window_w, window_h = self._scaled_window_size(
            int(first_cam.image_width),
            int(first_cam.image_height),
            args.max_window_size,
        )
        self.window = self._create_window(window_w, window_h, args.no_vsync)
        glfw.make_context_current(self.window)
        if glfw.get_current_context() is None:
            raise RuntimeError("Failed to obtain a current OpenGL context.")
        gl_version = GL.glGetString(GL.GL_VERSION)
        if not gl_version:
            raise RuntimeError(
                "OpenGL context creation failed (glGetString returned None). "
                "Ensure a display/GL driver is available (e.g., run inside a desktop session or enable WSLg/X11)."
            )
        self.quad = GLTextureQuad(self.window)

        self.freecam = self._make_freecam_from_cam(first_cam)
        self.use_freecam = not args.no_freecam
        self.mouse_look_active = False
        self.last_cursor = None
        self.key_state = set()
        self.last_time = time.time()

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.needs_render = True
        self.last_frame = None
        self._print_controls()

    def _print_controls(self) -> None:
        print("OpenGL viewer ready.")
        print("Controls: ←/→ to change dataset camera, Space to switch split, F to toggle free fly cam.")
        print("Freecam: hold Right Mouse to look, WASD move, Q/E down/up, R/T roll right/left, Shift boost, scroll speed. Esc/Q quit.")

    def _create_window(self, width: int, height: int, disable_vsync: bool):
        glfw.set_error_callback(_glfw_error_callback)
        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not has_display:
            raise RuntimeError(
                "No DISPLAY or WAYLAND_DISPLAY found. On WSL you need WSLg (Windows 11) "
                "or an X server like VcXsrv/Xming with DISPLAY exported, plus OpenGL drivers."
            )
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW (no DISPLAY/GL driver?).")
        # Match GLFW context creation to the PyOpenGL backend.
        pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "").lower()
        if pyopengl_platform == "egl" or "WSL_DISTRO_NAME" in os.environ:
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        else:
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(width, height, "Gaussian Viewer", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")
        glfw.make_context_current(window)
        glfw.swap_interval(0 if disable_vsync else 1)
        GL.glViewport(0, 0, width, height)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.07, 0.07, 0.08, 1.0)
        return window

    @staticmethod
    def _scaled_window_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
        if max_edge <= 0:
            return width, height
        scale = min(1.0, float(max_edge) / float(max(width, height)))
        return max(1, int(width * scale)), max(1, int(height * scale))

    def _current_camera(self):
        cams = self.camera_sets[self.current_split]
        return cams[self.indices[self.current_split]]

    def _current_view(self):
        if self.use_freecam:
            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            return self.freecam.make_camera(fb_w, fb_h)
        return self._current_camera()

    def _make_freecam_from_cam(self, dataset_cam):
        cam_center = dataset_cam.camera_center.detach().cpu().numpy()
        view_rm = dataset_cam.world_view_transform.T.detach().cpu().numpy()
        R_cw = view_rm[:3, :3]
        R_wc = R_cw.T
        forward = -R_wc[:, 2]
        norm_fwd = np.linalg.norm(forward)
        if norm_fwd < 1e-5:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / norm_fwd

        yaw = float(math.atan2(forward[0], forward[2] + 1e-8))
        pitch = float(math.asin(np.clip(forward[1], -0.999, 0.999)))
        extent = float(self.scene.cameras_extent if hasattr(self.scene, "cameras_extent") else 1.0)
        return FreeFlyCamera(cam_center, yaw, pitch, float(dataset_cam.FoVy), zfar=max(10.0, extent * 10.0), base_speed=max(0.5, extent * 0.5))

    def _update_freecam(self, dt: float) -> None:
        if not self.use_freecam:
            return
        dir_forward = 0.0
        dir_right = 0.0
        dir_up = 0.0
        if glfw.KEY_W in self.key_state:
            dir_forward -= 1.0
        if glfw.KEY_S in self.key_state:
            dir_forward += 1.0
        if glfw.KEY_D in self.key_state:
            dir_right += 1.0
        if glfw.KEY_A in self.key_state:
            dir_right -= 1.0
        if glfw.KEY_E in self.key_state:
            dir_up += 1.0
        if glfw.KEY_Q in self.key_state:
            dir_up -= 1.0
        roll_speed = 1.5  # rad/sec
        if glfw.KEY_R in self.key_state:
            self.freecam.roll_by(-roll_speed * dt)
            self.needs_render = True
        if glfw.KEY_T in self.key_state:
            self.freecam.roll_by(roll_speed * dt)
            self.needs_render = True

        speed_mult = 2.0 if glfw.KEY_LEFT_SHIFT in self.key_state or glfw.KEY_RIGHT_SHIFT in self.key_state else 1.0
        if dir_forward == dir_right == dir_up == 0.0:
            return
        self.freecam.move(dir_right, dir_up, dir_forward, dt, speed_mult=speed_mult)
        self.needs_render = True

    def _toggle_split(self) -> None:
        if len(self.available_splits) < 2:
            return
        idx = self.available_splits.index(self.current_split)
        idx = (idx + 1) % len(self.available_splits)
        self.current_split = self.available_splits[idx]
        self.needs_render = True

    def _move_camera(self, step: int) -> None:
        cams = self.camera_sets[self.current_split]
        if not cams:
            return
        self.indices[self.current_split] = (self.indices[self.current_split] + step) % len(cams)
        self.needs_render = True

    def _on_key(self, _window, key, _scancode, action, _mods) -> None:
        if action == glfw.PRESS:
            self.key_state.add(key)
        elif action == glfw.RELEASE and key in self.key_state:
            self.key_state.discard(key)

        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_RIGHT:
                self._move_camera(1)
            elif key == glfw.KEY_LEFT:
                self._move_camera(-1)
            elif key in (glfw.KEY_SPACE, glfw.KEY_TAB):
                self._toggle_split()
            elif key == glfw.KEY_R:
                self.needs_render = True
            elif key == glfw.KEY_F:
                self.use_freecam = not self.use_freecam
                if self.use_freecam:
                    self.freecam = self._make_freecam_from_cam(self._current_camera())
                self.needs_render = True

    def _on_resize(self, _window, width, height) -> None:
        GL.glViewport(0, 0, width, height)

    def _on_mouse_button(self, window, button, action, _mods) -> None:
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.mouse_look_active = True
                self.last_cursor = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.mouse_look_active = False
                self.last_cursor = None

    def _on_cursor(self, window, xpos, ypos) -> None:
        if not self.use_freecam or not self.mouse_look_active:
            return
        if self.last_cursor is None:
            self.last_cursor = (xpos, ypos)
            return
        dx = xpos - self.last_cursor[0]
        dy = ypos - self.last_cursor[1]
        self.last_cursor = (xpos, ypos)
        self.freecam.look(dx, dy)
        self.needs_render = True

    def _on_scroll(self, _window, _xoff, yoff) -> None:
        if not self.use_freecam:
            return
        # Adjust movement speed with scroll (coarse control)
        self.freecam.base_speed = float(max(0.05, self.freecam.base_speed * math.exp(0.1 * yoff)))

    def _render_current_camera(self) -> None:
        view = self._current_view()
        view.refl_mask = None
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            rendering = render_surfel(
                view, self.gaussians, self.pipeline, self.background, srgb=self.opt.srgb, opt=self.opt
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        frame = (
            render_color.permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frame_uint8 = np.flipud(frame_uint8)
        self.quad.update_texture(frame_uint8)
        self.last_frame = frame_uint8
        self.needs_render = False

        cams = self.camera_sets[self.current_split]
        if self.use_freecam:
            status = f"freecam | {frame_uint8.shape[1]}x{frame_uint8.shape[0]} | {1.0/elapsed:.1f} fps"
        else:
            idx = self.indices[self.current_split] + 1
            status = f"{self.current_split} {idx}/{len(cams)} | {frame_uint8.shape[1]}x{frame_uint8.shape[0]} | {1.0/elapsed:.1f} fps"
        glfw.set_window_title(self.window, f"Gaussian Viewer - {status}")

    def draw(self) -> None:
        if glfw.get_current_context() != self.window:
            glfw.make_context_current(self.window)
        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        self.last_time = now
        self._update_freecam(dt)
        if self.needs_render:
            self._render_current_camera()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self.quad.draw()
        glfw.swap_buffers(self.window)

    def run(self) -> None:
        try:
            while not glfw.window_should_close(self.window):
                self.draw()
                glfw.poll_events()
        finally:
            glfw.terminate()


def parse_args():
    parser = argparse.ArgumentParser(description="OpenGL GUI for Gaussian surfel renderer.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Checkpoint iteration to load; -1 picks the latest.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "both"],
        help="Which camera split to view. 'both' allows toggling.",
    )
    parser.add_argument(
        "--max_window_size",
        default=1600,
        type=int,
        help="Clamp the longest window edge to avoid oversized windows (<=0 disables scaling).",
    )
    parser.add_argument(
        "--no_freecam",
        action="store_true",
        help="Start in dataset camera mode; toggle free/orbit camera with the F key.",
    )
    parser.add_argument("--quiet", action="store_true", help="Silence torch/cuDNN init logs.")
    parser.add_argument("--no_vsync", action="store_true", help="Disable vsync to see raw rendering FPS.")
    return parser, model, opt, pipeline


def main() -> None:
    parser, model, opt, pipeline = parse_args()
    args = get_combined_args(parser)
    print("Rendering", args.model_path)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    opt_params = opt.extract(args)
    # opt_params.indirect = 1

    viewer = GaussianViewer(dataset, pipeline_params, opt_params, args)
    viewer.run()


if __name__ == "__main__":
    main()
