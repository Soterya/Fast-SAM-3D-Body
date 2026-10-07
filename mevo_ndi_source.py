"""Logitech Mevo NDI RGB source for this repo (self-contained).

Logic adapted from mesh-stream-server/temp.py but does not import that repo.
Frames are returned after a 90° counter-clockwise rotate (landscape -> portrait),
matching sample_data_logitec_mevo/intri.yml.

Requires:
  pip install cyndilib
  conda install -c conda-forge 'ffmpeg>=7'   # libavcodec.so.61 for NDI|HX
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_CAMERA_MATCH = "MEVO-2G9TP"
DEFAULT_INTRINSICS_PATH = Path(__file__).resolve().parent / "sample_data_logitec_mevo" / "intri.yml"
DEFAULT_FPS = 30.0


def preload_ndi_hx_ffmpeg() -> str:
    """Load FFmpeg 7 libavcodec before libndi dlopens the system FFmpeg 6.

    Mevo is NDI|HX. cyndilib's libndi dynamically loads libavcodec; with only
    Ubuntu's libavcodec.so.60 it can crash. Preloading conda-forge's .61 fixes receive.
    """
    env_lib = Path(sys.prefix) / "lib"
    soname = env_lib / "libavcodec.so.61"
    if not soname.exists():
        matches = sorted(glob.glob(str(env_lib / "libavcodec.so.61*")))
        if not matches:
            raise RuntimeError(
                "Mevo NDI|HX needs FFmpeg 7 (libavcodec.so.61). "
                f"Not found under {env_lib}. Install with:\n"
                "  conda install -c conda-forge 'ffmpeg>=7'"
            )
        soname = Path(matches[0])

    os.environ["LD_LIBRARY_PATH"] = (
        f"{env_lib}{os.pathsep}{os.environ['LD_LIBRARY_PATH']}"
        if os.environ.get("LD_LIBRARY_PATH")
        else str(env_lib)
    )

    ctypes.CDLL(str(soname), mode=ctypes.RTLD_GLOBAL)
    for name in (
        "libavutil.so.59",
        "libavformat.so.61",
        "libswscale.so.8",
        "libswresample.so.5",
    ):
        dep = env_lib / name
        if dep.exists():
            ctypes.CDLL(str(dep), mode=ctypes.RTLD_GLOBAL)
    return str(soname)


_AVCODEC = preload_ndi_hx_ffmpeg()

from cyndilib.finder import Finder  # noqa: E402
from cyndilib.receiver import Receiver  # noqa: E402
from cyndilib.video_frame import VideoFrameSync  # noqa: E402
from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat  # noqa: E402
from cyndilib.wrapper.ndi_structs import FourCC  # noqa: E402


def load_intrinsics_yml(
    path: str | Path,
    camera_name: str = "0",
) -> tuple[np.ndarray, np.ndarray]:
    """Load K (3x3) and dist coeffs from an EasyMocap-style OpenCV YAML.

    The K in sample_data_logitec_mevo/intri.yml is for the *rotated* portrait
    frame (after ROTATE_90_COUNTERCLOCKWISE).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open intrinsics YAML: {path}")
    try:
        k_node = fs.getNode(f"K_{camera_name}")
        d_node = fs.getNode(f"dist_{camera_name}")
        if k_node.empty():
            raise RuntimeError(f"Missing K_{camera_name} in {path}")
        k = np.asarray(k_node.mat(), dtype=np.float64).reshape(3, 3)
        if d_node.empty():
            dist = np.zeros((5,), dtype=np.float64)
        else:
            dist = np.asarray(d_node.mat(), dtype=np.float64).reshape(-1)
    finally:
        fs.release()
    return k, dist


def camera_params_from_k(k: np.ndarray) -> list[float]:
    """fx, fy, cx, cy for pupil_apriltags."""
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]


def project_points_pinhole(k: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points with a pinhole K -> Nx2 pixels."""
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-9, None)
    uv = (k @ pts.T).T
    uv = uv[:, :2] / z[:, None]
    return uv


def find_camera(finder: Finder, match: str, timeout_s: float = 30.0):
    deadline = time.time() + timeout_s
    needle = match.lower()
    while time.time() < deadline:
        finder.wait_for_sources(2)
        for source in finder:
            name = source.name or ""
            stream = source.stream_name or ""
            if needle in name.lower() or needle in stream.lower():
                return source
        time.sleep(0.25)
    names = finder.get_source_names()
    raise RuntimeError(
        f'Camera matching "{match}" not found within {timeout_s:.0f}s. '
        f"Available: {names or '(none)'}"
    )


def frame_to_bgr(vf: VideoFrameSync, rotate_90_ccw: bool = True) -> Optional[np.ndarray]:
    xres, yres = vf.get_resolution()
    if not xres or not yres:
        return None
    try:
        arr = vf.get_array()
    except Exception:
        return None
    if arr is None or arr.size == 0:
        return None

    fourcc = vf.get_fourcc()
    if arr.ndim == 1:
        if fourcc in (FourCC.BGRA, FourCC.BGRX, FourCC.RGBA, FourCC.RGBX):
            arr = arr.reshape((yres, xres, 4))
        elif fourcc in (FourCC.UYVY, FourCC.UYVA):
            arr = arr.reshape((yres, xres, 2))
        else:
            return None

    if fourcc in (FourCC.BGRA, FourCC.BGRX):
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif fourcc in (FourCC.RGBA, FourCC.RGBX):
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif fourcc == FourCC.UYVY:
        bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        bgr = arr
    else:
        return None

    if rotate_90_ccw:
        # 1920x1080 -> 1080x1920; matches sample_data_logitec_mevo/intri.yml
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


class MevoNDISource:
    """Live Mevo NDI RGB source that yields rotated BGR frames."""

    def __init__(
        self,
        camera_match: str = DEFAULT_CAMERA_MATCH,
        *,
        find_timeout_s: float = 30.0,
        connect_timeout_s: float = 6.0,
        first_frame_timeout_s: float = 10.0,
        rotate_90_ccw: bool = True,
        intrinsics_path: str | Path = DEFAULT_INTRINSICS_PATH,
        intrinsics_camera_name: str = "0",
    ):
        self.camera_match = camera_match
        self.find_timeout_s = float(find_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.first_frame_timeout_s = float(first_frame_timeout_s)
        self.rotate_90_ccw = bool(rotate_90_ccw)
        self.intrinsics_path = Path(intrinsics_path)
        self.intrinsics_camera_name = str(intrinsics_camera_name)

        self.k, self.dist = load_intrinsics_yml(
            self.intrinsics_path, self.intrinsics_camera_name
        )
        self.camera_params = camera_params_from_k(self.k)

        self._finder: Optional[Finder] = None
        self._receiver: Optional[Receiver] = None
        self._vf: Optional[VideoFrameSync] = None
        self._frame_sync = None
        self.source_name: Optional[str] = None
        self.width = 0
        self.height = 0
        self.fps = DEFAULT_FPS
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        print(f"Using {_AVCODEC} for NDI|HX decode", flush=True)
        print(f'Searching for "{self.camera_match}"...', flush=True)

        finder = Finder()
        finder.__enter__()
        try:
            source = find_camera(finder, self.camera_match, self.find_timeout_s)
            exact_name = source.name
            source = finder.get_source(exact_name)
            print(f"Found: {exact_name}", flush=True)

            receiver = Receiver(
                color_format=RecvColorFormat.BGRX_BGRA,
                bandwidth=RecvBandwidth.highest,
            )
            vf = VideoFrameSync()
            frame_sync = receiver.frame_sync
            frame_sync.set_video_frame(vf)
            receiver.set_source(source)

            deadline = time.time() + self.connect_timeout_s
            while time.time() < deadline:
                if receiver.is_connected():
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Timed out waiting for NDI connection.")

            print("Connected. Waiting for first frame...", flush=True)
            first: Optional[np.ndarray] = None
            deadline = time.time() + self.first_frame_timeout_s
            while time.time() < deadline:
                frame_sync.capture_video()
                first = frame_to_bgr(vf, rotate_90_ccw=self.rotate_90_ccw)
                if first is not None:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    "Connected but no video frames. Is NDI Mode still ON in the Mevo app?"
                )

            try:
                fps = float(vf.get_frame_rate())
                if fps <= 0:
                    fps = DEFAULT_FPS
            except Exception:
                fps = DEFAULT_FPS

            self._finder = finder
            self._receiver = receiver
            self._vf = vf
            self._frame_sync = frame_sync
            self.source_name = exact_name
            self.height, self.width = first.shape[:2]
            self.fps = fps
            self._started = True
            # Keep first frame for an immediate read if desired.
            self._pending_frame = first
            print(
                f"Mevo NDI RGB source: {self.width}x{self.height} @ {self.fps:.2f} fps "
                f"(rotate_90_ccw={self.rotate_90_ccw})",
                flush=True,
            )
            print(
                f"  K fx={self.k[0, 0]:.2f} fy={self.k[1, 1]:.2f} "
                f"cx={self.k[0, 2]:.2f} cy={self.k[1, 2]:.2f} "
                f"from {self.intrinsics_path}",
                flush=True,
            )
        except Exception:
            try:
                finder.__exit__(None, None, None)
            except Exception:
                pass
            raise

    def read(self) -> tuple[Optional[np.ndarray], float]:
        """Return (bgr_or_None, timestamp_s)."""
        if not self._started:
            raise RuntimeError("MevoNDISource.start() was not called")
        if getattr(self, "_pending_frame", None) is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame, time.time()

        assert self._receiver is not None and self._frame_sync is not None
        assert self._vf is not None
        if not self._receiver.is_connected():
            return None, time.time()
        self._frame_sync.capture_video()
        frame = frame_to_bgr(self._vf, rotate_90_ccw=self.rotate_90_ccw)
        return frame, time.time()

    def close(self) -> None:
        self._started = False
        self._pending_frame = None
        self._receiver = None
        self._vf = None
        self._frame_sync = None
        if self._finder is not None:
            try:
                self._finder.__exit__(None, None, None)
            except Exception:
                pass
            self._finder = None

    def __enter__(self) -> "MevoNDISource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
