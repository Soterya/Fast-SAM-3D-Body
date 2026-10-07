#!/usr/bin/env python3
"""One-shot Azure Kinect extrinsic calibration to the bed-center frame.

Captures a live Kinect RGB frame (3072p by default), detects four AprilTags
(tag36h11) at the bed corners, estimates each tag pose from RGB + intrinsics +
tag size (no depth), fits a Z-up bed frame, and writes:

  sample_data/camera_poses_wrt_bed_center_YYYY-MM-DD_HH-MM-SS.json

in the same schema consumed by azure_kinect_mhr_recorder.sh
(color_camera_wrt_bed_center / depth_camera_wrt_bed_center).

Default tag layout (tag36h11):
  0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right

Tag size default (0.22 m) matches data_processing/scripts/detect_apriltag.py
(outer black square).

This script is self-contained. It does not import from the data_processing repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from pupil_apriltags import Detector
except ImportError as exc:
    raise SystemExit(
        "Missing pupil_apriltags. Install with:\n"
        "  pip install pupil-apriltags"
    ) from exc

try:
    from pyk4a import (
        CalibrationType,
        ColorResolution,
        Config,
        DepthMode,
        FPS,
        ImageFormat,
        PyK4A,
    )
except ImportError as exc:
    raise SystemExit(
        "Azure Kinect support requires pyk4a and the Azure Kinect SDK."
    ) from exc


CORNER_ORDER = ("tl", "tr", "br", "bl")
# pupil_apriltags / AprilTag C order: BL, BR, TR, TL (CCW from bottom-left).
APRILTAG_CORNER_INDEX = {"bl": 0, "br": 1, "tr": 2, "tl": 3}
# Tag-local offsets (meters) for the four corners at z=0, matching AprilTag order.
APRILTAG_CORNER_LOCAL = {
    "bl": np.asarray([-0.5, -0.5, 0.0], dtype=np.float64),
    "br": np.asarray([0.5, -0.5, 0.0], dtype=np.float64),
    "tr": np.asarray([0.5, 0.5, 0.0], dtype=np.float64),
    "tl": np.asarray([-0.5, 0.5, 0.0], dtype=np.float64),
}
BED_FACING_MARKER_CORNERS = {"tl": "br", "tr": "bl", "br": "tl", "bl": "tr"}
DEFAULT_TAG_ID_MAP = {"tl": 0, "tr": 1, "bl": 2, "br": 3}
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.22


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # (4, 2) BL, BR, TR, TL
    pose_t: np.ndarray  # (3,) tag center in color-camera meters
    pose_r: np.ndarray  # (3, 3) tag->color rotation
    decision_margin: float = 0.0
    hamming: int = 0

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def parse_color_resolution(value: str) -> ColorResolution:
    mapping = {
        "720p": ColorResolution.RES_720P,
        "1080p": ColorResolution.RES_1080P,
        "1440p": ColorResolution.RES_1440P,
        "1536p": ColorResolution.RES_1536P,
        "2160p": ColorResolution.RES_2160P,
        "3072p": ColorResolution.RES_3072P,
    }
    key = value.lower().replace("_", "")
    if key not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unsupported color resolution {value!r}. Choose: {', '.join(mapping)}"
        )
    return mapping[key]


def parse_depth_mode(value: str) -> DepthMode:
    modes = {
        "nfov-unbinned": DepthMode.NFOV_UNBINNED,
        "nfov-2x2": DepthMode.NFOV_2X2BINNED,
        "wfov-unbinned": DepthMode.WFOV_UNBINNED,
        "wfov-2x2": DepthMode.WFOV_2X2BINNED,
    }
    try:
        return modes[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Use one of: {', '.join(sorted(modes))}"
        ) from exc


def parse_marker_id_map(value: Optional[str]) -> dict[str, int]:
    if not value:
        return DEFAULT_TAG_ID_MAP.copy()
    out: dict[str, int] = {}
    for raw in value.replace(",", " ").split():
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                "Marker map entries must look like tl=0,tr=1,br=3,bl=2"
            )
        key, raw_id = raw.split("=", 1)
        key = key.strip().lower()
        if key not in CORNER_ORDER:
            raise argparse.ArgumentTypeError(f"Unknown bed corner {key!r}")
        out[key] = int(raw_id)
    missing = [key for key in CORNER_ORDER if key not in out]
    if missing:
        raise argparse.ArgumentTypeError(
            f"--marker-id-map must include all four corners; missing {missing}"
        )
    return out


def parse_marker_corner_map(value: Optional[str]) -> dict[str, str]:
    if not value:
        return BED_FACING_MARKER_CORNERS.copy()
    out: dict[str, str] = {}
    for raw in value.replace(",", " ").split():
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                "Corner map entries must look like tl=br,tr=bl,br=tl,bl=tr"
            )
        bed_corner, marker_corner = raw.split("=", 1)
        bed_corner = bed_corner.strip().lower()
        marker_corner = marker_corner.strip().lower()
        if bed_corner not in CORNER_ORDER:
            raise argparse.ArgumentTypeError(f"Unknown bed corner {bed_corner!r}")
        if marker_corner not in APRILTAG_CORNER_INDEX:
            raise argparse.ArgumentTypeError(f"Unknown marker corner {marker_corner!r}")
        out[bed_corner] = marker_corner
    missing = [key for key in CORNER_ORDER if key not in out]
    if missing:
        raise argparse.ArgumentTypeError(f"--marker-corner-map missing {missing}")
    return out


def gui_available() -> bool:
    try:
        cv2.namedWindow("__kinect_bed_pose_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__kinect_bed_pose_probe__")
        return True
    except Exception:
        return False


def try_imshow(window: str, image: np.ndarray) -> bool:
    try:
        cv2.imshow(window, image)
        return True
    except Exception:
        return False


def try_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def color_camera_params(calibration) -> list[float]:
    """fx, fy, cx, cy for pupil_apriltags pose estimation."""
    k = np.asarray(calibration.get_camera_matrix(CalibrationType.COLOR), dtype=np.float64)
    return [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]


def detect_markers(
    image_bgr: np.ndarray,
    detector: Detector,
    *,
    equalize: bool,
    min_decision_margin: float,
    max_hamming: int,
    allowed_ids: set[int],
    camera_params: list[float],
    tag_size_m: float,
) -> list[MarkerDetection]:
    """Detect AprilTags at full image resolution and estimate tag poses."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if equalize:
        gray = cv2.equalizeHist(gray)

    raw = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=float(tag_size_m),
    )
    best: dict[int, MarkerDetection] = {}
    for det in raw:
        tag_id = int(det.tag_id)
        if tag_id not in allowed_ids:
            continue
        if int(det.hamming) > max_hamming:
            continue
        if float(det.decision_margin) < min_decision_margin:
            continue
        if det.pose_t is None or det.pose_R is None:
            continue
        candidate = MarkerDetection(
            marker_id=tag_id,
            corners=np.asarray(det.corners, dtype=np.float32).reshape(4, 2),
            pose_t=np.asarray(det.pose_t, dtype=np.float64).reshape(3),
            pose_r=np.asarray(det.pose_R, dtype=np.float64).reshape(3, 3),
            decision_margin=float(det.decision_margin),
            hamming=int(det.hamming),
        )
        prev = best.get(tag_id)
        if prev is None or candidate.decision_margin > prev.decision_margin:
            best[tag_id] = candidate
    return list(best.values())


def assign_markers_to_bed_corners(
    detections: list[MarkerDetection],
    marker_id_map: dict[str, int],
) -> dict[str, MarkerDetection]:
    by_id = {det.marker_id: det for det in detections}
    missing = [
        f"{key}={marker_id}"
        for key, marker_id in marker_id_map.items()
        if marker_id not in by_id
    ]
    if missing:
        raise RuntimeError(f"Missing required AprilTag(s): {', '.join(missing)}")
    return {key: by_id[marker_id_map[key]] for key in CORNER_ORDER}


def bed_points_from_markers(
    assigned: dict[str, MarkerDetection],
    mode: str,
    marker_corner_map: dict[str, str],
    tag_size_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (rgb_uv, points_color_m) for each bed corner label."""
    rgb_uv: dict[str, np.ndarray] = {}
    points_color: dict[str, np.ndarray] = {}
    for bed_corner in CORNER_ORDER:
        det = assigned[bed_corner]
        if mode == "center":
            rgb_uv[bed_corner] = det.center.copy().astype(np.float64)
            points_color[bed_corner] = det.pose_t.copy()
        elif mode == "bed-facing-marker-corner":
            marker_corner = marker_corner_map[bed_corner]
            rgb_uv[bed_corner] = det.corners[
                APRILTAG_CORNER_INDEX[marker_corner]
            ].astype(np.float64)
            local = APRILTAG_CORNER_LOCAL[marker_corner] * float(tag_size_m)
            points_color[bed_corner] = det.pose_r @ local + det.pose_t
        else:
            raise ValueError(f"Unhandled corner mode: {mode}")
    return rgb_uv, points_color


def fit_bed_frame_z_up(points_color: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a right-handed bed frame with Z up from four tag points in color camera space.

    Convention (matches desired bed frame):
      +X toward the right of the bed (TL -> TR)
      +Y toward the top / head of the bed (BL -> TL)
      +Z up (plane normal toward the camera / ceiling)

    A 180° rotation about Z would flip both X and Y together; here we align
    explicitly with the right/top hints so the result is stable.

    Returns:
        R_bed_color: 3x3, maps color-camera vectors into the bed frame
        t_bed_color: 3, color-camera origin expressed in the bed frame
        debug: dict with axes / residuals for plots
    """
    ordered = np.stack([points_color[k] for k in CORNER_ORDER], axis=0)  # tl,tr,br,bl
    center = ordered.mean(axis=0)

    centered = ordered - center[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Overhead Kinect looks roughly along +Z into the scene. Bed "up" (toward the
    # ceiling / camera) should point toward the camera, i.e. against +Z_camera.
    if normal[2] > 0:
        normal = -normal

    projected = ordered - np.outer((ordered - center) @ normal, normal)

    tl, tr, br, bl = projected
    x_hint = 0.5 * ((tr - tl) + (br - bl))  # toward right of bed
    y_hint = 0.5 * ((tl - bl) + (tr - br))  # toward top / head of bed

    x_axis = x_hint - np.dot(x_hint, normal) * normal
    if np.linalg.norm(x_axis) < 1e-6:
        raise RuntimeError("Could not build bed X axis from the four tags")
    x_axis = x_axis / np.linalg.norm(x_axis)

    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    # Prefer +Y toward the top of the bed. Flipping X and Y together is a 180°
    # rotation about Z and preserves a right-handed frame with Z up.
    if np.dot(y_axis, y_hint) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    # Prefer +X toward the right (should already match after the Y check).
    if np.dot(x_axis, x_hint) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    r_color_bed = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(r_color_bed) < 0:
        y_axis = -y_axis
        r_color_bed = np.stack([x_axis, y_axis, z_axis], axis=1)

    r_bed_color = r_color_bed.T
    t_bed_color = -r_bed_color @ center

    plane_residuals = np.abs((ordered - center) @ normal)
    debug = {
        "bed_center_color": center,
        "corners_color": {k: points_color[k] for k in CORNER_ORDER},
        "corners_projected_color": {
            k: projected[i] for i, k in enumerate(CORNER_ORDER)
        },
        "x_axis_color": x_axis,
        "y_axis_color": y_axis,
        "z_axis_color": z_axis,
        "plane_residual_m": plane_residuals,
        "plane_residual_rms_m": float(np.sqrt(np.mean(plane_residuals**2))),
    }
    return r_bed_color, t_bed_color, debug


def pose_dict(r_bed_sensor: np.ndarray, t_bed_sensor: np.ndarray) -> dict:
    quat_xyzw = Rotation.from_matrix(r_bed_sensor).as_quat().astype(np.float64)
    return {
        "translation_m": t_bed_sensor.astype(np.float64).tolist(),
        "quaternion_xyzw": quat_xyzw.tolist(),
    }


def depth_pose_from_color(
    calibration,
    r_bed_color: np.ndarray,
    t_bed_color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose color->bed with color->depth extrinsics into depth->bed."""
    r_depth_color, t_depth_color = calibration.get_extrinsic_parameters(
        CalibrationType.COLOR, CalibrationType.DEPTH
    )
    r_depth_color = np.asarray(r_depth_color, dtype=np.float64).reshape(3, 3)
    t_depth_color = np.asarray(t_depth_color, dtype=np.float64).reshape(3)
    if abs(float(np.linalg.det(r_depth_color))) < 1e-6:
        raise RuntimeError(
            "Kinect color↔depth extrinsics are invalid (zero/singular rotation). "
            "Start the device with a real depth mode (not OFF) so factory "
            "extrinsics are available — depth images are still unused for tags."
        )
    r_bed_depth = r_bed_color @ r_depth_color.T
    t_bed_depth = t_bed_color - r_bed_depth @ t_depth_color
    return r_bed_depth, t_bed_depth


def draw_rgb_overlay(
    image_bgr: np.ndarray,
    detections: list[MarkerDetection],
    assigned: dict[str, MarkerDetection],
    rgb_corners: dict[str, np.ndarray],
    sample_pixels: dict[str, np.ndarray],
    bed_center_uv: Optional[np.ndarray],
) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        corners = np.round(det.corners).astype(np.int32)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(out, p1, p2, (0, 255, 255), 3)
        cxy = tuple(np.round(det.center).astype(int))
        cv2.circle(out, cxy, 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"id={det.marker_id}",
            (cxy[0] + 10, cxy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    poly = np.asarray(
        [rgb_corners[k] for k in CORNER_ORDER], dtype=np.int32
    ).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], isClosed=True, color=(0, 255, 0), thickness=4)
    for label in CORNER_ORDER:
        pt = tuple(np.round(rgb_corners[label]).astype(int))
        sample = tuple(np.round(sample_pixels[label]).astype(int))
        det = assigned[label]
        cv2.drawMarker(out, pt, (0, 255, 255), cv2.MARKER_CROSS, 36, 3)
        cv2.circle(out, sample, 8, (255, 128, 0), 2)
        cv2.putText(
            out,
            f"{label.upper()} id={det.marker_id}",
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    if bed_center_uv is not None:
        cxy = tuple(np.round(bed_center_uv).astype(int))
        cv2.drawMarker(out, cxy, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 3)
        cv2.putText(
            out,
            "BED CENTER",
            (cxy[0] + 14, cxy[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return out


def save_axes_plot(path: Path, frame_debug: dict, camera_t_bed: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    center = frame_debug["bed_center_color"]
    x_axis = frame_debug["x_axis_color"]
    y_axis = frame_debug["y_axis_color"]
    z_axis = frame_debug["z_axis_color"]
    corners = np.stack([frame_debug["corners_color"][k] for k in CORNER_ORDER], axis=0)
    projected = np.stack(
        [frame_debug["corners_projected_color"][k] for k in CORNER_ORDER], axis=0
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c="tab:orange", s=60, label="corners")
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c="tab:green", s=40, label="projected")
    ax.scatter([center[0]], [center[1]], [center[2]], c="red", s=80, label="bed center")
    axis_len = float(np.mean(np.linalg.norm(corners - center, axis=1))) * 0.5
    for vec, color, name in (
        (x_axis, "r", "X"),
        (y_axis, "g", "Y"),
        (z_axis, "b", "Z up"),
    ):
        end = center + axis_len * vec
        ax.plot(
            [center[0], end[0]],
            [center[1], end[1]],
            [center[2], end[2]],
            color=color,
            linewidth=3,
            label=name,
        )
    ax.scatter([0.0], [0.0], [0.0], c="k", s=50, label="color cam origin")
    ax.set_xlabel("X color (m)")
    ax.set_ylabel("Y color (m)")
    ax.set_zlabel("Z color (m)")
    ax.set_title(
        f"Bed frame in color camera coords\n"
        f"plane RMS={frame_debug['plane_residual_rms_m']*1000:.1f} mm | "
        f"cam height in bed Z={camera_t_bed[2]:.3f} m"
    )
    ax.legend(loc="upper left")
    all_pts = np.vstack([corners, [[0, 0, 0]], center[None, :]])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins) + 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=140)
    plt.close(fig)


def project_point_color(calibration, point_color: np.ndarray) -> Optional[np.ndarray]:
    try:
        uv = calibration.convert_3d_to_2d(
            tuple(float(v) for v in point_color.reshape(3)),
            CalibrationType.COLOR,
            CalibrationType.COLOR,
        )
    except Exception:
        return None
    if uv is None:
        return None
    return np.asarray(uv, dtype=np.float64).reshape(2)


def capture_and_calibrate(args: argparse.Namespace) -> int:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = out_dir / f"camera_poses_wrt_bed_center_{stamp}.json"
    debug_rgb_path = out_dir / f"bed_pose_debug_rgb_{stamp}.jpg"
    debug_axes_path = out_dir / f"bed_pose_debug_axes_{stamp}.png"

    show = bool(args.show)
    if show and not gui_available():
        print(
            "OpenCV GUI is unavailable in this build; continuing without --show. "
            "Debug images will still be written to disk.",
            file=sys.stderr,
        )
        show = False

    # Depth mode must be ON for valid factory color↔depth extrinsics in the
    # pose JSON. Tag 3D still comes only from AprilTag pose (RGB), not depth.
    k4a = PyK4A(
        config=Config(
            color_resolution=args.color_resolution,
            depth_mode=DepthMode.WFOV_2X2BINNED,
            camera_fps={5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}[args.fps],
            color_format=ImageFormat.COLOR_BGRA32,
            synchronized_images_only=False,
        ),
        device_id=args.device_id,
    )
    print(
        f"Opening Kinect device {args.device_id} "
        f"(color={args.color_resolution_name}, depth=wfov-2x2 for extrinsics only, "
        f"{args.fps} fps)...",
        flush=True,
    )
    try:
        k4a.start()
    except Exception as exc:
        print(f"Failed to open Azure Kinect device {args.device_id}: {exc}", file=sys.stderr)
        return 1
    print("Kinect started.", flush=True)

    marker_id_map = parse_marker_id_map(args.marker_id_map)
    marker_corner_map = parse_marker_corner_map(args.marker_corner_map)
    allowed_ids = set(marker_id_map.values())
    camera_params = color_camera_params(k4a.calibration)
    print(
        f"AprilTag family={args.tag_family} ids={sorted(allowed_ids)} "
        f"tag_size={args.tag_size}m quad_decimate={args.quad_decimate} "
        f"(full-res detect; tag 3D from pose, not depth pixels)",
        flush=True,
    )
    print(
        "Note: first AprilTag detect on 3072p can take a while; progress prints after each frame.",
        flush=True,
    )
    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=float(args.quad_decimate),
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    last_error: Optional[BaseException] = None

    try:
        for frame_idx in range(1, args.max_frames + 1):
            capture = k4a.get_capture()
            if capture.color is None:
                print(
                    f"[{frame_idx}/{args.max_frames}] waiting for color...",
                    flush=True,
                )
                continue
            image_bgr = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            print(
                f"[{frame_idx}/{args.max_frames}] "
                f"frame {image_bgr.shape[1]}x{image_bgr.shape[0]}: "
                f"detecting AprilTags + estimating poses...",
                flush=True,
            )

            try:
                detections = detect_markers(
                    image_bgr,
                    detector,
                    equalize=args.equalize,
                    min_decision_margin=args.min_decision_margin,
                    max_hamming=args.max_hamming,
                    allowed_ids=allowed_ids,
                    camera_params=camera_params,
                    tag_size_m=float(args.tag_size),
                )
                found_ids = sorted(det.marker_id for det in detections)
                if len(detections) < 4:
                    raise RuntimeError(
                        f"Detected {len(detections)} AprilTag(s) {found_ids}; "
                        f"need 4 (ids {sorted(allowed_ids)})"
                    )
                assigned = assign_markers_to_bed_corners(detections, marker_id_map)
                rgb_corners, points_color = bed_points_from_markers(
                    assigned,
                    args.corner_mode,
                    marker_corner_map,
                    float(args.tag_size),
                )

                r_bed_color, t_bed_color, frame_debug = fit_bed_frame_z_up(points_color)
                r_bed_depth, t_bed_depth = depth_pose_from_color(
                    k4a.calibration, r_bed_color, t_bed_color
                )
            except Exception as exc:
                last_error = exc
                print(f"[{frame_idx}/{args.max_frames}] {exc}", flush=True)
                if show:
                    preview = image_bgr.copy()
                    cv2.putText(
                        preview,
                        str(exc)[:120],
                        (24, 54),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                    scale = min(1.0, 1280 / max(preview.shape[1], 1))
                    shown = cv2.resize(
                        preview,
                        (int(preview.shape[1] * scale), int(preview.shape[0] * scale)),
                    )
                    if not try_imshow("Kinect bed pose calibration", shown):
                        show = False
                    elif cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue

            bed_center_uv = project_point_color(
                k4a.calibration, frame_debug["bed_center_color"]
            )
            rgb_overlay = draw_rgb_overlay(
                image_bgr,
                detections,
                assigned,
                rgb_corners,
                rgb_corners,
                bed_center_uv,
            )
            if bed_center_uv is not None:
                axis_len = float(
                    np.mean(
                        [
                            np.linalg.norm(
                                points_color[k] - frame_debug["bed_center_color"]
                            )
                            for k in CORNER_ORDER
                        ]
                    )
                ) * 0.35
                for axis, color in (
                    (frame_debug["x_axis_color"], (0, 0, 255)),
                    (frame_debug["y_axis_color"], (0, 255, 0)),
                    (frame_debug["z_axis_color"], (255, 0, 0)),
                ):
                    end_uv = project_point_color(
                        k4a.calibration,
                        frame_debug["bed_center_color"] + axis_len * axis,
                    )
                    if end_uv is None:
                        continue
                    cv2.arrowedLine(
                        rgb_overlay,
                        tuple(np.round(bed_center_uv).astype(int)),
                        tuple(np.round(end_uv).astype(int)),
                        color,
                        4,
                        tipLength=0.12,
                    )

            cv2.imwrite(str(debug_rgb_path), rgb_overlay)
            save_axes_plot(debug_axes_path, frame_debug, t_bed_color)

            payload = {
                "captured_at": stamp,
                "source": "live_kinect_apriltag_pose_bed_pose",
                "device_id": int(args.device_id),
                "color_resolution": args.color_resolution_name,
                "depth_mode": "wfov-2x2",
                "kinect_fps": int(args.fps),
                "apriltag_family": args.tag_family,
                "apriltag_size_m": float(args.tag_size),
                "apriltag_corner_mode": args.corner_mode,
                "apriltag_id_map": {k: int(marker_id_map[k]) for k in CORNER_ORDER},
                "frame_index": int(frame_idx),
                "color_camera_wrt_bed_center": pose_dict(r_bed_color, t_bed_color),
                "depth_camera_wrt_bed_center": pose_dict(r_bed_depth, t_bed_depth),
                "height_from_depth_cam_center_to_floor_m": float(abs(t_bed_depth[2])),
                "calibration_debug": {
                    "plane_residual_rms_m": frame_debug["plane_residual_rms_m"],
                    "plane_residual_m_tl_tr_br_bl": frame_debug[
                        "plane_residual_m"
                    ].tolist(),
                    "corner_points_color_m_tl_tr_br_bl": [
                        points_color[k].tolist() for k in CORNER_ORDER
                    ],
                    "assigned_marker_ids": {
                        k: int(assigned[k].marker_id) for k in CORNER_ORDER
                    },
                    "camera_params_fx_fy_cx_cy": camera_params,
                    "notes": (
                        "Bed Z is up (plane normal toward the camera / ceiling). "
                        "+X = right of bed, +Y = top/head of bed. "
                        "Recorder applies p_bed = R @ p_color + t. "
                        "3D points from AprilTag pose (RGB + intrinsics + tag size); "
                        "depth pixels unused. depth_camera pose via factory extrinsics."
                    ),
                },
            }
            pose_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            if args.also_latest:
                latest = out_dir / "camera_poses_wrt_bed_center.json"
                latest_payload = {
                    "color_camera_wrt_bed_center": payload[
                        "color_camera_wrt_bed_center"
                    ],
                    "depth_camera_wrt_bed_center": payload[
                        "depth_camera_wrt_bed_center"
                    ],
                    "height_from_depth_cam_center_to_floor_m": payload[
                        "height_from_depth_cam_center_to_floor_m"
                    ],
                    "captured_at": stamp,
                    "source_file": pose_path.name,
                }
                latest.write_text(
                    json.dumps(latest_payload, indent=2) + "\n", encoding="utf-8"
                )

            print(f"Calibrated on frame {frame_idx}")
            print(f"  pose JSON : {pose_path}")
            print(f"  RGB debug : {debug_rgb_path}")
            print(f"  axes plot : {debug_axes_path}")
            print(
                f"  plane RMS : {frame_debug['plane_residual_rms_m']*1000:.2f} mm | "
                f"color cam height (bed Z) = {t_bed_color[2]:.3f} m"
            )
            if args.also_latest:
                print(f"  latest    : {out_dir / 'camera_poses_wrt_bed_center.json'}")
            print(
                "Point the recorder at this file with:\n"
                f"  CAMERA_POSE_PATH={pose_path} bash azure_kinect_mhr_recorder.sh"
            )
            return 0

        print(
            f"Could not calibrate within {args.max_frames} frame(s). "
            f"Last error: {last_error}",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            k4a.stop()
        except Exception:
            pass
        if show:
            try_destroy_windows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live Kinect AprilTag (tag36h11) calibration that writes "
            "sample_data/camera_poses_wrt_bed_center_<timestamp>.json"
        )
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--color-resolution",
        type=parse_color_resolution,
        default="3072p",
        help="Kinect color mode (default 3072p). Detection runs at full resolution.",
    )
    parser.add_argument("--fps", type=int, choices=[5, 15, 30], default=15)
    parser.add_argument(
        "--output-dir",
        default="sample_data",
        help="Directory for the timestamped pose JSON and debug images",
    )
    parser.add_argument(
        "--also-latest",
        action="store_true",
        help="Also overwrite sample_data/camera_poses_wrt_bed_center.json",
    )
    parser.add_argument("--tag-family", default=DEFAULT_TAG_FAMILY)
    parser.add_argument(
        "--tag-size",
        type=float,
        default=DEFAULT_TAG_SIZE_M,
        help="Outer black square size in meters (default 0.22 from prior script)",
    )
    parser.add_argument(
        "--marker-id-map",
        default=None,
        help="Default tl=0,tr=1,br=3,bl=2",
    )
    parser.add_argument(
        "--corner-mode",
        choices=["center", "bed-facing-marker-corner"],
        default="center",
        help=(
            "3D sample location per tag from AprilTag pose. Default 'center'. "
            "bed-facing-marker-corner uses the inner tag corner in 3D."
        ),
    )
    parser.add_argument("--marker-corner-map", default=None)
    parser.add_argument("--min-decision-margin", type=float, default=0.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument(
        "--quad-decimate",
        type=float,
        default=1.0,
        help="AprilTag detector quad_decimate (1.0 = full-res quads; slower)",
    )
    parser.add_argument(
        "--equalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Histogram-equalize grayscale before detection (helps blue tags)",
    )
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview while waiting for four tags (requires GUI OpenCV)",
    )
    args = parser.parse_args()
    # Keep the CLI name (e.g. 3072p); IntEnum str() is just the integer value.
    rev = {
        ColorResolution.RES_720P: "720p",
        ColorResolution.RES_1080P: "1080p",
        ColorResolution.RES_1440P: "1440p",
        ColorResolution.RES_1536P: "1536p",
        ColorResolution.RES_2160P: "2160p",
        ColorResolution.RES_3072P: "3072p",
    }
    args.color_resolution_name = rev.get(args.color_resolution, str(args.color_resolution))
    return capture_and_calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
