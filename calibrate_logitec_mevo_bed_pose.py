#!/usr/bin/env python3
"""One-shot Logitech Mevo extrinsic calibration to the bed-center frame.

Captures a live Mevo NDI RGB frame (rotated 90° CCW to portrait), detects four
AprilTags (tag36h11) at the bed corners, estimates each tag pose from RGB +
intrinsics + tag size (no depth — Mevo has none), fits a Z-up bed frame, and
writes:

  sample_data_logitec_mevo/camera_poses_wrt_bed_center_YYYY-MM-DD_HH-MM-SS.json

Default tag layout (tag36h11):
  0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right

Intrinsics default to sample_data_logitec_mevo/intri.yml (rotated-frame K).

This script is self-contained in this repo. It does not import from
mesh-stream-server or data_processing.
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

from mevo_ndi_source import (
    DEFAULT_CAMERA_MATCH,
    DEFAULT_INTRINSICS_PATH,
    MevoNDISource,
    project_points_pinhole,
)

CORNER_ORDER = ("tl", "tr", "br", "bl")
APRILTAG_CORNER_INDEX = {"bl": 0, "br": 1, "tr": 2, "tl": 3}
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
DEFAULT_OUTPUT_DIR = "sample_data_logitec_mevo"


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray  # (4, 2) BL, BR, TR, TL
    pose_t: np.ndarray  # (3,) tag center in camera meters
    pose_r: np.ndarray  # (3, 3) tag->camera rotation
    decision_margin: float = 0.0
    hamming: int = 0

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


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
        cv2.namedWindow("__mevo_bed_pose_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__mevo_bed_pose_probe__")
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
    rgb_uv: dict[str, np.ndarray] = {}
    points_cam: dict[str, np.ndarray] = {}
    for bed_corner in CORNER_ORDER:
        det = assigned[bed_corner]
        if mode == "center":
            rgb_uv[bed_corner] = det.center.copy().astype(np.float64)
            points_cam[bed_corner] = det.pose_t.copy()
        elif mode == "bed-facing-marker-corner":
            marker_corner = marker_corner_map[bed_corner]
            rgb_uv[bed_corner] = det.corners[
                APRILTAG_CORNER_INDEX[marker_corner]
            ].astype(np.float64)
            local = APRILTAG_CORNER_LOCAL[marker_corner] * float(tag_size_m)
            points_cam[bed_corner] = det.pose_r @ local + det.pose_t
        else:
            raise ValueError(f"Unhandled corner mode: {mode}")
    return rgb_uv, points_cam


def fit_bed_frame_z_up(points_cam: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build RH bed frame: +X right, +Y top/head, +Z up (toward camera)."""
    ordered = np.stack([points_cam[k] for k in CORNER_ORDER], axis=0)
    center = ordered.mean(axis=0)

    centered = ordered - center[None, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Camera looks roughly +Z into the scene; bed up points toward the camera.
    if normal[2] > 0:
        normal = -normal

    projected = ordered - np.outer((ordered - center) @ normal, normal)
    tl, tr, br, bl = projected
    x_hint = 0.5 * ((tr - tl) + (br - bl))
    y_hint = 0.5 * ((tl - bl) + (tr - br))

    x_axis = x_hint - np.dot(x_hint, normal) * normal
    if np.linalg.norm(x_axis) < 1e-6:
        raise RuntimeError("Could not build bed X axis from the four tags")
    x_axis = x_axis / np.linalg.norm(x_axis)

    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    if np.dot(y_axis, y_hint) < 0:
        x_axis = -x_axis
        y_axis = -y_axis
    if np.dot(x_axis, x_hint) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    r_cam_bed = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(r_cam_bed) < 0:
        y_axis = -y_axis
        r_cam_bed = np.stack([x_axis, y_axis, z_axis], axis=1)

    r_bed_cam = r_cam_bed.T
    t_bed_cam = -r_bed_cam @ center

    plane_residuals = np.abs((ordered - center) @ normal)
    debug = {
        "bed_center_color": center,
        "corners_color": {k: points_cam[k] for k in CORNER_ORDER},
        "corners_projected_color": {
            k: projected[i] for i, k in enumerate(CORNER_ORDER)
        },
        "x_axis_color": x_axis,
        "y_axis_color": y_axis,
        "z_axis_color": z_axis,
        "plane_residual_m": plane_residuals,
        "plane_residual_rms_m": float(np.sqrt(np.mean(plane_residuals**2))),
    }
    return r_bed_cam, t_bed_cam, debug


def detection_roi_xy_from_outer_tag_corners(
    assigned: dict[str, MarkerDetection],
    scale: float = 1.0,
) -> list[list[float]]:
    """Bed quadrilateral through the outer corner of each AprilTag.

    Pose fitting uses tag centers. The ROI uses, on each tag, the detected
    corner farthest from the centroid of the four centers, so the polygon
    reaches the outer edge of the tags and covers the full bed surface.
    Order is tl, tr, br, bl. scale expands that polygon about its centroid.
    """
    centers = np.stack(
        [assigned[k].center.astype(np.float64) for k in CORNER_ORDER], axis=0
    )
    bed_uv = centers.mean(axis=0)
    outer = []
    for key in CORNER_ORDER:
        corners = assigned[key].corners.astype(np.float64)
        dist = np.linalg.norm(corners - bed_uv, axis=1)
        outer.append(corners[int(np.argmax(dist))])
    pts = np.stack(outer, axis=0)
    if abs(float(scale) - 1.0) > 1e-6:
        center = pts.mean(axis=0)
        pts = center + float(scale) * (pts - center)
    return [[float(x), float(y)] for x, y in pts]


def draw_detection_roi(
    image_bgr: np.ndarray,
    roi_xy: list[list[float]],
) -> np.ndarray:
    """Draw the saved detection ROI in magenta so it is visible on the debug JPEG."""
    poly = np.round(np.asarray(roi_xy, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [poly], isClosed=True, color=(255, 0, 255), thickness=6)
    anchor = tuple(poly.reshape(-1, 2)[0])
    cv2.putText(
        image_bgr,
        "DETECTION ROI",
        (anchor[0] + 12, max(36, anchor[1] - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return image_bgr


def pose_dict(r_bed_sensor: np.ndarray, t_bed_sensor: np.ndarray) -> dict:
    quat_xyzw = Rotation.from_matrix(r_bed_sensor).as_quat().astype(np.float64)
    return {
        "translation_m": t_bed_sensor.astype(np.float64).tolist(),
        "quaternion_xyzw": quat_xyzw.tolist(),
    }


def draw_rgb_overlay(
    image_bgr: np.ndarray,
    detections: list[MarkerDetection],
    assigned: dict[str, MarkerDetection],
    rgb_corners: dict[str, np.ndarray],
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
        det = assigned[label]
        cv2.drawMarker(out, pt, (0, 255, 255), cv2.MARKER_CROSS, 36, 3)
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
    ax.scatter([0.0], [0.0], [0.0], c="k", s=50, label="cam origin")
    ax.set_xlabel("X cam (m)")
    ax.set_ylabel("Y cam (m)")
    ax.set_zlabel("Z cam (m)")
    ax.set_title(
        f"Bed frame in Mevo camera coords\n"
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

    source = MevoNDISource(
        camera_match=args.ndi_match,
        find_timeout_s=args.find_timeout,
        rotate_90_ccw=not args.no_rotate,
        intrinsics_path=args.intrinsics_path,
        intrinsics_camera_name=args.intrinsics_name,
    )
    try:
        source.start()
    except Exception as exc:
        print(f"Failed to open Mevo NDI source: {exc}", file=sys.stderr)
        return 1

    marker_id_map = parse_marker_id_map(args.marker_id_map)
    marker_corner_map = parse_marker_corner_map(args.marker_corner_map)
    allowed_ids = set(marker_id_map.values())
    camera_params = source.camera_params
    k = source.k

    print(
        f"AprilTag family={args.tag_family} ids={sorted(allowed_ids)} "
        f"tag_size={args.tag_size}m quad_decimate={args.quad_decimate} "
        f"(RGB-only pose; no depth)",
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
            image_bgr, _ts = source.read()
            if image_bgr is None:
                print(
                    f"[{frame_idx}/{args.max_frames}] waiting for frame...",
                    flush=True,
                )
                continue

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
                rgb_corners, points_cam = bed_points_from_markers(
                    assigned,
                    args.corner_mode,
                    marker_corner_map,
                    float(args.tag_size),
                )
                r_bed_cam, t_bed_cam, frame_debug = fit_bed_frame_z_up(points_cam)
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
                    scale = min(1.0, 960 / max(preview.shape[0], 1))
                    shown = cv2.resize(
                        preview,
                        (int(preview.shape[1] * scale), int(preview.shape[0] * scale)),
                    )
                    if not try_imshow("Mevo bed pose calibration", shown):
                        show = False
                    elif cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue

            bed_center_uv = project_points_pinhole(
                k, frame_debug["bed_center_color"][None, :]
            )[0]
            rgb_overlay = draw_rgb_overlay(
                image_bgr,
                detections,
                assigned,
                rgb_corners,
                bed_center_uv,
            )
            axis_len = float(
                np.mean(
                    [
                        np.linalg.norm(points_cam[c] - frame_debug["bed_center_color"])
                        for c in CORNER_ORDER
                    ]
                )
            ) * 0.35
            for axis, color in (
                (frame_debug["x_axis_color"], (0, 0, 255)),
                (frame_debug["y_axis_color"], (0, 255, 0)),
                (frame_debug["z_axis_color"], (255, 0, 0)),
            ):
                end_uv = project_points_pinhole(
                    k, (frame_debug["bed_center_color"] + axis_len * axis)[None, :]
                )[0]
                cv2.arrowedLine(
                    rgb_overlay,
                    tuple(np.round(bed_center_uv).astype(int)),
                    tuple(np.round(end_uv).astype(int)),
                    color,
                    4,
                    tipLength=0.12,
                )

            detection_roi_xy = detection_roi_xy_from_outer_tag_corners(
                assigned, scale=float(args.roi_scale)
            )
            draw_detection_roi(rgb_overlay, detection_roi_xy)

            cv2.imwrite(str(debug_rgb_path), rgb_overlay)
            save_axes_plot(debug_axes_path, frame_debug, t_bed_cam)

            payload = {
                "captured_at": stamp,
                "source": "live_mevo_ndi_apriltag_pose_bed_pose",
                "ndi_match": args.ndi_match,
                "ndi_source_name": source.source_name,
                "frame_size_wh": [int(source.width), int(source.height)],
                "rotate_90_ccw": bool(not args.no_rotate),
                "intrinsics_path": str(Path(args.intrinsics_path)),
                "apriltag_family": args.tag_family,
                "apriltag_size_m": float(args.tag_size),
                "apriltag_corner_mode": args.corner_mode,
                "apriltag_id_map": {c: int(marker_id_map[c]) for c in CORNER_ORDER},
                "detection_roi_xy": detection_roi_xy,
                "detection_roi_order": list(CORNER_ORDER),
                "detection_roi_scale": float(args.roi_scale),
                "detection_roi_source": "outer_apriltag_corners",
                "frame_index": int(frame_idx),
                "color_camera_wrt_bed_center": pose_dict(r_bed_cam, t_bed_cam),
                "height_from_color_cam_center_to_floor_m": float(abs(t_bed_cam[2])),
                "calibration_debug": {
                    "plane_residual_rms_m": frame_debug["plane_residual_rms_m"],
                    "plane_residual_m_tl_tr_br_bl": frame_debug[
                        "plane_residual_m"
                    ].tolist(),
                    "corner_points_cam_m_tl_tr_br_bl": [
                        points_cam[c].tolist() for c in CORNER_ORDER
                    ],
                    "assigned_marker_ids": {
                        c: int(assigned[c].marker_id) for c in CORNER_ORDER
                    },
                    "camera_params_fx_fy_cx_cy": camera_params,
                    "notes": (
                        "Mevo RGB-only calibration (no depth). "
                        "Bed Z is up (plane normal toward the camera). "
                        "+X = right of bed, +Y = top/head of bed. "
                        "Recorder applies p_bed = R @ p_cam + t. "
                        "Frames / K are for the 90° CCW rotated portrait image. "
                        "Pose fitting uses AprilTag centers. "
                        "detection_roi_xy is the quadrilateral through the outer "
                        "corner of each tag (tl, tr, br, bl) in that image, "
                        "for later 2D box filtering."
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
                    "height_from_color_cam_center_to_floor_m": payload[
                        "height_from_color_cam_center_to_floor_m"
                    ],
                    "detection_roi_xy": payload["detection_roi_xy"],
                    "detection_roi_order": payload["detection_roi_order"],
                    "detection_roi_scale": payload["detection_roi_scale"],
                    "detection_roi_source": payload["detection_roi_source"],
                    "frame_size_wh": payload["frame_size_wh"],
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
                f"cam height (bed Z) = {t_bed_cam[2]:.3f} m"
            )
            print(
                f"  detection ROI outer corners (tl,tr,br,bl, scale={args.roi_scale:.3f}): "
                + ", ".join(
                    f"({p[0]:.1f},{p[1]:.1f})" for p in detection_roi_xy
                )
            )
            if args.also_latest:
                print(f"  latest    : {latest}")
            return 0

        print(
            f"Could not calibrate within {args.max_frames} frame(s). "
            f"Last error: {last_error}",
            file=sys.stderr,
        )
        return 1
    finally:
        source.close()
        if show:
            try_destroy_windows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live Mevo NDI AprilTag (tag36h11) calibration that writes "
            "sample_data_logitec_mevo/camera_poses_wrt_bed_center_<timestamp>.json"
        )
    )
    parser.add_argument("--ndi-match", default=DEFAULT_CAMERA_MATCH)
    parser.add_argument("--find-timeout", type=float, default=30.0)
    parser.add_argument(
        "--intrinsics-path",
        default=str(DEFAULT_INTRINSICS_PATH),
        help="OpenCV YAML with K_0 / dist_0 for the rotated portrait frame",
    )
    parser.add_argument("--intrinsics-name", default="0")
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable 90° CCW rotate (default is rotate to match intri.yml)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamped pose JSON and debug images",
    )
    parser.add_argument(
        "--also-latest",
        action="store_true",
        help="Also overwrite sample_data_logitec_mevo/camera_poses_wrt_bed_center.json",
    )
    parser.add_argument("--tag-family", default=DEFAULT_TAG_FAMILY)
    parser.add_argument(
        "--tag-size",
        type=float,
        default=DEFAULT_TAG_SIZE_M,
        help="Outer black square size in meters (default 0.22)",
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
    )
    parser.add_argument("--marker-corner-map", default=None)
    parser.add_argument("--min-decision-margin", type=float, default=0.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--quad-decimate", type=float, default=1.0)
    parser.add_argument(
        "--equalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the outer-tag-corner ROI about its centroid before saving "
            "detection_roi_xy. 1.0 is the outer corner of each AprilTag."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview while waiting for four tags",
    )
    args = parser.parse_args()
    if args.roi_scale <= 0:
        parser.error("--roi-scale must be > 0")
    return capture_and_calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
