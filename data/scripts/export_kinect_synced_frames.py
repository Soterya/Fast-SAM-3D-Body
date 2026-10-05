#!/usr/bin/env python3
"""Export timestamp-synced Kinect color frames from a multi-camera MKV session.

Matching follows the device-timestamp nearest-neighbor logic used for Kinect
sync export: every kept master frame must have a frame on each other camera
within --max-delta-ms. Each kept sample is written as one full-resolution JPG
per camera, using the same export index so later packing stays frame-aligned.

Each camera also gets a calibration JSON with the color camera_matrix and a
gravity vector in the color-camera frame. Gravity is estimated from the MKV
IMU, then rotated from the accelerometer frame into the color camera.
"""

import argparse
import csv
import json
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyk4a import CalibrationType, PyK4APlayback, SeekOrigin

DATA_ROOT = Path("/home/rutwik/data")
RECORDINGS_ROOT = DATA_ROOT / "recordings"
INTERVALS_FILENAME = "data_capture_intervals.json"


@dataclass
class FeedInfo:
    name: str
    media_path: Path
    sidecar_path: Path
    device_timestamps_usec: list[int]
    save_timestamps_ns: list[int]
    frame_indices: list[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export synced Kinect color JPGs plus per-camera intrinsics and gravity."
        )
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Kinect session directory, or a directory that contains kinect*.mkv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory (default: output/kinect_synced/<session name>).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Feed names to export. Default: every discovered kinect*.mkv feed.",
    )
    parser.add_argument(
        "--reference-camera",
        type=str,
        default=None,
        help="Feed whose timestamps drive matching (default: kinect_master if present).",
    )
    parser.add_argument("--every-n", type=int, default=1, help="Keep every Nth reference frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on exported frames.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip reference frames before this frame index.",
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=20.0,
        help="Maximum nearest-neighbor device-timestamp delta.",
    )
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality [0-100].")
    parser.add_argument(
        "--imu-samples",
        type=int,
        default=100,
        help="Accelerometer samples used to estimate gravity (default: 100).",
    )
    parser.add_argument(
        "--gravity",
        nargs=3,
        type=float,
        default=None,
        metavar=("GX", "GY", "GZ"),
        help=(
            "Gravity direction in the color-camera frame, used when an MKV has no IMU track. "
            "Kinect color +Y is down, so an upright camera is 0 1 0."
        ),
    )
    parser.add_argument(
        "--intervals-json",
        type=Path,
        default=None,
        help="Override data_capture_intervals.json. Frames outside the intervals are dropped.",
    )
    parser.add_argument(
        "--no-interval-filter",
        action="store_true",
        help="Export all matched frames, ignoring data_capture_intervals.json.",
    )
    return parser.parse_args()


def resolve_session_dir(session_dir: Path) -> Path:
    candidate = session_dir.expanduser()
    if not candidate.is_absolute():
        local = (Path.cwd() / candidate).resolve()
        if local.exists():
            return local
    else:
        return candidate.resolve()

    parts = candidate.parts
    if parts and parts[0] == "data":
        return (DATA_ROOT / Path(*parts[1:])).resolve()
    if parts and parts[0] == "recordings":
        return (DATA_ROOT / candidate).resolve()
    if parts and parts[0].startswith("person_"):
        return (RECORDINGS_ROOT / candidate).resolve()
    return candidate.resolve()


def load_sidecar_csv(path: Path):
    device_timestamps_usec = []
    save_timestamps_ns = []
    frame_indices = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            color_present = int(row.get("color_present", "1"))
            color_ts = row.get("color_timestamp_usec", "")
            save_ts = row.get("save_timestamp_ns", "")
            if color_present != 1 or not color_ts or not save_ts:
                continue
            device_timestamps_usec.append(int(color_ts))
            save_timestamps_ns.append(int(save_ts))
            frame_indices.append(int(row["frame_idx"]))
    return device_timestamps_usec, save_timestamps_ns, frame_indices


def discover_feeds(session_dir: Path) -> list[FeedInfo]:
    candidate_roots = []
    if session_dir.name == "rgb_depth_data":
        candidate_roots.append(session_dir)
    for relative in (
        Path("data_collection") / "rgb_depth_data",
        Path("data_collection"),
        Path("rgb_depth_data"),
    ):
        root = session_dir / relative
        if root.exists():
            candidate_roots.append(root)
    candidate_roots.append(session_dir)

    seen_roots = set()
    search_roots = []
    for root in candidate_roots:
        resolved = root.resolve()
        if resolved not in seen_roots:
            search_roots.append(root)
            seen_roots.add(resolved)

    feeds = []
    seen_media = set()
    for root in search_roots:
        for media_path in sorted(root.glob("kinect*.mkv")):
            resolved_media = media_path.resolve()
            if resolved_media in seen_media:
                continue
            seen_media.add(resolved_media)
            sidecar_path = media_path.with_suffix(".save_timestamps.csv")
            if not sidecar_path.exists():
                continue
            device_ts, save_ts, frame_idx = load_sidecar_csv(sidecar_path)
            if not device_ts:
                continue
            feeds.append(
                FeedInfo(
                    name=media_path.stem,
                    media_path=media_path,
                    sidecar_path=sidecar_path,
                    device_timestamps_usec=device_ts,
                    save_timestamps_ns=save_ts,
                    frame_indices=frame_idx,
                )
            )
    return feeds


def find_intervals_json(anchor: Path) -> Path | None:
    path = anchor.expanduser().resolve()
    if path.is_file():
        roots = [path.parent, path.parent.parent, path.parent.parent.parent]
    else:
        roots = [path, path.parent, path.parent.parent]
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / INTERVALS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_capture_intervals(path: Path) -> list[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_intervals = payload.get("intervals") if isinstance(payload, dict) else payload
    if not isinstance(raw_intervals, list):
        raise RuntimeError(f"Intervals JSON must contain [start_ns, end_ns] pairs: {path}")
    intervals = []
    for idx, item in enumerate(raw_intervals):
        if not (isinstance(item, list) and len(item) == 2):
            raise RuntimeError(f"Invalid interval at index {idx} in {path}: {item!r}")
        start_ns = int(item[0])
        end_ns = int(item[1])
        if start_ns > end_ns:
            raise RuntimeError(f"Invalid interval at index {idx}; start > end: {item!r}")
        intervals.append((start_ns, end_ns))
    if not intervals:
        raise RuntimeError(f"No intervals found in {path}")
    return sorted(intervals)


def timestamp_in_intervals(timestamp_ns: int, intervals: list[tuple[int, int]]) -> bool:
    for start_ns, end_ns in intervals:
        if timestamp_ns < start_ns:
            return False
        if start_ns <= timestamp_ns <= end_ns:
            return True
    return False


def resolve_capture_intervals(session_dir: Path, intervals_json: Path | None, disabled: bool):
    if disabled:
        return None, None
    if intervals_json is not None:
        path = intervals_json.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Intervals JSON not found: {path}")
        return load_capture_intervals(path), path
    path = find_intervals_json(session_dir)
    if path is None:
        return None, None
    return load_capture_intervals(path), path


def find_nearest_index(target_usec: int, timestamps_usec: list[int]):
    pos = bisect_left(timestamps_usec, target_usec)
    best_idx = None
    best_delta = None
    for idx in (pos - 1, pos):
        if idx < 0 or idx >= len(timestamps_usec):
            continue
        delta = abs(timestamps_usec[idx] - target_usec)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx, best_delta


def build_matches(
    feeds: list[FeedInfo],
    reference_feed: FeedInfo,
    every_n: int,
    max_frames: int | None,
    start_frame: int,
    max_delta_ms: float,
    capture_intervals: list[tuple[int, int]] | None,
):
    max_delta_usec = int(max_delta_ms * 1000.0)
    matches = []
    stats = {
        "reference_candidates": 0,
        "sampled_candidates": 0,
        "kept_matches": 0,
        "dropped_no_nearest": 0,
        "dropped_delta_limit": 0,
        "dropped_outside_intervals": 0,
        "max_observed_delta_usec": 0,
        "max_observed_delta_feed": "",
    }
    ref_entries = list(
        zip(
            reference_feed.frame_indices,
            reference_feed.device_timestamps_usec,
            reference_feed.save_timestamps_ns,
        )
    )
    stats["reference_candidates"] = len(ref_entries)

    for list_index, (ref_frame_idx, ref_dev_ts_usec, ref_save_ts_ns) in enumerate(ref_entries):
        if ref_frame_idx < start_frame:
            continue
        if list_index % every_n != 0:
            continue
        stats["sampled_candidates"] += 1
        if capture_intervals is not None and not timestamp_in_intervals(
            ref_save_ts_ns, capture_intervals
        ):
            stats["dropped_outside_intervals"] += 1
            continue

        match = {
            "reference_device_timestamp_usec": ref_dev_ts_usec,
            "reference_save_timestamp_ns": ref_save_ts_ns,
            "frames": {},
        }
        valid = True
        for feed in feeds:
            nearest_idx, delta_usec = find_nearest_index(
                ref_dev_ts_usec, feed.device_timestamps_usec
            )
            if nearest_idx is None:
                stats["dropped_no_nearest"] += 1
                valid = False
                break
            if delta_usec is not None and delta_usec > max_delta_usec:
                stats["dropped_delta_limit"] += 1
                valid = False
                break
            if delta_usec is not None and delta_usec > stats["max_observed_delta_usec"]:
                stats["max_observed_delta_usec"] = delta_usec
                stats["max_observed_delta_feed"] = feed.name
            match["frames"][feed.name] = {
                "frame_idx": feed.frame_indices[nearest_idx],
                "device_timestamp_usec": feed.device_timestamps_usec[nearest_idx],
                "save_timestamp_ns": feed.save_timestamps_ns[nearest_idx],
                "delta_usec": delta_usec or 0,
            }
        if not valid:
            continue
        matches.append(match)
        stats["kept_matches"] += 1
        if max_frames is not None and len(matches) >= max_frames:
            break
    return matches, stats


def convert_kinect_to_bgr(color_format, color_image: np.ndarray):
    format_name = color_format.name
    if format_name == "COLOR_MJPG":
        return cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    if format_name == "COLOR_NV12":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_NV12)
    if format_name == "COLOR_YUY2":
        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)
    if format_name == "COLOR_BGRA32":
        return cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    return color_image


def fps_from_configuration(camera_fps) -> int:
    return int(str(camera_fps.name).split("_")[-1])


class KinectColorLoader:
    """Sequential MKV reader. Frame requests must be non-decreasing."""

    def __init__(self, playback: PyK4APlayback):
        self.playback = playback
        self.current_idx = -1
        self.current_color_bgr = None

    def seek_to_frame(self, frame_idx: int, device_timestamp_usec: int):
        self.playback.seek(device_timestamp_usec, SeekOrigin.DEVICE_TIME)
        self.current_idx = frame_idx - 1
        self.current_color_bgr = None

    def get_color(self, target_idx: int):
        if target_idx < self.current_idx:
            raise RuntimeError(
                f"Requested frame {target_idx} after already advancing to {self.current_idx}."
            )
        while self.current_idx < target_idx:
            capture = self.playback.get_next_capture()
            self.current_idx += 1
            if capture.color is not None:
                self.current_color_bgr = convert_kinect_to_bgr(
                    self.playback.configuration["color_format"], capture.color
                )
            else:
                self.current_color_bgr = None
        return self.current_color_bgr

    def close(self):
        self.playback.close()


def gravity_from_vector(gravity: list[float], media_path: Path) -> tuple[np.ndarray, float, int]:
    gravity_color = np.asarray(gravity, dtype=np.float64).reshape(3)
    gravity_norm = float(np.linalg.norm(gravity_color))
    if gravity_norm < 1e-12:
        raise RuntimeError("--gravity must be a non-zero vector")
    print(
        f"  {media_path.name}: no IMU track, using --gravity "
        f"[{gravity_color[0] / gravity_norm:+.3f}, "
        f"{gravity_color[1] / gravity_norm:+.3f}, "
        f"{gravity_color[2] / gravity_norm:+.3f}]"
    )
    return gravity_color / gravity_norm, gravity_norm, 0


def estimate_gravity_in_color_frame(
    playback: PyK4APlayback,
    imu_samples: int,
    fallback_gravity: list[float] | None,
) -> tuple[np.ndarray, float, int]:
    """Average the accelerometer and express gravity in the color camera frame.

    A resting accelerometer measures specific force, opposite gravity. The mean
    is negated, then rotated with the MKV accel-to-color extrinsics. Recordings
    with no IMU track use fallback_gravity instead.
    """
    media_path = playback.path
    if not playback.configuration["imu_track_enabled"]:
        if fallback_gravity is None:
            raise RuntimeError(
                f"IMU track is missing from {media_path}. "
                "Pass --gravity GX GY GZ in the color-camera frame. "
                "Kinect color +Y is down, so an upright camera is --gravity 0 1 0."
            )
        return gravity_from_vector(fallback_gravity, media_path)

    accels = []
    while len(accels) < imu_samples:
        try:
            sample = playback.get_next_imu_sample()
        except EOFError:
            break
        if sample is None:
            break
        accels.append(sample["acc_sample"])
    playback.seek(0, SeekOrigin.BEGIN)
    if len(accels) < 10:
        raise RuntimeError(
            f"Need at least 10 IMU samples to estimate gravity for {media_path}, got {len(accels)}"
        )

    mean_acc = np.mean(np.asarray(accels, dtype=np.float64), axis=0)
    gravity_imu = -mean_acc
    rotation, _translation = playback.calibration.get_extrinsic_parameters(
        CalibrationType.ACCEL, CalibrationType.COLOR
    )
    gravity_color = np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ gravity_imu
    gravity_norm = float(np.linalg.norm(gravity_color))
    if gravity_norm < 1e-6:
        raise RuntimeError(f"Invalid gravity magnitude for {media_path}")
    return gravity_color / gravity_norm, gravity_norm, len(accels)


def color_calibration(playback: PyK4APlayback) -> tuple[np.ndarray, np.ndarray]:
    calibration = playback.calibration
    camera_matrix = np.asarray(
        calibration.get_camera_matrix(CalibrationType.COLOR), dtype=np.float64
    )
    distortion = np.asarray(
        calibration.get_distortion_coefficients(CalibrationType.COLOR), dtype=np.float64
    )
    return camera_matrix, distortion


def write_calibration_json(
    path: Path,
    camera_name: str,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    gravity: np.ndarray,
    width: int,
    height: int,
    fps: int,
):
    payload = {
        "camera_name": camera_name,
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "gravity": gravity.reshape(3).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.every_n <= 0:
        raise ValueError("--every-n must be >= 1")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if not (0 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be in [0, 100]")
    if args.imu_samples < 10:
        raise ValueError("--imu-samples must be >= 10")
    if args.no_interval_filter and args.intervals_json is not None:
        raise ValueError("--intervals-json cannot be combined with --no-interval-filter")

    session_dir = resolve_session_dir(args.session_dir)
    if not session_dir.exists():
        raise RuntimeError(f"Session directory not found: {session_dir}")

    feeds = discover_feeds(session_dir)
    if not feeds:
        raise RuntimeError(f"No kinect*.mkv feeds with timestamp sidecars found in {session_dir}")
    feeds_by_name = {feed.name: feed for feed in feeds}
    if args.cameras:
        missing = [name for name in args.cameras if name not in feeds_by_name]
        if missing:
            known = ", ".join(feed.name for feed in feeds)
            raise RuntimeError(f"Unknown --cameras {missing}. Discovered feeds: {known}")
        selected_names = list(dict.fromkeys(args.cameras))
        feeds = [feeds_by_name[name] for name in selected_names]
    if len(feeds) < 2:
        raise RuntimeError(f"Need at least 2 Kinect feeds, found {len(feeds)}")

    if args.reference_camera:
        if args.reference_camera not in {feed.name for feed in feeds}:
            raise RuntimeError(
                f"--reference-camera {args.reference_camera} is not in the exported feed set"
            )
        reference_feed = feeds_by_name[args.reference_camera]
    else:
        reference_feed = next((feed for feed in feeds if feed.name == "kinect_master"), feeds[0])

    capture_intervals, intervals_path = resolve_capture_intervals(
        session_dir, args.intervals_json, args.no_interval_filter
    )
    print(f"Session: {session_dir}")
    print("Feeds:")
    for feed in feeds:
        print(f"  {feed.name}: {feed.media_path} ({len(feed.frame_indices)} color rows)")
    print(f"Reference feed: {reference_feed.name}")
    if capture_intervals is not None:
        print(f"Intervals: {intervals_path} ({len(capture_intervals)} interval(s))")
    elif args.no_interval_filter:
        print("Intervals: disabled")
    else:
        print("Intervals: none found; exporting all matched frames")

    matches, stats = build_matches(
        feeds=feeds,
        reference_feed=reference_feed,
        every_n=args.every_n,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        max_delta_ms=args.max_delta_ms,
        capture_intervals=capture_intervals,
    )
    print(
        "Match summary: "
        f"reference_candidates={stats['reference_candidates']}, "
        f"sampled={stats['sampled_candidates']}, "
        f"kept={stats['kept_matches']}, "
        f"dropped_outside_intervals={stats['dropped_outside_intervals']}, "
        f"dropped_no_nearest={stats['dropped_no_nearest']}, "
        f"dropped_delta_limit={stats['dropped_delta_limit']}"
    )
    print(
        "Max nearest delta: "
        f"{stats['max_observed_delta_usec']} usec "
        f"(feed={stats['max_observed_delta_feed'] or 'n/a'})"
    )
    if not matches:
        raise RuntimeError("No synced matches found. Try increasing --max-delta-ms.")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (repo_root / "output" / "kinect_synced" / session_dir.name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for feed in feeds:
        (output_dir / feed.name).mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    loaders: dict[str, KinectColorLoader] = {}
    pending_playbacks: list[PyK4APlayback] = []
    calibration_by_feed = {}
    try:
        for feed in feeds:
            playback = PyK4APlayback(feed.media_path)
            playback.open()
            pending_playbacks.append(playback)
            gravity, gravity_mag, imu_count = estimate_gravity_in_color_frame(
                playback, args.imu_samples, args.gravity
            )
            camera_matrix, distortion = color_calibration(playback)
            fps = fps_from_configuration(playback.configuration["camera_fps"])
            print(
                f"  {feed.name} gravity: "
                f"[{gravity[0]:+.3f}, {gravity[1]:+.3f}, {gravity[2]:+.3f}] "
                f"from {imu_count} samples ({gravity_mag:.2f} m/s^2)"
            )
            calibration_by_feed[feed.name] = {
                "camera_matrix": camera_matrix,
                "distortion": distortion,
                "gravity": gravity,
                "fps": fps,
                "width": None,
                "height": None,
            }
            loader = KinectColorLoader(playback)
            first = matches[0]["frames"][feed.name]
            if first["frame_idx"] > 0:
                loader.seek_to_frame(first["frame_idx"], first["device_timestamp_usec"])
            loaders[feed.name] = loader
            pending_playbacks.remove(playback)

        jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)]
        companion_path = output_dir / "companion.csv"
        fieldnames = [
            "export_idx",
            "reference_camera",
            "reference_device_timestamp_usec",
            "reference_save_timestamp_ns",
        ]
        for feed in feeds:
            fieldnames.extend(
                [
                    f"frame_idx_{feed.name}",
                    f"delta_usec_{feed.name}",
                    f"rgb_{feed.name}",
                ]
            )

        written = 0
        skipped_missing_rgb = 0
        with companion_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for match_index, match in enumerate(matches):
                colors = {}
                missing = []
                for feed in feeds:
                    frame_idx = match["frames"][feed.name]["frame_idx"]
                    color = loaders[feed.name].get_color(frame_idx)
                    if color is None:
                        missing.append(f"{feed.name}={frame_idx}")
                    else:
                        colors[feed.name] = color
                if missing:
                    skipped_missing_rgb += 1
                    print(
                        f"Skipping match {match_index} (missing RGB decode): {', '.join(missing)}",
                        flush=True,
                    )
                    continue

                export_idx = written
                row = {
                    "export_idx": export_idx,
                    "reference_camera": reference_feed.name,
                    "reference_device_timestamp_usec": match["reference_device_timestamp_usec"],
                    "reference_save_timestamp_ns": match["reference_save_timestamp_ns"],
                }
                for feed in feeds:
                    image = colors[feed.name]
                    height, width = image.shape[:2]
                    calib = calibration_by_feed[feed.name]
                    if calib["width"] is None:
                        calib["width"] = width
                        calib["height"] = height
                    elif calib["width"] != width or calib["height"] != height:
                        raise RuntimeError(
                            f"{feed.name} changed resolution from "
                            f"{calib['width']}x{calib['height']} to {width}x{height}"
                        )
                    relative = Path(feed.name) / f"{export_idx:06d}.jpg"
                    if not cv2.imwrite(str(output_dir / relative), image, jpg_params):
                        raise RuntimeError(f"Failed to write {output_dir / relative}")
                    info = match["frames"][feed.name]
                    row[f"frame_idx_{feed.name}"] = info["frame_idx"]
                    row[f"delta_usec_{feed.name}"] = info["delta_usec"]
                    row[f"rgb_{feed.name}"] = str(relative)
                writer.writerow(row)
                written += 1
                if written % 100 == 0 or match_index + 1 == len(matches):
                    print(
                        f"Progress: written={written} scanned={match_index + 1}/{len(matches)}",
                        flush=True,
                    )
    finally:
        for playback in pending_playbacks:
            try:
                playback.close()
            except Exception:
                pass
        for loader in loaders.values():
            loader.close()

    if written == 0:
        raise RuntimeError("No frames were written. Every synced match failed RGB decode.")

    for feed in feeds:
        calib = calibration_by_feed[feed.name]
        write_calibration_json(
            output_dir / f"{feed.name}.json",
            camera_name=feed.name,
            camera_matrix=calib["camera_matrix"],
            distortion=calib["distortion"],
            gravity=calib["gravity"],
            width=calib["width"],
            height=calib["height"],
            fps=calib["fps"],
        )

    manifest = {
        "session_dir": str(session_dir),
        "reference_camera": reference_feed.name,
        "cameras": [feed.name for feed in feeds],
        "num_frames": written,
        "max_delta_ms": args.max_delta_ms,
        "jpg_quality": args.jpg_quality,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {written} synced frames to {output_dir}")
    if skipped_missing_rgb:
        print(f"Skipped {skipped_missing_rgb} matches because an RGB frame failed to decode.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
