#!/usr/bin/env python3
"""Pack two synced Kinect cameras into the video layout used by run_multiview_publisher.py.

Input is a directory written by export_kinect_synced_frames.py. Output is one
MP4 and one JSON per camera. Frame i of both videos is the same synced sample.
Each JSON contains camera_matrix and gravity.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert two synced Kinect JPG sequences into MP4s and publisher JSON files."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Directory produced by export_kinect_synced_frames.py.",
    )
    parser.add_argument(
        "--cameras",
        nargs=2,
        required=True,
        metavar=("CAM0", "CAM1"),
        help="Two camera names. Order is the publisher camera index (--main-camera).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for <camera>.mp4 and <camera>.json.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS written into both videos. Default: fps stored in the export JSON.",
    )
    return parser.parse_args()


def load_calibration(export_dir: Path, camera_name: str) -> dict:
    path = export_dir / f"{camera_name}.json"
    if not path.is_file():
        raise RuntimeError(f"Calibration JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("camera_matrix", "gravity", "width", "height", "fps") if key not in payload]
    if missing:
        raise RuntimeError(f"{path} is missing required fields: {missing}")
    return payload


def frame_path(export_dir: Path, camera_name: str, index: int) -> Path:
    return export_dir / camera_name / f"{index:06d}.jpg"


def count_synced_frames(export_dir: Path, cameras: list[str]) -> int:
    manifest_path = export_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        count = int(manifest["num_frames"])
    else:
        count = len(list((export_dir / cameras[0]).glob("*.jpg")))
    if count <= 0:
        raise RuntimeError(f"No synced frames found in {export_dir}")
    for camera_name in cameras:
        for index in range(count):
            path = frame_path(export_dir, camera_name, index)
            if not path.is_file():
                raise RuntimeError(f"Missing synced frame: {path}")
    return count


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    # OpenCV's avc1 fourcc selects the V4L2 hardware H.264 encoder on this
    # machine, which is not present. mp4v is the software encoder VideoCapture
    # can read back.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to open video writer for {path}")
    print(f"Writing {path.name} with mp4v at {width}x{height} {fps:g} fps")
    return writer


def publisher_json(calibration: dict) -> dict:
    camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    gravity = np.asarray(calibration["gravity"], dtype=np.float64).reshape(3)
    gravity_norm = float(np.linalg.norm(gravity))
    if gravity_norm < 1e-12:
        raise RuntimeError(f"Invalid gravity vector for {calibration.get('camera_name')}")
    gravity = gravity / gravity_norm
    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "width": int(calibration["width"]),
        "height": int(calibration["height"]),
        "camera_matrix": camera_matrix.tolist(),
        "gravity": gravity.tolist(),
    }


def verify_written_video(path: Path, frame_count: int, width: int, height: int, fps: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to reopen written video: {path}")
    try:
        written_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        written_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        written_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        written_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if (written_width, written_height) != (width, height):
        raise RuntimeError(
            f"{path} resolution is {written_width}x{written_height}, expected {width}x{height}"
        )
    if written_frames not in (0, frame_count):
        raise RuntimeError(f"{path} has {written_frames} frames, expected {frame_count}")
    if written_fps <= 0:
        raise RuntimeError(f"{path} reports FPS {written_fps}")
    if abs(written_fps - fps) > 1e-3:
        raise RuntimeError(f"{path} reports FPS {written_fps}, expected {fps}")


def main():
    args = parse_args()
    export_dir = args.export_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not export_dir.is_dir():
        raise RuntimeError(f"Export directory not found: {export_dir}")

    cameras = list(args.cameras)
    if cameras[0] == cameras[1]:
        raise RuntimeError("--cameras must name two different feeds")

    calibrations = [load_calibration(export_dir, name) for name in cameras]
    width = int(calibrations[0]["width"])
    height = int(calibrations[0]["height"])
    fps_values = [float(item["fps"]) for item in calibrations]
    if args.fps is None:
        if abs(fps_values[0] - fps_values[1]) > 1e-3:
            raise RuntimeError(
                f"Camera FPS differs: {cameras[0]}={fps_values[0]} {cameras[1]}={fps_values[1]}"
            )
        fps = fps_values[0]
    else:
        if args.fps <= 0:
            raise RuntimeError("--fps must be > 0")
        fps = float(args.fps)

    for name, calibration in zip(cameras, calibrations):
        if int(calibration["width"]) != width or int(calibration["height"]) != height:
            raise RuntimeError(
                f"{name} resolution {calibration['width']}x{calibration['height']} "
                f"does not match {width}x{height}"
            )

    frame_count = count_synced_frames(export_dir, cameras)
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = []
    video_paths = []
    try:
        for name in cameras:
            video_path = output_dir / f"{name}.mp4"
            video_paths.append(video_path)
            writers.append(open_writer(video_path, fps, width, height))

        for index in range(frame_count):
            for camera_index, name in enumerate(cameras):
                image_path = frame_path(export_dir, name, index)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Failed to read {image_path}")
                image_height, image_width = image.shape[:2]
                if (image_width, image_height) != (width, height):
                    raise RuntimeError(
                        f"{image_path} is {image_width}x{image_height}, expected {width}x{height}"
                    )
                writers[camera_index].write(image)
            if (index + 1) % 100 == 0 or index + 1 == frame_count:
                print(f"Progress: {index + 1}/{frame_count}", flush=True)
    finally:
        for writer in writers:
            writer.release()

    for video_path in video_paths:
        verify_written_video(video_path, frame_count, width, height, fps)

    for name, calibration in zip(cameras, calibrations):
        json_path = output_dir / f"{name}.json"
        json_path.write_text(
            json.dumps(publisher_json(calibration), indent=2) + "\n", encoding="utf-8"
        )

    print(f"Saved {frame_count} frames at {width}x{height} {fps:g} fps to {output_dir}")
    print("Publisher command:")
    print(
        "  python run_multiview_publisher.py --source video \\\n"
        f"    --videos {video_paths[0]} {video_paths[1]} \\\n"
        f"    --intrinsics {output_dir / (cameras[0] + '.json')} "
        f"{output_dir / (cameras[1] + '.json')} \\\n"
        "    --main-camera 0"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
