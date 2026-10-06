"""Play back a recorded smpl_data.npz as a front/side SMPL mesh."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from debug_smpl_stream import RENDER_HEIGHT, RENDER_WIDTH, prepare_render_records
from mocap.utils.renderer import Renderer
from mocap.utils.smpl_render_utils import (
    draw_skeleton,
    load_smpl_model,
    smpl_vertices_joints_from_pose,
    _project_joints,
)


def load_records(npz_path):
    data = np.load(npz_path)
    required = ("timestamps", "body_quats", "smpl_joints", "smpl_poses")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"{npz_path} is missing keys: {missing}")

    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    body_quats = np.asarray(data["body_quats"], dtype=np.float64)
    smpl_joints = np.asarray(data["smpl_joints"], dtype=np.float64)
    smpl_poses = np.asarray(data["smpl_poses"], dtype=np.float64)
    count = len(timestamps)
    if not (
        body_quats.shape == (count, 4)
        and smpl_joints.shape == (count, 24, 3)
        and smpl_poses.shape == (count, 21, 3)
    ):
        raise RuntimeError(
            "Unexpected shapes: "
            f"timestamps {timestamps.shape}, body_quats {body_quats.shape}, "
            f"smpl_joints {smpl_joints.shape}, smpl_poses {smpl_poses.shape}"
        )
    if count == 0:
        raise RuntimeError(f"{npz_path} has no poses")

    records = []
    for index in range(count):
        records.append(
            {
                "frame_index": index,
                "timestamp": float(timestamps[index]),
                "body_quat": body_quats[index],
                "smpl_joints": smpl_joints[index],
                "smpl_pose": smpl_poses[index],
            }
        )
    return records


def infer_fps(records, fallback):
    if len(records) < 2:
        return fallback
    deltas = np.diff([record["timestamp"] for record in records])
    deltas = deltas[deltas > 1e-4]
    if len(deltas) == 0:
        return fallback
    return float(np.clip(1.0 / np.median(deltas), 1.0, 60.0))


def render_frames(records, smpl_model_path, show_joints):
    smpl_model, faces, device, num_betas = load_smpl_model(smpl_model_path)
    width, height = RENDER_WIDTH, RENDER_HEIGHT
    focal_length = float(max(width, height))
    renderer = Renderer(focal_length=focal_length, faces=faces)
    white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
    cam_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    frames = []

    try:
        for index, record in enumerate(records):
            verts, mesh_joints = smpl_vertices_joints_from_pose(
                record["smpl_pose"],
                smpl_model=smpl_model,
                device=device,
                num_betas=num_betas,
                body_quat=record["body_quat"],
            )
            front = renderer(
                verts,
                cam_t,
                white_bg.copy(),
                mesh_base_color=(0.65, 0.74, 0.86),
                scene_bg_color=(1, 1, 1),
            )
            side = renderer(
                verts,
                cam_t,
                white_bg.copy(),
                side_view=True,
                mesh_base_color=(0.65, 0.74, 0.86),
                scene_bg_color=(1, 1, 1),
            )
            frame_front = (front * 255).astype(np.uint8)
            frame_side = (side * 255).astype(np.uint8)
            if show_joints:
                joints_front, mask_front = _project_joints(
                    mesh_joints, cam_t, focal_length, width, height
                )
                draw_skeleton(frame_front, joints_front, mask_front)
                joints_side, mask_side = _project_joints(
                    mesh_joints, cam_t, focal_length, width, height, side_view=True
                )
                draw_skeleton(frame_side, joints_side, mask_side)

            cv2.putText(
                frame_front, "Front", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame_side, "Side", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA
            )
            combined = np.concatenate([frame_front, frame_side], axis=1)
            t0 = records[0]["timestamp"]
            cv2.putText(
                combined,
                f"{index + 1}/{len(records)}  t={record['timestamp'] - t0:.3f}s",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 40, 40),
                2,
                cv2.LINE_AA,
            )
            frames.append(combined)
            print(f"Rendered {index + 1}/{len(records)}", flush=True)
    finally:
        renderer.delete()

    return frames


def write_video(frames, output_path, fps):
    height, width = frames[0].shape[:2]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_contact_sheet(frames, output_path):
    sheet = np.concatenate(frames, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write {output_path}")


def play(frames, fps):
    if not os.environ.get("DISPLAY"):
        print("No DISPLAY set, skipping the interactive window.")
        return
    delay = max(1, int(round(1000.0 / fps)))
    index = 0
    playing = True
    window = "SMPL record"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error:
        print("OpenCV was built without a GUI, so there is no playback window.")
        print("Open the written png or mp4 instead.")
        return
    print("Window: space play/pause, a/d step, q quit")
    while True:
        cv2.imshow(window, frames[index])
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key in (81, 2, ord("a"), ord("p")):  # left
            playing = False
            index = (index - 1) % len(frames)
        elif key in (83, 3, ord("d"), ord("n")):  # right
            playing = False
            index = (index + 1) % len(frames)
        elif playing:
            index = (index + 1) % len(frames)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visually check a recorded smpl_data.npz (front and side SMPL mesh)"
    )
    parser.add_argument(
        "--npz",
        default="output/records/2026-10-05_19-00-29/smpl_data.npz",
        help="Path to smpl_data.npz",
    )
    parser.add_argument(
        "--smpl-model-path",
        default="mhr2smpl/data/SMPL_NEUTRAL.pkl",
        help="SMPL model used to build the mesh",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Playback FPS. 0 uses the recorded timestamps.")
    parser.add_argument("--output", default="", help="Preview mp4 path. Default is next to the npz.")
    parser.add_argument("--sheet", default="", help="Contact-sheet png path. Default is next to the npz.")
    parser.add_argument("--show-joints", action="store_true", help="Draw the SMPL skeleton on the mesh")
    parser.add_argument("--no-show", action="store_true", help="Write files only, do not open a window")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    records = load_records(npz_path)
    fps = args.fps if args.fps > 0 else infer_fps(records, fallback=10.0)
    duration = records[-1]["timestamp"] - records[0]["timestamp"]
    print(
        f"{npz_path}: {len(records)} poses, span {duration:.3f}s, playback {fps:.2f} fps"
    )

    render_records = prepare_render_records(records)
    frames = render_frames(render_records, args.smpl_model_path, args.show_joints)

    video_path = Path(args.output) if args.output else npz_path.with_name("smpl_preview.mp4")
    sheet_path = Path(args.sheet) if args.sheet else npz_path.with_name("smpl_preview.png")
    write_video(frames, video_path, fps)
    write_contact_sheet(frames, sheet_path)
    print(f"Wrote {video_path}")
    print(f"Wrote {sheet_path}")

    if not args.no_show:
        play(frames, fps)


if __name__ == "__main__":
    main()
