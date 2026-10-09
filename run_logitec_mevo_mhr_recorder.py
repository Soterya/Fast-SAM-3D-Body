#!/usr/bin/env python3
"""Record live Logitech Mevo NDI RGB frames as SAM 3D Body MHR predictions.

Uses the rotated portrait frame (90° CCW) from mevo_ndi_source.MevoNDISource,
matching sample_data_logitec_mevo/intri.yml and calibrate_logitec_mevo_bed_pose.py.

Kinect scripts are untouched; this file does not import pyk4a or mesh-stream-server.
"""

import argparse
import io
import json
import os
import threading
import time
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

from mevo_ndi_source import (
    DEFAULT_CAMERA_MATCH,
    DEFAULT_INTRINSICS_PATH,
    MevoNDISource,
)
from notebook.utils import setup_sam_3d_body


def build_inference_k(k, width, height, focal, focal_scale, center_principal_point):
    inference_k = np.asarray(k, dtype=np.float32).copy()
    if focal is not None:
        inference_k[0, 0] = focal
        inference_k[1, 1] = focal
    else:
        inference_k[0, 0] *= focal_scale
        inference_k[1, 1] *= focal_scale
    if center_principal_point:
        inference_k[0, 2] = width / 2.0
        inference_k[1, 2] = height / 2.0
    return inference_k


def load_pose_file(path):
    with open(path, "r") as f:
        return json.load(f)


def load_color_camera_to_bed(data):
    color = data["color_camera_wrt_bed_center"]
    rotation = Rotation.from_quat(
        np.asarray(color["quaternion_xyzw"], dtype=np.float64)
    ).as_matrix()
    translation = np.asarray(color["translation_m"], dtype=np.float64).reshape(3)
    return rotation, translation


def load_detection_roi_xy(data, path):
    """AprilTag bed polygon from the pose JSON. Missing key means no filter."""
    raw = data.get("detection_roi_xy")
    if raw is None:
        print(
            f"WARNING: {path} has no detection_roi_xy; "
            "person detections will not be filtered.",
            flush=True,
        )
        return None
    roi = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    if len(roi) < 3:
        print(
            f"WARNING: detection_roi_xy in {path} has {len(roi)} point(s); "
            "need at least 3. Running unfiltered.",
            flush=True,
        )
        return None
    source = data.get("detection_roi_source", "polygon")
    print(
        f"Detection ROI ({len(roi)} points, {source}) from {path}",
        flush=True,
    )
    return roi


def mhr_arrays_from_output(person_output):
    arrays = {}
    for key in (
        "pred_vertices",
        "pred_cam_t",
        "bbox",
        "global_rot",
        "shape_params",
        "expr_params",
        "mhr_model_params",
    ):
        if key in person_output:
            arrays[key] = np.asarray(person_output[key])
    return arrays


def transform_mhr_arrays_to_bed(arrays, color_r, color_t):
    out = dict(arrays)
    vertices_local = np.asarray(out["pred_vertices"], dtype=np.float64)
    root_t = np.asarray(out["pred_cam_t"], dtype=np.float64).reshape(3)
    vertices_bed = (color_r @ (vertices_local + root_t[None, :]).T).T + color_t[None, :]
    root_bed = color_r @ root_t + color_t

    out["pred_vertices_color"] = np.asarray(out["pred_vertices"], dtype=np.float32)
    out["pred_cam_t_color"] = np.asarray(out["pred_cam_t"], dtype=np.float32)
    out["pred_vertices"] = (vertices_bed - root_bed[None, :]).astype(np.float32)
    out["pred_cam_t"] = root_bed.astype(np.float32)

    global_rot = np.asarray(out.get("global_rot", []), dtype=np.float64)
    if global_rot.size == 3:
        out["global_rot_color"] = np.asarray(out["global_rot"], dtype=np.float32)
        out["global_rot"] = (
            Rotation.from_matrix(color_r) * Rotation.from_euler("ZYX", global_rot)
        ).as_euler("ZYX").astype(np.float32)

    out["publish_frame"] = np.asarray("bed", dtype=str)
    return out


def prepare_mhr_arrays(arrays, publish_frame, color_r, color_t):
    if publish_frame == "bed":
        return transform_mhr_arrays_to_bed(arrays, color_r, color_t)
    out = dict(arrays)
    out["publish_frame"] = np.asarray("camera", dtype=str)
    return out


def save_mhr_outputs(
    outputs,
    output_dir,
    frame_index,
    timestamp,
    rgb_shape,
    k,
    inference_k,
    publish_frame,
    color_r,
    color_t,
):
    saved = 0
    timestamp_ns = int(round(timestamp * 1e9))
    for pid, person_output in enumerate(outputs):
        arrays = mhr_arrays_from_output(person_output)
        if "pred_vertices" not in arrays or "pred_cam_t" not in arrays:
            continue
        arrays = prepare_mhr_arrays(arrays, publish_frame, color_r, color_t)

        arrays.update(
            {
                "timestamp": np.asarray(timestamp, dtype=np.float64),
                "timestamp_ns": np.asarray(timestamp_ns, dtype=np.int64),
                "frame_index": np.asarray(frame_index, dtype=np.int64),
                "person_index": np.asarray(pid, dtype=np.int64),
                "image_height": np.asarray(rgb_shape[0], dtype=np.int64),
                "image_width": np.asarray(rgb_shape[1], dtype=np.int64),
                "camera_intrinsics": np.asarray(k, dtype=np.float32),
                "inference_intrinsics": np.asarray(inference_k, dtype=np.float32),
            }
        )
        path = os.path.join(
            output_dir,
            f"{timestamp_ns}_frame_{frame_index:06d}_person_{pid:03d}_mhr.npz",
        )
        np.savez_compressed(path, **arrays)
        saved += 1
    return saved


def mhr_payload_from_outputs(
    outputs,
    frame_index,
    timestamp,
    rgb_shape,
    k,
    inference_k,
    publish_frame,
    color_r,
    color_t,
):
    people = []
    for pid, person_output in enumerate(outputs):
        if "pred_vertices" not in person_output or "pred_cam_t" not in person_output:
            continue
        record = prepare_mhr_arrays(
            mhr_arrays_from_output(person_output),
            publish_frame,
            color_r,
            color_t,
        )
        record["pid"] = pid
        people.append(record)

    if not people:
        return None

    arrays = {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "num_people": np.asarray(len(people), dtype=np.int64),
        "source_timestamp": np.asarray(timestamp, dtype=np.float64),
        "timestamp_ns": np.asarray(int(round(timestamp * 1e9)), dtype=np.int64),
        "image_height": np.asarray(rgb_shape[0], dtype=np.int64),
        "image_width": np.asarray(rgb_shape[1], dtype=np.int64),
        "camera_intrinsics": np.asarray(k, dtype=np.float32),
        "inference_intrinsics": np.asarray(inference_k, dtype=np.float32),
        "publish_frame": np.asarray(publish_frame, dtype=str),
    }
    for person in people:
        pid = person["pid"]
        for key, value in person.items():
            if key == "pid":
                continue
            arrays[f"person_{pid:03d}_{key}"] = np.asarray(value)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    metadata = {
        "frame_index": int(frame_index),
        "num_people": len(people),
        "source_timestamp": float(timestamp),
        "created_timestamp": time.time(),
        "format": "npz",
        "schema": "fast_sam_3d_body.mhr.v1",
        "publish_frame": publish_frame,
        "camera": "logitec_mevo",
    }
    return metadata, buffer.getvalue()


def publish_mhr_payload(socket, topic, payload_tuple):
    if payload_tuple is None:
        return False
    metadata, payload = payload_tuple
    publish_metadata = dict(metadata)
    publish_metadata["publish_timestamp"] = time.time()
    socket.send_multipart(
        [
            topic.encode("utf-8"),
            json.dumps(publish_metadata).encode("utf-8"),
            payload,
        ]
    )
    return True


def fixed_rate_publish_loop(socket, topic, publish_fps, latest_lock, latest_state, stop_event):
    period = 1.0 / publish_fps
    next_publish = time.perf_counter()
    print(f"Fixed-rate MHR publishing enabled: fps={publish_fps}", flush=True)
    while not stop_event.is_set():
        now = time.perf_counter()
        wait = next_publish - now
        if wait > 0:
            stop_event.wait(min(wait, 0.05))
            continue

        with latest_lock:
            payload_tuple = latest_state.get("payload")
        publish_mhr_payload(socket, topic, payload_tuple)

        next_publish += period
        if next_publish < now - period:
            next_publish = now + period


def main():
    parser = argparse.ArgumentParser(
        description="Live Mevo NDI -> SAM 3D Body MHR recorder/publisher (rotated portrait frames)"
    )
    parser.add_argument("--ndi-match", default=DEFAULT_CAMERA_MATCH)
    parser.add_argument("--find-timeout", type=float, default=30.0)
    parser.add_argument(
        "--intrinsics-path",
        default=str(DEFAULT_INTRINSICS_PATH),
        help="OpenCV YAML K for the rotated portrait frame",
    )
    parser.add_argument("--intrinsics-name", default="0")
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable 90° CCW rotate (default rotate matches intri.yml / calibration)",
    )
    parser.add_argument("--read-fps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="./output_mevo_mhr")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-endpoint", default="tcp://*:5557")
    parser.add_argument("--publish-topic", default="mevo.mhr")
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--mhr-publish-frame", default="bed", choices=["camera", "bed"])
    parser.add_argument(
        "--camera-pose-path",
        default="./sample_data_logitec_mevo/camera_poses_wrt_bed_center.json",
    )
    parser.add_argument("--model", default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--local-checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    parser.add_argument("--detector", default="yolo_pose", choices=["vitdet", "yolo", "yolo_pose"])
    parser.add_argument("--detector-model", default="./checkpoints/yolo/yolo11x-pose.engine")
    parser.add_argument("--hand-box-source", default="yolo_pose", choices=["body_decoder", "yolo_pose"])
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument("--center-principal-point", action="store_true", default=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.read_fps <= 0:
        parser.error("--read-fps must be > 0")
    if args.publish_fps <= 0:
        parser.error("--publish-fps must be > 0")

    color_r = None
    color_t = None
    pose_data = None
    if args.mhr_publish_frame == "bed":
        pose_data = load_pose_file(args.camera_pose_path)
        color_r, color_t = load_color_camera_to_bed(pose_data)
        print(
            f"Publishing/saving MHR in bed frame using camera pose: {args.camera_pose_path}",
            flush=True,
        )
    elif os.path.isfile(args.camera_pose_path):
        pose_data = load_pose_file(args.camera_pose_path)
        print("Publishing/saving MHR in camera frame", flush=True)
    else:
        print("Publishing/saving MHR in camera frame", flush=True)
        print(
            f"WARNING: no pose file at {args.camera_pose_path}; detection ROI disabled.",
            flush=True,
        )

    roi_polygon = None
    if pose_data is not None:
        roi_polygon = load_detection_roi_xy(pose_data, args.camera_pose_path)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    pub_socket = None
    latest_lock = threading.Lock()
    latest_state = {"payload": None}
    stop_event = threading.Event()
    publisher_thread = None
    if args.publish:
        context = zmq.Context.instance()
        pub_socket = context.socket(zmq.PUB)
        pub_socket.setsockopt(zmq.SNDHWM, 1)
        pub_socket.setsockopt(zmq.LINGER, 0)
        pub_socket.bind(args.publish_endpoint)
        print(
            f"Publishing MHR to {args.publish_endpoint} topic={args.publish_topic}",
            flush=True,
        )
        publisher_thread = threading.Thread(
            target=fixed_rate_publish_loop,
            args=(
                pub_socket,
                args.publish_topic,
                args.publish_fps,
                latest_lock,
                latest_state,
                stop_event,
            ),
            daemon=True,
        )
        publisher_thread.start()

    source = MevoNDISource(
        camera_match=args.ndi_match,
        find_timeout_s=args.find_timeout,
        rotate_90_ccw=not args.no_rotate,
        intrinsics_path=args.intrinsics_path,
        intrinsics_camera_name=args.intrinsics_name,
    )
    source.start()
    print(
        f"Working in rotated portrait frame: {source.width}x{source.height} "
        f"(rotate_90_ccw={not args.no_rotate})",
        flush=True,
    )

    inference_k = build_inference_k(
        source.k,
        source.width,
        source.height,
        args.focal,
        args.focal_scale,
        args.center_principal_point,
    )
    print("K used for inference:")
    print(inference_k)

    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        fov_name=None,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )
    cam_int = torch.from_numpy(inference_k[None]).cuda()

    frame_index = 0
    processed_count = 0
    processing_times = deque(maxlen=50)
    loop_start = time.perf_counter()
    read_period = 1.0 / args.read_fps
    next_process_timestamp = 0.0
    print(
        f"Mevo NDI | Read/process FPS target: {args.read_fps} | Publish FPS: {args.publish_fps}",
        flush=True,
    )
    try:
        while args.max_frames <= 0 or processed_count < args.max_frames:
            bgr, timestamp = source.read()
            if bgr is None or timestamp is None:
                continue

            # SAM 3D Body expects RGB; MevoNDISource yields rotated BGR.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            if timestamp < next_process_timestamp:
                frame_index += 1
                continue
            if frame_index % args.frame_stride != 0:
                frame_index += 1
                continue
            next_process_timestamp = timestamp + read_period

            t0 = time.perf_counter()
            outputs = estimator.process_one_image(
                rgb,
                cam_int=cam_int,
                inference_type="body",
                hand_box_source=args.hand_box_source,
                roi_polygon=roi_polygon,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            processing_times.append(dt)

            saved = 0
            if args.save:
                saved = save_mhr_outputs(
                    outputs,
                    args.output_dir,
                    frame_index,
                    timestamp,
                    rgb.shape,
                    source.k,
                    inference_k,
                    args.mhr_publish_frame,
                    color_r,
                    color_t,
                )
            published = False
            if pub_socket is not None:
                payload_tuple = mhr_payload_from_outputs(
                    outputs,
                    frame_index,
                    timestamp,
                    rgb.shape,
                    source.k,
                    inference_k,
                    args.mhr_publish_frame,
                    color_r,
                    color_t,
                )
                if payload_tuple is not None:
                    with latest_lock:
                        latest_state["payload"] = payload_tuple
                    published = True
                elif roi_polygon is not None:
                    with latest_lock:
                        latest_state["payload"] = None
            rolling_fps = (
                1.0 / float(np.mean(processing_times))
                if processing_times
                else 0.0
            )
            throughput_fps = (processed_count + 1) / (
                time.perf_counter() - loop_start
            )
            print(
                f"frame={frame_index} people={len(outputs)} saved={saved} "
                f"published={int(published)} "
                f"dt={dt:.3f}s inst_fps={1.0 / dt:.2f} "
                f"rolling_fps={rolling_fps:.2f} throughput_fps={throughput_fps:.2f} "
                f"ts={timestamp:.6f}",
                flush=True,
            )
            frame_index += 1
            processed_count += 1
    except KeyboardInterrupt:
        print("Stopping Mevo MHR recorder.")
    finally:
        stop_event.set()
        if publisher_thread is not None:
            publisher_thread.join(timeout=2.0)
        if pub_socket is not None:
            pub_socket.close(0)
        source.close()


if __name__ == "__main__":
    main()
