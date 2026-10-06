import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from view_smpl_npz_open3d import (
    SMPL_PARENTS,
    import_open3d,
    smpl_forward,
)

_SMPL_MODEL_CACHE = {}


def decode_smpl_payload(payload):
    data = np.load(io.BytesIO(payload))
    num_people = int(np.asarray(data["num_people"]).reshape(()))
    if num_people < 1:
        return None

    prefix = "person_000_"
    smpl_pose = np.asarray(data[prefix + "smpl_pose"], dtype=np.float32)
    betas = np.asarray(data[prefix + "betas"], dtype=np.float32)
    body_quat = (
        np.asarray(data[prefix + "body_quat_xyzw"], dtype=np.float32)
        if prefix + "body_quat_xyzw" in data
        else None
    )
    pred_cam_t = (
        np.asarray(data[prefix + "pred_cam_t"], dtype=np.float32)
        if prefix + "pred_cam_t" in data
        else np.zeros(3, dtype=np.float32)
    )
    official_transl = (
        np.asarray(data[prefix + "official_transl"], dtype=np.float32)
        if prefix + "official_transl" in data
        else None
    )
    official_global_orient = (
        np.asarray(data[prefix + "official_global_orient"], dtype=np.float32)
        if prefix + "official_global_orient" in data
        else None
    )
    smpl_body_pose_full = (
        np.asarray(data[prefix + "smpl_body_pose_full"], dtype=np.float32)
        if prefix + "smpl_body_pose_full" in data
        else None
    )
    smpl_vertices = (
        np.asarray(data[prefix + "smpl_vertices"], dtype=np.float32)
        if prefix + "smpl_vertices" in data
        else None
    )
    smpl_joints = (
        np.asarray(data[prefix + "smpl_joints"], dtype=np.float32)
        if prefix + "smpl_joints" in data
        else None
    )
    publish_frame = (
        str(np.asarray(data[prefix + "publish_frame"]).reshape(()))
        if prefix + "publish_frame" in data
        else "camera"
    )
    frame_index = int(np.asarray(data["frame_index"]).reshape(()))
    source_timestamp = (
        float(np.asarray(data["source_timestamp"]).reshape(()))
        if "source_timestamp" in data
        else None
    )
    return {
        "frame_index": frame_index,
        "num_people": num_people,
        "source_timestamp": source_timestamp,
        "smpl_pose": smpl_pose,
        "betas": betas,
        "body_quat": body_quat,
        "pred_cam_t": pred_cam_t,
        "official_transl": official_transl,
        "official_global_orient": official_global_orient,
        "smpl_body_pose_full": smpl_body_pose_full,
        "smpl_vertices": smpl_vertices,
        "smpl_joints": smpl_joints,
        "publish_frame": publish_frame,
    }


def decode_mhr_payload(payload):
    data = np.load(io.BytesIO(payload))
    num_people = int(np.asarray(data["num_people"]).reshape(()))
    if num_people < 1:
        return None

    prefix = "person_000_"
    if prefix + "pred_vertices" not in data or prefix + "pred_cam_t" not in data:
        return None

    frame_index = int(np.asarray(data["frame_index"]).reshape(()))
    source_timestamp = (
        float(np.asarray(data["source_timestamp"]).reshape(()))
        if "source_timestamp" in data
        else None
    )
    global_rot = (
        np.asarray(data[prefix + "global_rot"], dtype=np.float32)
        if prefix + "global_rot" in data
        else None
    )
    publish_frame = (
        str(np.asarray(data[prefix + "publish_frame"]).reshape(()))
        if prefix + "publish_frame" in data
        else "camera"
    )
    return {
        "frame_index": frame_index,
        "num_people": num_people,
        "source_timestamp": source_timestamp,
        "pred_vertices": np.asarray(data[prefix + "pred_vertices"], dtype=np.float32),
        "pred_cam_t": np.asarray(data[prefix + "pred_cam_t"], dtype=np.float32),
        "global_rot": global_rot,
        "publish_frame": publish_frame,
    }


def load_mesh_faces(o3d, path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {path}")
    return np.asarray(mesh.triangles, dtype=np.int32)


def load_smpl_faces(smpl_model_path):
    import smplx

    model = smplx.SMPL(model_path=str(smpl_model_path), gender="neutral", batch_size=1)
    return np.asarray(model.faces, dtype=np.int32)


def load_smpl_model_cached(smpl_model_path):
    import smplx
    import torch

    key = str(Path(smpl_model_path).expanduser().resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (key, str(device))
    if cache_key not in _SMPL_MODEL_CACHE:
        model = smplx.SMPL(model_path=key, gender="neutral", batch_size=1).to(device)
        model.eval()
        _SMPL_MODEL_CACHE[cache_key] = (model, device, int(getattr(model, "num_betas", 10)))
    return _SMPL_MODEL_CACHE[cache_key]


def official_smpl_forward(
    smpl_model_path,
    smpl_body_pose_full,
    betas,
    official_global_orient,
    official_transl,
):
    import torch

    smpl_model, device, num_betas = load_smpl_model_cached(smpl_model_path)

    body_pose_np = np.asarray(smpl_body_pose_full, dtype=np.float32).reshape(-1)
    if body_pose_np.shape != (69,):
        raise RuntimeError(
            f"Expected smpl_body_pose_full shape (69,), got {body_pose_np.shape}"
        )

    global_orient_np = np.asarray(official_global_orient, dtype=np.float32).reshape(3)
    transl_np = np.asarray(official_transl, dtype=np.float32).reshape(3)
    betas_src = np.asarray(betas, dtype=np.float32).reshape(-1)
    betas_np = np.zeros((num_betas,), dtype=np.float32)
    betas_np[: min(num_betas, betas_src.shape[0])] = betas_src[:num_betas]

    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.from_numpy(global_orient_np.reshape(1, 3)).to(device),
            body_pose=torch.from_numpy(body_pose_np.reshape(1, 69)).to(device),
            betas=torch.from_numpy(betas_np.reshape(1, num_betas)).to(device),
            transl=torch.from_numpy(transl_np.reshape(1, 3)).to(device),
        )

    vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0, :24].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    return vertices, joints, faces


def load_bed_pose(path):
    with open(path, "r") as f:
        data = json.load(f)

    color = data["color_camera_wrt_bed_center"]
    color_t = np.asarray(color["translation_m"], dtype=np.float64)
    color_r = Rotation.from_quat(np.asarray(color["quaternion_xyzw"], dtype=np.float64)).as_matrix()

    floor_z = 0.0
    if (
        "depth_camera_wrt_bed_center" in data
        and "height_from_depth_cam_center_to_floor_m" in data
    ):
        depth_t = np.asarray(
            data["depth_camera_wrt_bed_center"]["translation_m"], dtype=np.float64
        )
        floor_z = float(depth_t[2] - data["height_from_depth_cam_center_to_floor_m"])

    return color_r, color_t, floor_z


def camera_translation(pred_cam_t, convention):
    t = np.asarray(pred_cam_t, dtype=np.float64).reshape(3).copy()
    if convention == "mhr_flip_yz":
        t[1] *= -1.0
        t[2] *= -1.0
    elif convention == "raw":
        pass
    else:
        raise RuntimeError(f"Unknown translation convention: {convention}")
    return t


def transform_to_bed(vertices, joints, root_t_color, color_r, color_t):
    vertices_color = vertices.astype(np.float64) + root_t_color[None, :]
    joints_color = joints.astype(np.float64) + root_t_color[None, :]
    vertices_bed = (color_r @ vertices_color.T).T + color_t[None, :]
    joints_bed = (color_r @ joints_color.T).T + color_t[None, :]
    root_bed = color_r @ root_t_color + color_t
    return vertices_bed, joints_bed, root_bed


def transform_to_camera(vertices, joints, root_t_color):
    vertices_camera = vertices.astype(np.float64) + root_t_color[None, :]
    joints_camera = joints.astype(np.float64) + root_t_color[None, :]
    return vertices_camera, joints_camera, root_t_color


def make_floor_grid(o3d, floor_z, size=4.0, step=0.25):
    half = size * 0.5
    coords = np.arange(-half, half + 1e-6, step)
    points = []
    lines = []
    for c in coords:
        start = len(points)
        points.extend([[-half, c, floor_z], [half, c, floor_z]])
        lines.append([start, start + 1])
        start = len(points)
        points.extend([[c, -half, floor_z], [c, half, floor_z]])
        lines.append([start, start + 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    grid.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.55, 0.55, 0.55]], dtype=np.float64), (len(lines), 1))
    )
    return grid


def make_frame_lines(o3d, origin, rotation, scale=0.35):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    endpoints = origin[None, :] + (rotation @ (np.eye(3) * scale)).T
    points = np.vstack([origin[None, :], endpoints])
    lines = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    frame = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frame.colors = o3d.utility.Vector3dVector(
        np.asarray([[1.0, 0.05, 0.05], [0.05, 0.8, 0.05], [0.05, 0.2, 1.0]], dtype=np.float64)
    )
    return frame


def update_frame_lines(o3d, frame, origin, rotation, scale=0.35):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    endpoints = origin[None, :] + (rotation @ (np.eye(3) * scale)).T
    frame.points = o3d.utility.Vector3dVector(np.vstack([origin[None, :], endpoints]))


def make_lines():
    lines = []
    for idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx >= 0:
            lines.append([parent_idx, idx])
    return np.asarray(lines, dtype=np.int32)


def create_scene(o3d, vertices, faces, joints):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.62, 0.74, 0.9])

    joint_cloud = o3d.geometry.PointCloud()
    joint_cloud.points = o3d.utility.Vector3dVector(joints.astype(np.float64))
    joint_cloud.paint_uniform_color([0.95, 0.16, 0.1])

    lines = make_lines()
    skeleton = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(joints.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    skeleton.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.05, 0.05, 0.05]], dtype=np.float64), (len(lines), 1))
    )

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35)
    return mesh, joint_cloud, skeleton, frame


def create_mesh_scene(o3d, vertices, faces, color):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def make_sub_socket(context, endpoint, topic):
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)
    return socket


def drain_latest(socket):
    msg = None
    while True:
        try:
            msg = socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", default="both", choices=["smpl", "mhr", "both"])
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--smpl_endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--smpl_topic", default="kinect_master.smpl")
    parser.add_argument("--mhr_endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--mhr_topic", default="kinect_master.mhr")
    parser.add_argument("--smpl_model_path", default="mhr2smpl/data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--mhr_mesh_path", default="mhr2smpl/data/mhr_face_mask.ply")
    parser.add_argument(
        "--frame",
        default="bed",
        choices=["camera", "bed"],
        help="Visualization frame. camera plots published SMPL in its original frame; bed applies camera_poses_wrt_bed_center.json.",
    )
    parser.add_argument("--camera_pose_path", default="sample_data/camera_poses_wrt_bed_center.json")
    parser.add_argument(
        "--translation_convention",
        default="raw",
        choices=["mhr_flip_yz", "raw"],
        help="How to convert published pred_cam_t before applying the color-camera pose.",
    )
    parser.add_argument(
        "--translation_source",
        default="auto",
        choices=["auto", "pred_cam_t", "official_transl"],
        help="Translation source. auto uses official_transl when present, otherwise pred_cam_t.",
    )
    parser.add_argument("--no_apply_body_quat", action="store_true")
    parser.add_argument(
        "--geometry_source",
        default="published",
        choices=["published", "reconstruct", "official_params"],
        help="published uses official smpl_vertices/smpl_joints when available; reconstruct reruns local SMPL forward; official_params reruns SMPL from official full pose parameters.",
    )
    parser.add_argument("--root_frame_size", type=float, default=0.35)
    parser.add_argument("--floor_size", type=float, default=4.0)
    parser.add_argument("--floor_step", type=float, default=0.25)
    parser.add_argument("--view", default="top", choices=["top", "angled"])
    parser.add_argument("--window_name", default="Live Kinect SMPL")
    parser.add_argument("--poll_hz", type=float, default=60.0)
    args = parser.parse_args()

    if args.poll_hz <= 0:
        raise RuntimeError("--poll_hz must be > 0")
    if args.endpoint is None:
        args.endpoint = (
            args.mhr_endpoint
            if args.input_type == "mhr"
            else args.smpl_endpoint
        )
    if args.topic is None:
        args.topic = (
            args.mhr_topic if args.input_type == "mhr" else args.smpl_topic
        )

    o3d = import_open3d()
    mhr_faces = (
        load_mesh_faces(o3d, Path(args.mhr_mesh_path))
        if args.input_type in ("mhr", "both")
        else None
    )
    smpl_faces = (
        load_smpl_faces(args.smpl_model_path)
        if args.input_type in ("smpl", "both")
        else None
    )
    if args.frame == "bed":
        color_r, color_t, floor_z = load_bed_pose(args.camera_pose_path)
        color_frame = make_frame_lines(o3d, color_t, color_r, scale=0.25)
        floor_grid = make_floor_grid(
            o3d, floor_z, size=args.floor_size, step=args.floor_step
        )
        global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.45)
    else:
        color_r = np.eye(3, dtype=np.float64)
        color_t = np.zeros(3, dtype=np.float64)
        floor_z = 0.0
        color_frame = None
        floor_grid = None
        global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.45)

    context = zmq.Context.instance()
    sockets = []
    if args.input_type == "both":
        sockets.append(
            ("mhr", make_sub_socket(context, args.mhr_endpoint, args.mhr_topic), args.mhr_topic)
        )
        sockets.append(
            (
                "smpl",
                make_sub_socket(context, args.smpl_endpoint, args.smpl_topic),
                args.smpl_topic,
            )
        )
        print(f"Subscribing to MHR endpoint: {args.mhr_endpoint} topic={args.mhr_topic}")
        print(f"Subscribing to SMPL endpoint: {args.smpl_endpoint} topic={args.smpl_topic}")
        print("Waiting for first MHR/SMPL packet...")
    else:
        sockets.append((args.input_type, make_sub_socket(context, args.endpoint, args.topic), args.topic))
        print(f"Subscribing to {args.input_type.upper()} endpoint: {args.endpoint}")
        print(f"Topic: {args.topic}")
        print(f"Waiting for first {args.input_type.upper()} packet...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.window_name, width=1000, height=800)

    mesh = None
    joint_cloud = None
    skeleton = None
    pelvis_frame = None
    smpl_pelvis_frame = None
    mhr_pelvis_frame = None
    mhr_mesh = None
    faces = None
    base_geometry_added = False
    smpl_geometry_added = False
    mhr_geometry_added = False
    last_frame_index = None
    next_status = time.perf_counter()
    sleep_dt = 1.0 / args.poll_hz

    try:
        while True:
            for stream_type, socket, expected_topic in sockets:
                msg = drain_latest(socket)
                if msg is None:
                    continue
                if len(msg) != 3:
                    print(f"Skipping malformed multipart message with {len(msg)} parts")
                else:
                    topic_frame, metadata_frame, payload = msg
                    topic = topic_frame.decode("utf-8") if isinstance(topic_frame, bytes) else str(topic_frame)
                    if topic == expected_topic:
                        metadata = json.loads(metadata_frame.decode("utf-8"))
                        sample = (
                            decode_mhr_payload(payload)
                            if stream_type == "mhr"
                            else decode_smpl_payload(payload)
                        )
                        if sample is not None:
                            if not base_geometry_added:
                                vis.add_geometry(global_frame)
                                if floor_grid is not None:
                                    vis.add_geometry(floor_grid)
                                if color_frame is not None:
                                    vis.add_geometry(color_frame)
                                view_control = vis.get_view_control()
                                if args.view == "top":
                                    view_control.set_front([0.0, 0.0, -1.0])
                                    view_control.set_lookat([0.0, 0.0, floor_z])
                                    view_control.set_up([0.0, 1.0, 0.0])
                                    view_control.set_zoom(0.55)
                                else:
                                    view_control.set_front([0.0, -1.0, 0.35])
                                    view_control.set_lookat(
                                        [0.0, 0.0, max(0.5, floor_z + 1.0)]
                                    )
                                    view_control.set_up([0.0, 0.0, 1.0])
                                    view_control.set_zoom(0.75)
                                base_geometry_added = True

                            if stream_type == "mhr":
                                already_in_bed = sample.get("publish_frame") == "bed"
                                root_t_color = camera_translation(
                                    sample["pred_cam_t"], args.translation_convention
                                )
                                vertices_local = np.asarray(
                                    sample["pred_vertices"], dtype=np.float64
                                )
                                if args.frame == "bed" and already_in_bed:
                                    vertices = vertices_local + root_t_color[None, :]
                                    root_position = root_t_color
                                elif args.frame == "bed":
                                    vertices = (color_r @ (vertices_local + root_t_color[None, :]).T).T + color_t[None, :]
                                    root_position = color_r @ root_t_color + color_t
                                else:
                                    vertices = vertices_local + root_t_color[None, :]
                                    root_position = root_t_color
                                body_r_color = np.eye(3)
                                if (
                                    sample["global_rot"] is not None
                                    and sample["global_rot"].size == 3
                                ):
                                    body_r_color = Rotation.from_euler(
                                        "ZYX", sample["global_rot"]
                                    ).as_matrix()
                                body_r_vis = (
                                    body_r_color
                                    if already_in_bed
                                    else color_r @ body_r_color
                                )

                                if not mhr_geometry_added:
                                    mhr_mesh = create_mesh_scene(
                                        o3d, vertices, mhr_faces, [0.62, 0.9, 0.68]
                                    )
                                    mhr_pelvis_frame = make_frame_lines(
                                        o3d,
                                        root_position,
                                        body_r_vis,
                                        scale=args.root_frame_size,
                                    )
                                    vis.add_geometry(mhr_mesh)
                                    vis.add_geometry(mhr_pelvis_frame)
                                    mhr_geometry_added = True
                                else:
                                    mhr_mesh.vertices = o3d.utility.Vector3dVector(
                                        vertices.astype(np.float64)
                                    )
                                    mhr_mesh.triangles = o3d.utility.Vector3iVector(
                                        mhr_faces.astype(np.int32)
                                    )
                                    mhr_mesh.compute_vertex_normals()
                                    update_frame_lines(
                                        o3d,
                                        mhr_pelvis_frame,
                                        root_position,
                                        body_r_vis,
                                        scale=args.root_frame_size,
                                    )
                                    vis.update_geometry(mhr_mesh)
                                    vis.update_geometry(mhr_pelvis_frame)

                                last_frame_index = sample["frame_index"]
                                now = time.perf_counter()
                                if now >= next_status:
                                    print(
                                        "live "
                                        f"type=mhr "
                                        f"frame={last_frame_index} "
                                        f"people={sample['num_people']} "
                                        f"frame_mode={args.frame} "
                                        f"publish_frame={sample.get('publish_frame')} "
                                        f"root={root_position.round(3).tolist()} "
                                        f"source_ts={sample['source_timestamp']} "
                                        f"published_ts={metadata.get('publish_timestamp')}",
                                        flush=True,
                                    )
                                    next_status = now + 2.0
                                continue

                            use_published_geometry = (
                                args.geometry_source == "published"
                                and sample["smpl_vertices"] is not None
                                and sample["smpl_joints"] is not None
                            )
                            use_official_params = args.geometry_source == "official_params"
                            if use_published_geometry:
                                faces_now = smpl_faces
                            elif use_official_params:
                                missing_official = [
                                    key
                                    for key in (
                                        "smpl_body_pose_full",
                                        "official_global_orient",
                                        "official_transl",
                                    )
                                    if sample[key] is None
                                ]
                                if missing_official:
                                    print(
                                        "Skipping SMPL official-params reconstruction: "
                                        f"missing {missing_official}",
                                        flush=True,
                                    )
                                    continue
                                vertices_recon, joints_recon, faces_now = official_smpl_forward(
                                    args.smpl_model_path,
                                    sample["smpl_body_pose_full"],
                                    sample["betas"],
                                    sample["official_global_orient"],
                                    sample["official_transl"],
                                )
                            else:
                                vertices_recon, joints_recon, faces_now = smpl_forward(
                                    args.smpl_model_path,
                                    sample["smpl_pose"],
                                    sample["betas"],
                                    sample["body_quat"],
                                    not args.no_apply_body_quat,
                                )
                            use_official_transl = (
                                args.translation_source == "official_transl"
                                or (
                                    args.translation_source == "auto"
                                    and sample["official_transl"] is not None
                                )
                            )
                            if use_published_geometry:
                                vertices_published = np.asarray(
                                    sample["smpl_vertices"], dtype=np.float64
                                )
                                joints_published = np.asarray(
                                    sample["smpl_joints"], dtype=np.float64
                                )
                                root_position = joints_published[0].copy()
                                vertices_local = vertices_published - root_position[None, :]
                                joints_local = joints_published - root_position[None, :]
                                root_t_color = root_position
                            elif use_official_params:
                                root_position = joints_recon[0].copy()
                                vertices_local = vertices_recon - root_position[None, :]
                                joints_local = joints_recon - root_position[None, :]
                                root_t_color = root_position
                            elif use_official_transl:
                                root_t_color = np.asarray(
                                    sample["official_transl"], dtype=np.float64
                                ).reshape(3)
                                vertices_local = vertices_recon
                                joints_local = joints_recon
                            else:
                                root_t_color = camera_translation(
                                    sample["pred_cam_t"], args.translation_convention
                                )
                                vertices_local = vertices_recon
                                joints_local = joints_recon
                            already_in_bed = sample.get("publish_frame") == "bed"
                            if args.frame == "bed" and already_in_bed:
                                vertices, joints, root_position = transform_to_camera(
                                    vertices_local,
                                    joints_local,
                                    root_t_color,
                                )
                            elif args.frame == "bed":
                                vertices, joints, root_position = transform_to_bed(
                                    vertices_local,
                                    joints_local,
                                    root_t_color,
                                    color_r,
                                    color_t,
                                )
                            else:
                                vertices, joints, root_position = transform_to_camera(
                                    vertices_local,
                                    joints_local,
                                    root_t_color,
                                )
                            if (
                                use_official_params
                                and sample["official_global_orient"] is not None
                            ):
                                body_r_color = Rotation.from_rotvec(
                                    np.asarray(
                                        sample["official_global_orient"],
                                        dtype=np.float64,
                                    ).reshape(3)
                                ).as_matrix()
                            else:
                                body_r_color = (
                                    Rotation.from_quat(sample["body_quat"]).as_matrix()
                                    if sample["body_quat"] is not None
                                    and sample["body_quat"].size == 4
                                    and not args.no_apply_body_quat
                                    else np.eye(3)
                                )
                            body_r_vis = (
                                body_r_color
                                if already_in_bed
                                else color_r @ body_r_color
                            )

                            if not smpl_geometry_added:
                                faces = faces_now
                                mesh, joint_cloud, skeleton, _frame_unused = create_scene(
                                    o3d, vertices, faces, joints
                                )
                                smpl_pelvis_frame = make_frame_lines(
                                    o3d,
                                    root_position,
                                    body_r_vis,
                                    scale=args.root_frame_size,
                                )
                                vis.add_geometry(mesh)
                                vis.add_geometry(joint_cloud)
                                vis.add_geometry(skeleton)
                                vis.add_geometry(smpl_pelvis_frame)
                                smpl_geometry_added = True
                            else:
                                mesh.vertices = o3d.utility.Vector3dVector(
                                    vertices.astype(np.float64)
                                )
                                mesh.triangles = o3d.utility.Vector3iVector(
                                    faces.astype(np.int32)
                                )
                                mesh.compute_vertex_normals()
                                joint_cloud.points = o3d.utility.Vector3dVector(
                                    joints.astype(np.float64)
                                )
                                skeleton.points = o3d.utility.Vector3dVector(
                                    joints.astype(np.float64)
                                )
                                update_frame_lines(
                                    o3d,
                                    smpl_pelvis_frame,
                                    root_position,
                                    body_r_vis,
                                    scale=args.root_frame_size,
                                )
                                vis.update_geometry(mesh)
                                vis.update_geometry(joint_cloud)
                                vis.update_geometry(skeleton)
                                vis.update_geometry(smpl_pelvis_frame)

                            last_frame_index = sample["frame_index"]
                            now = time.perf_counter()
                            if now >= next_status:
                                print(
                                    "live "
                                    f"frame={last_frame_index} "
                                    f"people={sample['num_people']} "
                                    f"frame_mode={args.frame} "
                                    f"publish_frame={sample.get('publish_frame')} "
                                    f"root={root_position.round(3).tolist()} "
                                    f"source_ts={sample['source_timestamp']} "
                                    f"published_ts={metadata.get('publish_timestamp')}",
                                    flush=True,
                                )
                                next_status = now + 2.0

            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("Stopping live SMPL visualizer.")
    finally:
        for _stream_type, socket, _topic in sockets:
            socket.close(0)
        vis.destroy_window()


if __name__ == "__main__":
    main()
