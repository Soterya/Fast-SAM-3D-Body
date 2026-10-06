import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# smplx/chumpy compatibility for modern Python/NumPy.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _alias, _target in (
    ("bool", np.bool_),
    ("int", np.int_),
    ("float", np.float64),
    ("complex", np.complex128),
    ("object", np.object_),
    ("str", np.str_),
    ("unicode", np.str_),
):
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)

import smplx  # noqa: E402


SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def import_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Open3D is not installed in this environment.\n"
            "Install it with:\n"
            "  pip install open3d\n"
            "or, from your conda env:\n"
            "  /home/rutwik/miniconda3/envs/fast_sam_3d_body/bin/python -m pip install open3d"
        ) from exc
    return o3d


def load_npz(path):
    data = np.load(path)
    required = ("smpl_pose", "betas")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"{path} is missing required key(s): {missing}")

    smpl_pose = np.asarray(data["smpl_pose"], dtype=np.float32)
    if smpl_pose.shape == (63,):
        smpl_pose = smpl_pose.reshape(21, 3)
    if smpl_pose.shape != (21, 3):
        raise RuntimeError(f"Expected smpl_pose shape (21, 3), got {smpl_pose.shape}")

    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    body_quat = np.asarray(data["body_quat_xyzw"], dtype=np.float32) if "body_quat_xyzw" in data else None
    smpl_joints = np.asarray(data["smpl_joints"], dtype=np.float32) if "smpl_joints" in data else None
    return smpl_pose, betas, body_quat, smpl_joints


def smpl_forward(smpl_model_path, smpl_pose, betas, body_quat, apply_body_quat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = smplx.SMPL(model_path=str(smpl_model_path), gender="neutral", batch_size=1).to(device)
    smpl_model.eval()

    body_pose_np = np.zeros((23, 3), dtype=np.float32)
    body_pose_np[:21] = smpl_pose

    num_betas = int(getattr(smpl_model, "num_betas", 10))
    betas_np = np.zeros((num_betas,), dtype=np.float32)
    betas_np[: min(num_betas, betas.shape[0])] = betas[:num_betas]

    if apply_body_quat and body_quat is not None and body_quat.size == 4:
        global_orient_np = Rotation.from_quat(body_quat.reshape(4)).as_rotvec().astype(np.float32)
    else:
        global_orient_np = np.zeros((3,), dtype=np.float32)

    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.from_numpy(global_orient_np.reshape(1, 3)).to(device),
            body_pose=torch.from_numpy(body_pose_np.reshape(1, 69)).to(device),
            betas=torch.from_numpy(betas_np.reshape(1, num_betas)).to(device),
        )

    vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0, :24].detach().cpu().numpy().astype(np.float32)
    root = joints[0:1].copy()
    vertices -= root
    joints -= root
    faces = np.asarray(smpl_model.faces, dtype=np.int32)
    return vertices, joints, faces


def make_joint_geometries(o3d, joints):
    geometries = []
    points = o3d.utility.Vector3dVector(joints.astype(np.float64))
    lines = []
    for idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx >= 0:
            lines.append([parent_idx, idx])
    line_set = o3d.geometry.LineSet(
        points=points,
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.08, 0.08, 0.08]], dtype=np.float64), (len(lines), 1))
    )
    geometries.append(line_set)

    for joint in joints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.018)
        sphere.translate(joint.astype(np.float64))
        sphere.paint_uniform_color([0.95, 0.18, 0.12])
        geometries.append(sphere)
    return geometries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="output_kinect_live/latest_smpl_000.npz")
    parser.add_argument("--smpl_model_path", default="mhr2smpl/data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--apply_body_quat", action="store_true")
    parser.add_argument("--no_joints", action="store_true")
    args = parser.parse_args()

    o3d = import_open3d()
    npz_path = Path(args.npz)
    smpl_model_path = Path(args.smpl_model_path)
    if not npz_path.is_file():
        raise RuntimeError(f"SMPL npz not found: {npz_path}")
    if not smpl_model_path.is_file():
        raise RuntimeError(f"SMPL model not found: {smpl_model_path}")

    smpl_pose, betas, body_quat, saved_joints = load_npz(npz_path)
    vertices, joints, faces = smpl_forward(
        smpl_model_path,
        smpl_pose,
        betas,
        body_quat,
        args.apply_body_quat,
    )

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.62, 0.74, 0.9])

    geometries = [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35)]
    if not args.no_joints:
        geometries.extend(make_joint_geometries(o3d, joints))

    print(f"Loaded: {npz_path}")
    print(f"vertices={vertices.shape} faces={faces.shape} joints={joints.shape}")
    if saved_joints is not None:
        print(f"saved_smpl_joints={saved_joints.shape}")
    print("Tip: use --apply_body_quat to view with the publisher-style global body orientation.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=str(npz_path),
        width=1000,
        height=800,
    )


if __name__ == "__main__":
    main()
