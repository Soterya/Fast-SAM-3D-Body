## Preprocess Offline Data


- Extract Synced RGB Frames
```bash
python data/scripts/export_kinect_synced_frames.py data/kinect   --cameras kinect_subordinate3 kinect_subordinate4   --reference-camera kinect_subordinate3   --gravity 0 1 0   --output-dir output/kinect_synced/kinect
```

- Arrange them in Appropriate format for SAM3D Runner
```bash
python data/scripts/kinect_synced_to_multiview.py   --export-dir output/kinect_synced/kinect   --cameras kinect_subordinate3 kinect_subordinate4   --output-dir output/kinect_multiview/kinect
```

- Run the Pipeline for SMPL Results (in the camera frame)
```bash
python run_multiview_publisher.py   --source video   --videos output/kinect_multiview/kinect/kinect_subordinate3.mp4            output/kinect_multiview/kinect/kinect_subordinate4.mp4   --intrinsics output/kinect_multiview/kinect/kinect_subordinate3.json                output/kinect_multiview/kinect/kinect_subordinate4.json   --main-camera 0   --smpl-model-path mhr2smpl/data/SMPL_NEUTRAL.pkl   --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500   --mhr2smpl-mapping-path mhr2smpl/data/mhr2smpl_mapping.npz   --mhr-mesh-path mhr2smpl/data/mhr_face_mask.ply   --smoother-dir mhr2smpl/experiments/smoother_w5   --no-loop --record
```

- Visualize the results
```bash
python view_smpl_record.py   --npz output/records/2026-10-05_19-07-30/smpl_data.npz   --smpl-model-path mhr2smpl/data/SMPL_NEUTRAL.pkl
```



## Running the Logitec-MEVO Pipeline
```bash
# MHR Recorder and Publisher
bash ./logitec_mevo_mhr_recorder.sh 

# Optional Visualizer
python view_smpl_zmq_open3d.py --input_type mhr --mhr_topic mevo.mhr --frame bed --camera_pose_path sample_data_logitec_mevo/camera_poses_wrt_bed_center.json
```
