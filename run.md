## Preprocess Offline Data


- Extract Synced RGB Frames
```bash
python data/scripts/export_kinect_synced_frames.py data/kinect   --cameras kinect_subordinate3 kinect_subordinate4   --reference-camera kinect_subordinate3   --gravity 0 1 0   --output-dir output/kinect_synced/kinect --max-frames 400
```

- Arrange them in Appropriate format for SAM3D Runner
```bash
python data/scripts/kinect_synced_to_multiview.py   --export-dir output/kinect_synced/kinect   --cameras kinect_subordinate3 kinect_subordinate4   --output-dir output/kinect_multiview/kinect
```