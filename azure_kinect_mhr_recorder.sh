#!/bin/bash
set -e

# Azure Kinect RGB -> SAM 3D Body MHR recorder/publisher.

export GPU_HAND_PREP=1
export LAYER_DTYPE=fp32
export SKIP_KEYPOINT_PROMPT=1
export IMG_SIZE=384

export USE_COMPILE=1
export USE_COMPILE_BACKBONE=1
export DECODER_COMPILE=1
export COMPILE_MODE=reduce-overhead
export COMPILE_WARMUP_BATCH_SIZES=1

export MHR_USE_CUDA_GRAPH=0
export KEYPOINT_PROMPT_INTERM_INTERVAL=999
export BODY_INTERM_PRED_LAYERS=0,1,2
export HAND_INTERM_PRED_LAYERS=0,1
export MHR_NO_CORRECTIVES=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

DEVICE_ID=${DEVICE_ID:-0}
COLOR_RESOLUTION=${COLOR_RESOLUTION:-3072p}
KINECT_FPS=${KINECT_FPS:-15}
READ_FPS=${READ_FPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-./output_azure_mhr}
SAVE=${SAVE:-0}
PUBLISH=${PUBLISH:-1}
PUBLISH_ENDPOINT=${PUBLISH_ENDPOINT:-tcp://*:5557}
PUBLISH_TOPIC=${PUBLISH_TOPIC:-kinect_master.mhr}
PUBLISH_FPS=${PUBLISH_FPS:-10}
MHR_PUBLISH_FRAME=${MHR_PUBLISH_FRAME:-bed}
CAMERA_POSE_PATH=${CAMERA_POSE_PATH:-./sample_data/camera_poses_wrt_bed_center.json}
MODEL=${MODEL:-facebook/sam-3d-body-dinov3}
LOCAL_CHECKPOINT=${LOCAL_CHECKPOINT:-./checkpoints/sam-3d-body-dinov3}
DETECTOR=${DETECTOR:-yolo_pose}
HAND_BOX_SOURCE=${HAND_BOX_SOURCE:-yolo_pose}
FOCAL_SCALE=${FOCAL_SCALE:-1.0}
CENTER_PRINCIPAL_POINT=${CENTER_PRINCIPAL_POINT:-1}
FRAME_STRIDE=${FRAME_STRIDE:-1}
MAX_FRAMES=${MAX_FRAMES:-0}

if [ -z "${DETECTOR_MODEL:-}" ]; then
    if [ -f ./checkpoints/yolo/yolo11x-pose.engine ]; then
        DETECTOR_MODEL=./checkpoints/yolo/yolo11x-pose.engine
    elif [ -f ./checkpoints/yolo/yolo11m-pose.engine ]; then
        DETECTOR_MODEL=./checkpoints/yolo/yolo11m-pose.engine
    elif [ -f ./checkpoints/yolo/yolo11x-pose.pt ]; then
        DETECTOR_MODEL=./checkpoints/yolo/yolo11x-pose.pt
    elif [ -f ./checkpoints/yolo/yolo11m-pose.pt ]; then
        DETECTOR_MODEL=./checkpoints/yolo/yolo11m-pose.pt
    elif [ -f ./yolo11x-pose.pt ]; then
        DETECTOR_MODEL=./yolo11x-pose.pt
    else
        DETECTOR_MODEL=yolo11m-pose.pt
    fi
fi

echo "Device ID: $DEVICE_ID"
echo "Color resolution: $COLOR_RESOLUTION"
echo "Kinect FPS: $KINECT_FPS"
echo "Read FPS: $READ_FPS"
echo "Publish endpoint: $PUBLISH_ENDPOINT"
echo "Publish topic: $PUBLISH_TOPIC"
echo "Publish FPS: $PUBLISH_FPS"
echo "MHR publish frame: $MHR_PUBLISH_FRAME"
echo "Camera pose path: $CAMERA_POSE_PATH"
echo "Using detector model: $DETECTOR_MODEL"
echo "Output dir: $OUTPUT_DIR"
echo "Save MHR npz: $SAVE"
echo "Publish MHR: $PUBLISH"
echo "Frame stride: $FRAME_STRIDE"
echo "Max frames: $MAX_FRAMES"

CMD=(
    python run_azure_kinect_mhr_recorder.py
    --device-id "$DEVICE_ID"
    --color-resolution "$COLOR_RESOLUTION"
    --kinect-fps "$KINECT_FPS"
    --read-fps "$READ_FPS"
    --output-dir "$OUTPUT_DIR"
    --publish-endpoint "$PUBLISH_ENDPOINT"
    --publish-topic "$PUBLISH_TOPIC"
    --publish-fps "$PUBLISH_FPS"
    --mhr-publish-frame "$MHR_PUBLISH_FRAME"
    --camera-pose-path "$CAMERA_POSE_PATH"
    --model "$MODEL"
    --local-checkpoint "$LOCAL_CHECKPOINT"
    --detector "$DETECTOR"
    --detector-model "$DETECTOR_MODEL"
    --hand-box-source "$HAND_BOX_SOURCE"
    --focal-scale "$FOCAL_SCALE"
    --frame-stride "$FRAME_STRIDE"
    --max-frames "$MAX_FRAMES"
)

if [ "$SAVE" = "1" ]; then
    CMD+=(--save)
fi

if [ "$PUBLISH" = "1" ]; then
    CMD+=(--publish)
fi

if [ "$CENTER_PRINCIPAL_POINT" = "1" ]; then
    CMD+=(--center-principal-point)
fi

if [ -n "${FOCAL:-}" ]; then
    CMD+=(--focal "$FOCAL")
fi

"${CMD[@]}"
