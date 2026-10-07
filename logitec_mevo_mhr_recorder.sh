#!/bin/bash
set -e

# Logitech Mevo NDI RGB (rotated portrait) -> SAM 3D Body MHR recorder/publisher.
# Kinect flow (azure_kinect_mhr_recorder.sh) is separate and untouched.

export GPU_HAND_PREP=1
export LAYER_DTYPE=fp32
export SKIP_KEYPOINT_PROMPT=1
export IMG_SIZE=512

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

NDI_MATCH=${NDI_MATCH:-MEVO-2G9TP}
INTRINSICS_PATH=${INTRINSICS_PATH:-./sample_data_logitec_mevo/intri.yml}
READ_FPS=${READ_FPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-./output_mevo_mhr}
SAVE=${SAVE:-0}
PUBLISH=${PUBLISH:-1}
PUBLISH_ENDPOINT=${PUBLISH_ENDPOINT:-tcp://*:5557}
PUBLISH_TOPIC=${PUBLISH_TOPIC:-mevo.mhr}
PUBLISH_FPS=${PUBLISH_FPS:-10}
MHR_PUBLISH_FRAME=${MHR_PUBLISH_FRAME:-bed}
CAMERA_POSE_DIR=${CAMERA_POSE_DIR:-./sample_data_logitec_mevo}
# If CAMERA_POSE_PATH is unset, use the newest timestamped calibration JSON.
if [ -z "${CAMERA_POSE_PATH:-}" ]; then
    LATEST_TIMESTAMPED=$(
        ls -1t "${CAMERA_POSE_DIR}"/camera_poses_wrt_bed_center_????-??-??_??-??-??.json 2>/dev/null | head -n 1 || true
    )
    if [ -n "$LATEST_TIMESTAMPED" ]; then
        CAMERA_POSE_PATH="$LATEST_TIMESTAMPED"
    elif [ -f "${CAMERA_POSE_DIR}/camera_poses_wrt_bed_center.json" ]; then
        CAMERA_POSE_PATH="${CAMERA_POSE_DIR}/camera_poses_wrt_bed_center.json"
    else
        echo "ERROR: No Mevo camera pose JSON found in ${CAMERA_POSE_DIR}." >&2
        echo "Run: python calibrate_logitec_mevo_bed_pose.py --also-latest" >&2
        exit 1
    fi
fi
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

echo "NDI match: $NDI_MATCH"
echo "Intrinsics: $INTRINSICS_PATH"
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
echo "Frame orientation: rotated 90° CCW portrait (matches calibration)"

CMD=(
    python run_logitec_mevo_mhr_recorder.py
    --ndi-match "$NDI_MATCH"
    --intrinsics-path "$INTRINSICS_PATH"
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

if [ "${NO_ROTATE:-0}" = "1" ]; then
    CMD+=(--no-rotate)
fi

"${CMD[@]}"
