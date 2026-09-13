"""Frame-accurate, 2D-first pose extraction for AvatarStage.

The video-to-VRM renderer needs the dancer's visible pose, not a guessed
camera-space body mesh.  RTMW estimates 133 image-plane landmarks directly;
this module retains the OpenPose body subset used by the Blender bridge.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

_BODY_POINTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
}
_HAND_OFFSETS = {
    "Thumb": 1,
    "Index": 5,
    "Middle": 9,
    "Ring": 13,
    "Little": 17,
}
_HAND_POINTS = {
    **{
        f"{side}HandRoot": root
        for side, root in (("L", 92), ("R", 113))
    },
    **{
        f"{side}{finger}{joint}": root + offset + joint - 1
        for side, root in (("L", 92), ("R", 113))
        for finger, offset in _HAND_OFFSETS.items()
        for joint in range(1, 5)
    },
}
_TRACKED_POINTS = {**_BODY_POINTS, **_HAND_POINTS}
_REQUIRED_BODY_POINTS = frozenset((
    "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
))
_MIN_SCORE = 0.22
_DETAIL_MIN_SCORE = 0.30
_SMOOTH_KERNEL = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float32) / 16.0
_LAST_TRACKED_INDEX = max(_TRACKED_POINTS.values())


def _choose_person(
    points: np.ndarray,
    scores: np.ndarray,
    previous_center: np.ndarray | None,
    frame_diagonal: float,
) -> int | None:
    """Select the stable full-body dancer if a detector yields several people."""

    if points.ndim != 3 or scores.ndim != 2 or not len(points):
        return None
    required = np.asarray(tuple(_REQUIRED_BODY_POINTS), dtype=np.str_)
    required_indices = np.asarray([_BODY_POINTS[name] for name in required], dtype=np.int64)
    confidence = scores[:, required_indices].mean(axis=1)
    hip_indices = np.asarray((_BODY_POINTS["RHip"], _BODY_POINTS["LHip"]), dtype=np.int64)
    centers = points[:, hip_indices].mean(axis=1)
    quality = confidence.copy()
    if previous_center is not None:
        distance = np.linalg.norm(centers - previous_center[None, :], axis=1)
        # This is only an identity tie-breaker.  A clearly better full-body
        # detection wins even if the dancer moves quickly across the frame.
        quality -= 0.20 * np.minimum(1.0, distance / max(frame_diagonal, 1.0))
    candidate = int(np.argmax(quality))
    return candidate if confidence[candidate] >= _MIN_SCORE else None


def _interpolate_and_smooth(series: np.ndarray) -> np.ndarray:
    """Fill short detector gaps, then smooth sub-pixel tracker jitter."""

    count = len(series)
    positions = np.arange(count, dtype=np.float32)
    for axis in range(2):
        values = series[:, axis]
        valid = np.isfinite(values)
        if not valid.any():
            raise RuntimeError("RTMW could not track a required body landmark")
        values[~valid] = np.interp(positions[~valid], positions[valid], values[valid])
        padded = np.pad(values, (2, 2), mode="edge")
        series[:, axis] = np.convolve(padded, _SMOOTH_KERNEL, mode="valid")
    return series


def _optional_detail_track(samples: list[list[float]]) -> np.ndarray | None:
    """Return a stable detail landmark only when it was visibly tracked.

    A hidden hand or profile face must leave the target rig at its rest pose;
    interpolating a largely absent detail track would invent a gesture.
    """

    series = np.asarray(samples, dtype=np.float32)
    if len(series) < 2:
        return None
    visible = np.isfinite(series).all(axis=1)
    return _interpolate_and_smooth(series) if visible.mean() >= 0.65 else None


def _depth_signal(points: dict[str, list[list[float]]]) -> np.ndarray:
    """Estimate only coherent whole-body scale change, in log image scale.

    A dancer turning sideways changes shoulder width but not all torso measures.
    Require shoulder, hip and torso scale to agree before retaining a depth
    signal, which prevents a 2D-only retarget from inventing camera lunges.
    """

    right_shoulder = np.asarray(points["RShoulder"], dtype=np.float32)
    left_shoulder = np.asarray(points["LShoulder"], dtype=np.float32)
    right_hip = np.asarray(points["RHip"], dtype=np.float32)
    left_hip = np.asarray(points["LHip"], dtype=np.float32)
    neck = np.asarray(points["Neck"], dtype=np.float32)
    midhip = (right_hip + left_hip) * 0.5
    measures = np.stack((
        np.linalg.norm(right_shoulder - left_shoulder, axis=1),
        np.linalg.norm(right_hip - left_hip, axis=1),
        np.linalg.norm(neck - midhip, axis=1),
        np.linalg.norm(right_shoulder - right_hip, axis=1),
        np.linalg.norm(left_shoulder - left_hip, axis=1),
    ), axis=1)
    baseline = np.maximum(np.median(measures, axis=0), 1e-4)
    log_scales = np.log(np.maximum(measures / baseline, 1e-4))
    coherent = np.ptp(log_scales, axis=1) <= 0.10
    candidates = np.median(log_scales, axis=1)
    signal = np.zeros(len(measures), dtype=np.float32)
    stable_frames = 0
    previous_candidate: float | None = None
    previous_signal = 0.0
    for index, candidate in enumerate(candidates):
        # A sudden apparent zoom is usually a detector swap or a crop change,
        # not a person moving through the stage.  Keep the last safe position
        # rather than filling the gap with invented in-between motion.
        stable = bool(coherent[index])
        if previous_candidate is not None and abs(float(candidate) - previous_candidate) > 0.025:
            stable = False
        if stable:
            stable_frames += 1
            previous_candidate = float(candidate)
        else:
            stable_frames = 0
            previous_candidate = None
        if stable_frames >= 5:
            # Ignore sub-two-percent image-scale noise, then use a causal
            # low-pass.  This is deliberately a fixed-camera 2.5D effect,
            # never a claim of recovered metric depth.
            if abs(float(candidate)) <= 0.02:
                candidate = 0.0
            elif candidate > 0.0:
                candidate -= 0.02
            else:
                candidate += 0.02
            previous_signal += 0.14 * (float(candidate) - previous_signal)
        signal[index] = previous_signal
    # This is a source-relative, unitless scale signal. Blender calibrates it
    # to each avatar's height and applies a second, physical displacement cap.
    return np.clip(signal - signal[0], -0.20, 0.20)


def extract_pose_2d(video_path: Path, output_path: Path, *, check_interrupt=None) -> dict[str, Any]:
    """Write a stable OpenPose-body motion file from the actual source video.

    The ONNX sessions are local variables on purpose.  AvatarStage jobs do not
    keep the detector or pose model resident after their motion JSON is written.
    """

    try:
        import cv2
        from rtmlib import PoseTracker, Wholebody
    except ImportError as exc:  # pragma: no cover - exercised in installed runtime
        raise RuntimeError("AvatarStage requires the installed RTMW 2D pose runtime") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("could not read AvatarStage source video for 2D pose extraction")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("AvatarStage source video has invalid dimensions")

    # RTMW's 133-point whole-body model, converted to the familiar OpenPose
    # ordering.  Detection is refreshed every ten frames; tracking supplies
    # frame-accurate motion between refreshes without re-detecting the room.
    tracker = PoseTracker(
        Wholebody,
        det_frequency=10,
        tracking=True,
        to_openpose=True,
        backend="onnxruntime",
        device="cpu",
    )
    tracked: dict[str, list[list[float]]] = {name: [] for name in _TRACKED_POINTS}
    previous_center: np.ndarray | None = None
    frame_diagonal = float(np.hypot(width, height))
    try:
        while True:
            if check_interrupt is not None:
                check_interrupt()
            ok, frame = capture.read()
            if not ok:
                break
            points, scores = tracker(frame)
            if points.ndim != 3 or scores.ndim != 2 or points.shape[-1] < 2 or points.shape[1] <= _LAST_TRACKED_INDEX or scores.shape[1] <= _LAST_TRACKED_INDEX:
                raise RuntimeError("RTMW did not return the expected 134-point OpenPose layout")
            person = _choose_person(points, scores, previous_center, frame_diagonal)
            if person is None:
                for samples in tracked.values():
                    samples.append([float("nan"), float("nan")])
                continue
            selected = points[person]
            selected_scores = scores[person]
            hips = selected[[_BODY_POINTS["RHip"], _BODY_POINTS["LHip"]]]
            previous_center = hips.mean(axis=0)
            for name, point_index in _TRACKED_POINTS.items():
                threshold = _MIN_SCORE if name in _REQUIRED_BODY_POINTS else _DETAIL_MIN_SCORE
                if float(selected_scores[point_index]) >= threshold:
                    tracked[name].append([float(selected[point_index, 0]), float(selected[point_index, 1])])
                else:
                    tracked[name].append([float("nan"), float("nan")])
    finally:
        capture.release()
        # ONNX sessions own native allocations; release them before Blender
        # starts so completed jobs do not leave an inference model resident.
        del tracker
        gc.collect()

    if not tracked["Neck"] or len(tracked["Neck"]) < 2:
        raise RuntimeError("RTMW extracted fewer than two AvatarStage pose frames")
    points_payload: dict[str, list[list[float]]] = {}
    for name, samples in tracked.items():
        if name in _REQUIRED_BODY_POINTS:
            smoothed = _interpolate_and_smooth(np.asarray(samples, dtype=np.float32))
        else:
            smoothed = _optional_detail_track(samples)
        if smoothed is not None:
            points_payload[name] = [[round(float(x), 4), round(float(y), 4)] for x, y in smoothed]
    # The midpoint is the reliable image-plane root signal.  Do not substitute
    # monocular camera depth: a source dancer who stays grounded must not make
    # the avatar float toward or away from a virtual camera.
    root = (np.asarray(points_payload["RHip"], dtype=np.float32) + np.asarray(points_payload["LHip"], dtype=np.float32)) * 0.5
    points_payload["Root"] = [[round(float(x), 4), round(float(y), 4)] for x, y in root]
    depth = _depth_signal(points_payload)
    points_payload["Depth"] = [[round(float(value), 6), 0.0] for value in depth]
    payload = {
        "schema": 2,
        "kind": "rtmw_2d_openpose_body",
        "fps": fps,
        "width": width,
        "height": height,
        "frames": len(points_payload["Root"]),
        "points": points_payload,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        "kind": payload["kind"],
        "fps": fps,
        "width": width,
        "height": height,
        "frames": payload["frames"],
    }
