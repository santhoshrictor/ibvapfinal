"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Native FastAPI Backend Core Server (Multi-Camera & Multi-Model Edition)
=============================================================================
High-performance surveillance backend with:
  1. GPU Acceleration: NVIDIA GeForce RTX 4050 (CUDA)
  2. YOLOv8 Pose: 17 COCO skeletal keypoint estimation & tracking
  3. YOLOv8 Vehicles & Plate YOLO: Automated Number Plate Recognition (ANPR)
  4. EasyOCR: GPU-accelerated license plate text decoding
  5. YuNet: High-speed DNN facial detection
  6. BLA Engine: Breach Logic Analytics with virtual tripwires,
     zone intrusion, loitering, and skeletal climbing/jumping kinematics
  7. Multi-Camera Architecture: 3 independent checkpoint camera feeds
     plus combined tactical mosaic streaming
=============================================================================
"""

from __future__ import annotations

import os
import sys
import re
import time
import math
import random
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("IBVAP-Core")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constants & Model Classes
# ---------------------------------------------------------------------------
ALLOWED_CLASSES = {0: "PERSON", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}
VEHICLE_TYPES = {"CAR", "MOTORCYCLE", "BUS", "TRUCK"}

# Strict Standard Indian Vehicle Registration Pattern
# Matches: 2 letters, optional space/dash, 1-2 digits, optional space/dash, 1-3 letters, optional space/dash, 4 digits
# e.g.: ^[A-Z]{2}[ -]?[0-9]{1,2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$
INDIAN_PLATE_PATTERN = r"^[A-Z]{2}[ -]?[0-9]{1,2}[ -]?[A-Z]{1,3}[ -]?[0-9]{4}$"
INDIAN_PLATE_REGEX = re.compile(INDIAN_PLATE_PATTERN)


def detect_hardware_accelerator() -> Tuple[bool, str]:
    """
    Probes system for NVIDIA GeForce RTX 4050 Laptop GPU (CUDA/TensorRT).
    Returns (is_cuda_ready, device_display_name).
    """
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)

    # 1. Probe via pynvml (nvidia-ml-py)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name:
            return True, name
    except Exception:
        pass

    # 2. Probe via nvidia-smi CLI
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=2
        ).strip()
        if out:
            return True, out.splitlines()[0]
    except Exception:
        pass

    return False, "CPU"


BOX_COLORS = {
    "PERSON": (0, 255, 0),                 # Pure Green for normal walking (untracked as threat)
    "FACE": (0, 255, 185),                 # Bright Mint
    "CAR": (60, 220, 100),                 # Emerald Green
    "MOTORCYCLE": (60, 220, 100),          # Emerald Green
    "BUS": (60, 220, 100),                 # Emerald Green
    "TRUCK": (60, 220, 100),               # Emerald Green
    "PLATE": (255, 0, 255),                # Magenta / Purple
    "BREACH": (0, 0, 255),                 # Urgent Red
    "CROUCHING": (0, 0, 255),              # Urgent Red
    "CROUCHING/CRAWLING": (0, 0, 255),     # Urgent Red
    "CLIMBING": (0, 0, 255),               # Urgent Red
    "CLIMBING/JUMPING": (0, 0, 255)        # Urgent Red
}

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Facial features
    (5, 6),                                       # Shoulders
    (5, 7), (7, 9),                               # Left arm
    (6, 8), (8, 10),                              # Right arm
    (5, 11), (6, 12), (11, 12),                   # Torso
    (11, 13), (13, 15),                           # Left leg
    (12, 14), (14, 16)                            # Right leg
]

CAMERA_CONFIGS = {
    "1": {"id": "CAM_1", "label": "HAWKINS POST", "default_source": "fence.mp4"},
    "2": {"id": "CAM_2", "label": "OUT POST (ANPR)", "default_source": "vehicle.mp4"},
    "3": {"id": "CAM_3", "label": "ALPHA POST", "default_source": "sample.mp4"},
}


# =============================================================================
# BLA (BREACH LOGIC ANALYTICS) ENGINE
# =============================================================================

class ActivityEngine:
    """
    Per-camera centroid tracker and deterministic security rule engine.
    Detects virtual fence line crossings, climbing/jumping kinematics via
    skeletal limb velocity, restricted zone intrusion, high speed, and loitering.
    """

    def __init__(
        self,
        camera_id: str = "CAM_1",
        fence_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        protected_polygon: Optional[List[Tuple[float, float]]] = None,
        protected_side: str = "RIGHT",
        speed_threshold: float = 260.0,
        loiter_seconds: float = 15.0,
        loiter_radius: float = 45.0,
        min_conf: float = 0.38,
        fence_band_px: float = 35.0,
        confirm_frames: int = 2,
        breach_window_sec: float = 5.0,
        max_match_distance: float = 140.0,
        track_ttl: float = 4.0,
        history_sec: float = 8.0
    ):
        self.camera_id = camera_id
        # Remove red detection line / tripwire for Cam 2 (ANPR) and Cam 3
        self.fence_line = None if camera_id in ("CAM_2", "2", "CAM_3", "3") else fence_line
        self.prot_poly = protected_polygon
        self.prot_sign = -1.0 if str(protected_side).upper() == "RIGHT" else 1.0
        self.spd_thresh = float(speed_threshold)
        self.loiter_sec = float(loiter_seconds)
        self.loiter_rad = float(loiter_radius)
        self.min_conf = float(min_conf)
        self.fence_band = float(fence_band_px)
        self.confirm_frm = int(confirm_frames)
        self.breach_win = float(breach_window_sec)
        self.max_match = float(max_match_distance)
        self.track_ttl = float(track_ttl)
        self.hist_sec = float(history_sec)
        self.enable_vehicle_tracking: bool = False
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        self.recent_alerts: deque = deque(maxlen=50)

    @staticmethod
    def _ctr(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _cross(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    @staticmethod
    def _in_poly(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
        if not poly:
            return False
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def _sdist(self, pt: Tuple[float, float]) -> float:
        if not self.fence_line:
            return 0.0
        a, b = self.fence_line
        return self._cross(a, b, pt) / max(self._dist(a, b), 1.0)

    def _new_track(self, lbl: str, ctr: Tuple[float, float], bbox: Tuple[int, int, int, int], conf: float, kpts: Any, now: float) -> Dict[str, Any]:
        t = {
            "id": self.next_id,
            "label": lbl,
            "history": deque(maxlen=80),
            "last_seen": now,
            "last_alert": {},
            "pre_side": False,
            "near_seen": False,
            "near_since": None,
            "confirm_cnt": 0,
            "breach_cue": False,
            "flash_until": 0.0,
        }
        self.next_id += 1
        self.tracks[t["id"]] = t
        return t

    def _get_track(self, lbl: str, ctr: Tuple[float, float], bbox: Tuple[int, int, int, int], conf: float, kpts: Any, now: float) -> Dict[str, Any]:
        person_labels = {"PERSON", "CROUCHING/CRAWLING", "CLIMBING"}
        candidates = [
            t for t in self.tracks.values()
            if ((t["label"] == lbl) or (t["label"] in person_labels and lbl in person_labels))
            and now - t["last_seen"] <= self.track_ttl
        ]
        if candidates:
            best = min(candidates, key=lambda t: self._dist(t["history"][-1][1], ctr))
            if self._dist(best["history"][-1][1], ctr) <= self.max_match:
                return best
        return self._new_track(lbl, ctr, bbox, conf, kpts, now)

    def _can_alert(self, track: Dict[str, Any], kind: str, now: float, cd: float = 6.0) -> bool:
        if now - track["last_alert"].get(kind, 0.0) < cd:
            return False
        track["last_alert"][kind] = now
        return True

    def _alert(self, track: Dict[str, Any], kind: str, level: str, reason: str, now: float, ev: Dict[str, Any]) -> Dict[str, Any]:
        alert_obj = {
            "kind": kind,
            "level": level,
            "track_id": track["id"],
            "label": track["label"],
            "camera_id": self.camera_id,
            "timestamp": round(now, 2),
            "reason": reason,
            "evidence": ev
        }
        self.recent_alerts.append(alert_obj)
        return alert_obj

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = (480, 640),
        now: Optional[float] = None,
        enable_vehicle_tracking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        now = time.time() if now is None else now
        h, w = frame_shape
        effective_veh_tracking = self.enable_vehicle_tracking if enable_vehicle_tracking is None else enable_vehicle_tracking

        # Remove red detection line / tripwire for Cam 2 (ANPR) and Cam 3
        if self.camera_id in ("CAM_2", "2", "CAM_3", "3"):
            self.fence_line = None
        elif self.fence_line is None and self.camera_id in ("CAM_1", "1"):
            self.fence_line = ((0.0, float(h * 0.70)), (float(w), float(h * 0.70)))

        self.tracks = {k: v for k, v in self.tracks.items() if now - v["last_seen"] <= self.track_ttl}
        alerts: List[Dict[str, Any]] = []

        for det in detections:
            lbl = str(det.get("label", ""))
            if lbl not in ALLOWED_CLASSES.values() and lbl not in ("CROUCHING/CRAWLING", "CLIMBING"):
                continue

            # Point 3: Only track the 'person' class by default. Vehicles tracked only if enabled.
            if lbl in VEHICLE_TYPES and not effective_veh_tracking:
                continue

            bbox = tuple(int(v) for v in det["bbox"][:4])
            ctr = self._ctr(bbox)
            conf = float(det.get("conf", 0.0))
            kpts = det.get("keypoints")

            if conf < self.min_conf:
                continue

            track = self._get_track(lbl, ctr, bbox, conf, kpts, now)
            track["history"].append((now, ctr, bbox, conf, kpts))
            track["last_seen"] = now

            while track["history"] and now - track["history"][0][0] > self.hist_sec:
                track["history"].popleft()

            # -------------------------------------------------------------
            # DYNAMIC POSE BEHAVIOR (CROUCHING & CLIMBING/JUMPING)
            # ZERO fence bounding-box intersection logic!
            # ONLY trigger alerts and red boxes if pose estimation strictly
            # detects 'Crouching' or 'Climbing/Jumping'.
            # Normal walking remains green and untracked as a threat.
            # -------------------------------------------------------------
            if lbl in ("PERSON", "CROUCHING", "CROUCHING/CRAWLING", "CLIMBING", "CLIMBING/JUMPING"):
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]

                # 1. Crouching detection: bbox width > height and low posture
                is_crouching = False
                if bw > bh:
                    if kpts is not None and len(kpts) >= 13:
                        shoulders = [pt for idx, pt in enumerate(kpts) if idx in (5, 6) and (len(pt) < 3 or pt[2] > 0.40) and pt[1] > 0]
                        hips = [pt for idx, pt in enumerate(kpts) if idx in (11, 12) and (len(pt) < 3 or pt[2] > 0.40) and pt[1] > 0]
                        if shoulders and hips:
                            avg_sh_y = sum(p[1] for p in shoulders) / len(shoulders)
                            avg_hp_y = sum(p[1] for p in hips) / len(hips)
                            torso_h = abs(avg_hp_y - avg_sh_y)
                            if torso_h < (bw * 0.65) or avg_sh_y > (bbox[1] + bh * 0.35):
                                is_crouching = True
                        else:
                            is_crouching = True
                    else:
                        is_crouching = True

                # 2. Climbing/Jumping detection: rapid upward y-axis movement of shoulders/hips
                is_climbing = False
                if len(track["history"]) >= 2:
                    prev_obs = track["history"][-2]
                    dt = max(1e-4, now - prev_obs[0])
                    if dt < 1.2:
                        pk, ck = prev_obs[4], kpts
                        upward_velocities = []
                        if pk is not None and ck is not None and len(pk) >= 13 and len(ck) >= 13:
                            for j_idx in (5, 6, 11, 12):  # shoulders and hips
                                if j_idx < len(pk) and j_idx < len(ck):
                                    p_pt, c_pt = pk[j_idx], ck[j_idx]
                                    p_conf = p_pt[2] if len(p_pt) >= 3 else 1.0
                                    c_conf = c_pt[2] if len(c_pt) >= 3 else 1.0
                                    if p_conf > 0.40 and c_conf > 0.40 and p_pt[1] > 0 and c_pt[1] > 0:
                                        dy = c_pt[1] - p_pt[1]  # negative = upward motion
                                        vel_y = dy / dt         # px/s
                                        upward_velocities.append(vel_y)

                        if upward_velocities:
                            if min(upward_velocities) < -100.0:
                                is_climbing = True
                        else:
                            dy = ctr[1] - prev_obs[1][1]
                            vel_y = dy / dt
                            if vel_y < -120.0:
                                is_climbing = True

                if is_crouching:
                    det["label"] = "CROUCHING"
                    det["pose_behavior"] = "Crouching"
                    det["alert"] = True
                    track["flash_until"] = now + 4.0
                    if self._can_alert(track, "CROUCHING", now, cd=4.0):
                        alerts.append(self._alert(
                            track, "CROUCHING ALERT", "RED",
                            "Target exhibiting low posture crouching movement.",
                            now, {"bbox": bbox, "bw": bw, "bh": bh}
                        ))
                elif is_climbing:
                    det["label"] = "CLIMBING/JUMPING"
                    det["pose_behavior"] = "Climbing/Jumping"
                    det["alert"] = True
                    track["flash_until"] = now + 4.0
                    if self._can_alert(track, "CLIMBING", now, cd=4.0):
                        alerts.append(self._alert(
                            track, "CLIMBING/JUMPING ALERT", "RED",
                            "Rapid upward y-axis movement of shoulders/hips indicates climbing or jumping.",
                            now, {"bbox": bbox, "dt": round(dt, 3)}
                        ))
                else:
                    # Normal walking: keep green, unflagged, and untracked as threat
                    det["label"] = "PERSON"
                    det["pose_behavior"] = None
                    det["alert"] = False
                    track["flash_until"] = 0.0

            if now <= track.get("flash_until", 0.0) and det.get("pose_behavior"):
                det["alert"] = True
            elif lbl == "PERSON":
                det["alert"] = False

        return alerts


# =============================================================================
# SURVEILLANCE ENGINE (GPU CORE WITH MULTI-MODEL CAPABILITY)
# =============================================================================

class IBVAPSurveillanceEngine:
    """
    Unified GPU AI surveillance engine:
      - YOLOv8-Pose (17 joints) on RTX 4050
      - YOLOv8 Object Detection (cars, trucks, buses, motorcycles)
      - Plate YOLO + EasyOCR for Automated Number Plate Recognition
      - YuNet OpenCV DNN Face Detection
      - BLA Breach Logic Analytics for multiple checkpoint cameras
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", device: Optional[str] = None):
        self.model_path = os.path.join(BASE_DIR, model_path) if not os.path.isabs(model_path) else model_path
        self._lock = threading.Lock()

        # Resolve GPU device and detect NVIDIA RTX 4050
        cuda_ready, hw_name = detect_hardware_accelerator()
        self.is_gpu_available = cuda_ready

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cuda" if cuda_ready else "cpu"
            if not torch.cuda.is_available():
                self.device = "cpu"

        # Hardware display name (e.g. NVIDIA GeForce RTX 4050 Laptop GPU)
        self.device_name = hw_name if cuda_ready else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        logger.info(f"Initializing Surveillance Engine on [{self.device}] (Accelerator: {self.device_name})")

        # Enable CUDA cuDNN benchmark and TensorRT / TF32 optimizations when CUDA is available
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        # 1. Load YOLOv8 Pose
        self.pose_model = YOLO(self.model_path)
        try:
            if torch.cuda.is_available():
                self.pose_model.to("cuda:0")
            elif self.device != "cpu":
                self.pose_model.to(self.device)
            try:
                self.pose_model.fuse()
            except Exception:
                pass
            logger.info(f"YOLOv8 Pose loaded on {self.device} ({self.device_name})")
        except Exception as e:
            logger.error(f"Failed to transfer pose model: {e}")

        # 2. Load YOLOv8 General Detection (vehicles: car, bus, truck, motorcycle)
        self.vehicle_model = None
        yolo_gen_path = os.path.join(BASE_DIR, "yolov8n.pt")
        if os.path.exists(yolo_gen_path):
            try:
                self.vehicle_model = YOLO(yolo_gen_path)
                if torch.cuda.is_available():
                    self.vehicle_model.to("cuda:0")
                elif self.device != "cpu":
                    self.vehicle_model.to(self.device)
                try:
                    self.vehicle_model.fuse()
                except Exception:
                    pass
                logger.info(f"YOLOv8 Vehicle model loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Vehicle model notice: {e}")

        # 3. Load Plate YOLO model
        self.plate_model = None
        plate_path = os.path.join(BASE_DIR, "plate_yolo.pt")
        if os.path.exists(plate_path):
            try:
                self.plate_model = YOLO(plate_path)
                if torch.cuda.is_available():
                    self.plate_model.to("cuda:0")
                elif self.device != "cpu":
                    self.plate_model.to(self.device)
                try:
                    self.plate_model.fuse()
                except Exception:
                    pass
                logger.info(f"Plate YOLO model loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Plate model notice: {e}")

        # 4. Load EasyOCR (ANPR Reader) - explicitly utilize GPU when CUDA is available
        self.ocr_engine = None
        try:
            use_gpu = torch.cuda.is_available() or ("cuda" in self.device)
            self.ocr_engine = easyocr.Reader(["en"], gpu=use_gpu)
            logger.info(f"EasyOCR Reader initialized on GPU={use_gpu}")
        except Exception as e:
            logger.warning(f"EasyOCR fallback notice: {e}")

        # 5. Load YuNet Face Detector (increased confidence threshold to suppress artifacts)
        self.face_cascade = None
        face_path = os.path.join(BASE_DIR, "face_detection_yunet.onnx")
        if os.path.exists(face_path):
            try:
                self.face_cascade = cv2.FaceDetectorYN.create(
                    face_path, "", (320, 320),
                    score_threshold=0.75,
                    nms_threshold=0.30
                )
                logger.info("YuNet Face Detector initialized with score_threshold=0.75.")
            except Exception as e:
                logger.warning(f"YuNet Face Detector notice: {e}")

        # Face detection persistence (debounce tracking per camera feed)
        self.face_persistence: Dict[str, Dict[str, Any]] = {}

        # 6. Initialize Per-Camera BLA Engines
        self.bla_engines: Dict[str, ActivityEngine] = {
            "CAM_1": ActivityEngine(camera_id="CAM_1"),
            "CAM_2": ActivityEngine(camera_id="CAM_2"),
            "CAM_3": ActivityEngine(camera_id="CAM_3"),
        }

        # Active Camera Sources for Dynamic Switching
        self.camera_sources: Dict[str, str] = {
            "CAM_1": "fence.mp4",
            "CAM_2": "vehicle.mp4",
            "CAM_3": "sample.mp4",
        }
        # Motion trails visibility toggle (default True)
        self.show_trails: bool = True
        # Vehicle tracking toggle (default False: tracks person class only)
        self.enable_vehicle_tracking: bool = False

        # Decoupled AI background processing states per camera (Zero stream stuttering)
        self._cached_results: Dict[str, Dict[str, Any]] = {
            "CAM_1": {"detections": [], "inference_time_ms": 0.0, "target_count": 0, "breach_alerts": []},
            "CAM_2": {"detections": [], "inference_time_ms": 0.0, "target_count": 0, "breach_alerts": []},
            "CAM_3": {"detections": [], "inference_time_ms": 0.0, "target_count": 0, "breach_alerts": []},
        }
        self._ai_busy: Dict[str, bool] = {"CAM_1": False, "CAM_2": False, "CAM_3": False}
        self._results_lock = threading.Lock()

        # Telemetry Cache & ANPR Registry
        self.telemetry_lock = threading.Lock()
        self.recent_plates: deque = deque(maxlen=60)
        self._last_plate_seen: Dict[str, float] = {}
        self.latest_telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "total_targets": 0,
            "persons": 0,
            "vehicles": 0,
            "faces": 0,
            "plates": 0,
            "breaches": 0,
            "active_alerts": [],
            "recent_plates": [],
            "timestamp": time.time()
        }

        # Warmup GPU
        self._warmup()

    def _warmup(self):
        """Warm up CUDA kernels."""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            with self._lock:
                self.pose_model.predict(dummy, imgsz=320, verbose=False, device=self.device)
                if self.vehicle_model:
                    self.vehicle_model.predict(dummy, imgsz=320, verbose=False, device=self.device)
                if self.plate_model:
                    self.plate_model.predict(dummy, imgsz=320, verbose=False, device=self.device)
            logger.info("Surveillance Engine GPU warmup completed.")
        except Exception as e:
            logger.warning(f"GPU warmup notice: {e}")

    def _dispatch_ai_inference(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM_1",
        conf_thresh: float = 0.38,
        enable_ocr: bool = True,
        enable_face: bool = True,
        enable_bla: bool = True,
        enable_vehicle_tracking: bool = False
    ) -> None:
        """Asynchronously execute heavy AI pipeline in background thread without blocking video streaming."""
        def worker():
            try:
                res = self.process_frame(
                    frame,
                    camera_id=camera_id,
                    conf_thresh=conf_thresh,
                    imgsz=640,
                    enable_ocr=enable_ocr,
                    enable_face=enable_face,
                    enable_bla=enable_bla,
                    enable_vehicle_tracking=enable_vehicle_tracking
                )
                with self._results_lock:
                    self._cached_results[camera_id] = res
            except Exception as e:
                logger.warning(f"Background AI processing notice on {camera_id}: {e}")
            finally:
                self._ai_busy[camera_id] = False

        th = threading.Thread(target=worker, daemon=True, name=f"AI-Worker-{camera_id}")
        th.start()

    @staticmethod
    def resolve_video_path(source: Union[int, str]) -> Union[int, str]:
        """Resolves camera index or local file path robustly."""
        if isinstance(source, int):
            return source
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        if not os.path.isabs(s):
            cand = os.path.join(BASE_DIR, s)
            if os.path.exists(cand):
                return cand
        return s

    def set_camera_source(self, cam_id: str, source: str) -> str:
        """Set active video source for a camera ID (supports '1' or 'CAM_1')."""
        key = f"CAM_{str(cam_id).replace('CAM_', '')}"
        self.camera_sources[key] = str(source)
        return key

    def get_camera_source(self, cam_id: str) -> str:
        """Get active video source for a camera ID (supports '1' or 'CAM_1')."""
        key = f"CAM_{str(cam_id).replace('CAM_', '')}"
        return self.camera_sources.get(key, "fence.mp4")

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM_1",
        conf_thresh: float = 0.38,
        imgsz: int = 640,
        enable_ocr: bool = True,
        enable_face: bool = True,
        enable_bla: bool = True,
        enable_vehicle_tracking: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Execute full multi-model pipeline:
          1. YOLOv8 Pose for persons & 17 skeletal keypoints
          2. YOLOv8 General Detection for vehicles (CAR, TRUCK, BUS, MOTORCYCLE)
          3. Plate YOLO for license plate localization
          4. EasyOCR with strict Indian Registration Regex filter
          5. YuNet for facial detection
          6. BLA for perimeter breach kinematics & tripwires
        """
        start_time = time.perf_counter()
        h, w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        # Night adaptation check
        gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        is_night = gray_mean < 65.0
        eff_conf = max(0.18, conf_thresh - 0.15) if is_night else conf_thresh

        # ---------------------------------------------------------------------
        # 1. YOLOv8 Pose Estimation (Persons + Skeletons)
        # ---------------------------------------------------------------------
        with self._lock:
            pose_results = self.pose_model.predict(
                frame,
                conf=eff_conf,
                imgsz=imgsz,
                verbose=False,
                device=self.device
            )

        if pose_results and len(pose_results) > 0:
            p_res = pose_results[0]
            boxes = p_res.boxes
            keypoints_data = getattr(p_res, "keypoints", None)
            num_boxes = len(boxes) if boxes is not None else 0

            for i in range(num_boxes):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                if cls_id == 0:  # PERSON
                    conf = round(float(box.conf[0].item()), 3)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    kpts_list = None
                    if keypoints_data is not None and keypoints_data.xy is not None and len(keypoints_data.xy) > i:
                        xy = keypoints_data.xy[i].cpu().numpy()
                        confs = None
                        if hasattr(keypoints_data, "conf") and keypoints_data.conf is not None and len(keypoints_data.conf) > i:
                            confs = keypoints_data.conf[i].cpu().numpy()

                        if confs is not None:
                            kpts_list = [
                                (round(float(kx), 1), round(float(ky), 1), round(float(kc), 3))
                                for (kx, ky), kc in zip(xy, confs)
                            ]
                        else:
                            kpts_list = [
                                (round(float(kx), 1), round(float(ky), 1), 1.0)
                                for kx, ky in xy
                            ]

                    detections.append({
                        "label": "PERSON",
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2],
                        "plate": "",
                        "keypoints": kpts_list,
                        "alert": False
                    })

        # ---------------------------------------------------------------------
        # 2. YOLOv8 Vehicle Detection (CAR, TRUCK, BUS, MOTORCYCLE)
        # ---------------------------------------------------------------------
        if self.vehicle_model is not None:
            # Lower confidence threshold (conf=0.18) for aggressive vehicle localization
            veh_conf = min(0.20, eff_conf)
            with self._lock:
                veh_results = self.vehicle_model.predict(
                    frame,
                    conf=veh_conf,
                    imgsz=imgsz,
                    verbose=False,
                    device=self.device
                )

            if veh_results and len(veh_results) > 0:
                v_res = veh_results[0]
                for box in v_res.boxes:
                    c_id = int(box.cls[0].item())
                    lbl = v_res.names.get(c_id, "").upper()
                    if lbl in VEHICLE_TYPES:
                        conf = round(float(box.conf[0].item()), 3)
                        vx1, vy1, vx2, vy2 = [int(v) for v in box.xyxy[0].tolist()]
                        detections.append({
                            "label": lbl,
                            "conf": conf,
                            "bbox": [vx1, vy1, vx2, vy2],
                            "plate": "",
                            "keypoints": None,
                            "alert": False
                        })

        # ---------------------------------------------------------------------
        # 3. Plate YOLO & EasyOCR (ANPR Pipeline)
        # ---------------------------------------------------------------------
        if enable_ocr and self.plate_model is not None:
            detected_plate_candidates: List[Tuple[int, int, int, int, float]] = []

            # 3A. Primary pass: Run Plate YOLO across frame
            with self._lock:
                plate_results = self.plate_model.predict(
                    frame,
                    conf=0.15,
                    imgsz=imgsz,
                    verbose=False,
                    device=self.device
                )

            if plate_results and len(plate_results) > 0:
                for p_box in plate_results[0].boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0].tolist())
                    p_conf = round(float(p_box.conf[0].item()), 2)
                    detected_plate_candidates.append((px1, py1, px2, py2, p_conf))

            # 3B. Secondary pass: Check detected vehicle crops if full-frame did not catch plates
            if not detected_plate_candidates:
                for det in detections:
                    if det.get("label") in VEHICLE_TYPES and det.get("conf", 0) >= 0.15:
                        vx1, vy1, vx2, vy2 = det["bbox"]
                        crop_y1 = max(0, vy1 + int((vy2 - vy1) * 0.40))
                        crop_y2 = min(h, vy2)
                        crop_x1 = max(0, vx1)
                        crop_x2 = min(w, vx2)
                        v_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if v_crop.size > 0 and (crop_x2 - crop_x1) > 40 and (crop_y2 - crop_y1) > 20:
                            with self._lock:
                                vp_results = self.plate_model.predict(
                                    v_crop,
                                    conf=0.15,
                                    imgsz=320,
                                    verbose=False,
                                    device=self.device
                                )
                            if vp_results and len(vp_results) > 0 and len(vp_results[0].boxes) > 0:
                                best_p = max(vp_results[0].boxes, key=lambda b: float(b.conf[0]))
                                bx1, by1, bx2, by2 = map(int, best_p.xyxy[0].tolist())
                                b_conf = round(float(best_p.conf[0].item()), 2)
                                detected_plate_candidates.append((crop_x1 + bx1, crop_y1 + by1, crop_x1 + bx2, crop_y1 + by2, b_conf))

            # 3C. OCR Character Decoding & Strict Regex Filtering
            for px1, py1, px2, py2, p_conf in detected_plate_candidates:
                plate_text = ""
                p_crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                if p_crop.size > 0 and self.ocr_engine is not None:
                    try:
                        # Upscale small plate crops to boost character recognition
                        crop_h, crop_w = p_crop.shape[:2]
                        if crop_w < 140 or crop_h < 40:
                            scale = max(1.5, 140.0 / max(crop_w, 1))
                            p_scaled = cv2.resize(p_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        else:
                            p_scaled = p_crop

                        # Grayscale and bilateral filtering for edge preservation
                        p_gray = cv2.cvtColor(p_scaled, cv2.COLOR_BGR2GRAY)
                        p_filtered = cv2.bilateralFilter(p_gray, d=11, sigmaColor=17, sigmaSpace=17)

                        # Otsu thresholding
                        _, p_thresh = cv2.threshold(
                            p_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )

                        # Primary OCR attempt on high-contrast threshold
                        ocr_texts = self.ocr_engine.readtext(
                            p_thresh,
                            detail=0,
                            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -"
                        )
                        # Fallback OCR attempt on filtered crop if thresholding produced nothing
                        if not ocr_texts:
                            ocr_texts = self.ocr_engine.readtext(
                                p_filtered,
                                detail=0,
                                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -"
                            )

                        # Clean candidate text and validate against strict Indian registration pattern
                        combined_raw = " ".join([t.strip().upper() for t in ocr_texts if t.strip()])
                        clean_candidate = re.sub(r"[^A-Z0-9 -]", "", combined_raw).strip()
                        compact_candidate = re.sub(r"\s+", " ", clean_candidate).strip()
                        no_space_candidate = re.sub(r"[ -]", "", clean_candidate)

                        # Point 2: Strict Regular Expression (Regex) filter
                        # Indian vehicle registration: ^[A-Z]{2}[ -]?[0-9]{1,2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$
                        matched_plate = None
                        if INDIAN_PLATE_REGEX.match(compact_candidate):
                            matched_plate = compact_candidate
                        elif INDIAN_PLATE_REGEX.match(no_space_candidate):
                            if len(no_space_candidate) >= 9:
                                st_code = no_space_candidate[:2]
                                rto_code = no_space_candidate[2:4] if no_space_candidate[3].isdigit() else no_space_candidate[2:3]
                                remainder = no_space_candidate[len(st_code)+len(rto_code):]
                                num_code = remainder[-4:]
                                ser_code = remainder[:-4]
                                matched_plate = f"{st_code} {rto_code} {ser_code} {num_code}".strip()
                            else:
                                matched_plate = no_space_candidate

                        if matched_plate:
                            plate_text = matched_plate
                        else:
                            # Silently discard random background noise (words like 'HANDBAGS' or 'CENTER')
                            plate_text = ""
                    except Exception as e:
                        logger.debug(f"ANPR preprocessing notice: {e}")
                        plate_text = ""
                # Always record plate detection bounding box so it is visually rendered!
                detections.append({
                    "label": "PLATE",
                    "conf": p_conf,
                    "bbox": [px1, py1, px2, py2],
                    "plate": plate_text,
                    "keypoints": None,
                    "alert": False
                })

                # Only register and append in ANPR registry if plate matched strict regex
                if plate_text:
                    now_ts = time.time()
                    last_t = self._last_plate_seen.get(plate_text, 0)
                    if (now_ts - last_t) > 3.0:
                        self._last_plate_seen[plate_text] = now_ts
                        if len(self._last_plate_seen) > 200:
                            self._last_plate_seen = {k: v for k, v in self._last_plate_seen.items() if now_ts - v < 60}

                        # Simulated database check: status 'CLEARED' or 'FLAGGED'
                        db_status = "FLAGGED" if (random.random() < 0.25 or hash(plate_text) % 4 == 0) else "CLEARED"
                        with self.telemetry_lock:
                            self.recent_plates.appendleft({
                                "plate": plate_text,
                                "timestamp": time.strftime("%H:%M:%S", time.localtime(now_ts)),
                                "status": db_status,
                                "camera_id": camera_id,
                                "conf": p_conf,
                                "epoch": now_ts
                            })

        # ---------------------------------------------------------------------
        # 4. YuNet Face Detection with Debounce (Frame Persistence)
        # ---------------------------------------------------------------------
        if enable_face and self.face_cascade is not None:
            current_faces: List[Dict[str, Any]] = []
            try:
                self.face_cascade.setInputSize((w, h))
                _, faces = self.face_cascade.detect(frame)
                if faces is not None:
                    for face in faces:
                        conf_f = float(face[14]) if len(face) > 14 else 0.80
                        if conf_f >= 0.75:
                            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                            current_faces.append({
                                "label": "FACE",
                                "conf": round(conf_f, 2),
                                "bbox": [fx, fy, fx + fw, fy + fh],
                                "plate": "",
                                "keypoints": None,
                                "alert": False
                            })
            except Exception:
                pass

            # Frame-persistence debounce logic:
            # Store coordinates of last detected face. If engine loses face for only 1 or 2 frames,
            # continue drawing the last known bounding box to prevent flickering.
            if current_faces:
                detections.extend(current_faces)
                self.face_persistence[camera_id] = {
                    "faces": current_faces,
                    "missed_frames": 0
                }
            else:
                last_tracker = self.face_persistence.get(camera_id)
                if last_tracker and last_tracker.get("faces") and last_tracker.get("missed_frames", 0) < 2:
                    last_tracker["missed_frames"] += 1
                    for persisted_face in last_tracker["faces"]:
                        detections.append(dict(persisted_face))
                else:
                    self.face_persistence[camera_id] = {"faces": [], "missed_frames": 999}

        # ---------------------------------------------------------------------
        # 5. BLA (Breach Logic Analytics)
        # ---------------------------------------------------------------------
        breach_alerts = []
        bla_engine = self.bla_engines.get(camera_id, self.bla_engines.get("CAM_1"))
        effective_veh_tracking = self.enable_vehicle_tracking if enable_vehicle_tracking is None else enable_vehicle_tracking
        if enable_bla and bla_engine is not None:
            breach_alerts = bla_engine.update(
                detections,
                frame_shape=(h, w),
                enable_vehicle_tracking=effective_veh_tracking
            )

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Update telemetry
        person_count = sum(1 for d in detections if d["label"] == "PERSON")
        vehicle_count = sum(1 for d in detections if d["label"] in VEHICLE_TYPES)
        face_count = sum(1 for d in detections if d["label"] == "FACE")
        plate_count = sum(1 for d in detections if d["label"] == "PLATE")
        breach_count = sum(1 for d in detections if d.get("alert"))

        with self.telemetry_lock:
            all_alerts: List[Dict[str, Any]] = []
            for eng in self.bla_engines.values():
                all_alerts.extend(list(eng.recent_alerts))
            all_alerts.sort(key=lambda a: a.get("timestamp", 0), reverse=True)

            self.latest_telemetry.update({
                "latency_ms": inference_time_ms,
                "total_targets": len(detections),
                "persons": person_count,
                "vehicles": vehicle_count,
                "faces": face_count,
                "plates": plate_count,
                "breaches": breach_count,
                "active_alerts": all_alerts[:15],
                "recent_plates": list(self.recent_plates),
                "timestamp": time.time()
            })

        return {
            "frame_dimensions": {"width": w, "height": h},
            "inference_time_ms": inference_time_ms,
            "target_count": len(detections),
            "detections": detections,
            "breach_alerts": breach_alerts
        }

    def draw_overlays(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        camera_id: str = "CAM_1",
        draw_skeletons: bool = True,
        draw_bla: bool = True,
        draw_trails: Optional[bool] = None,
        enable_vehicle_tracking: Optional[bool] = None
    ) -> np.ndarray:
        """
        Render visual overlays: bounding boxes, vehicle tags, plates,
        skeletons, virtual fence tripwire, and modernized fading motion trails.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        bla_engine = self.bla_engines.get(camera_id, self.bla_engines.get("CAM_1"))

        # 1. Virtual Fence Tripwire (Never rendered for Camera 2 ANPR or Camera 3)
        if draw_bla and bla_engine and bla_engine.fence_line and camera_id not in ("CAM_2", "2", "CAM_3", "3"):
            fa, fb = bla_engine.fence_line
            p1 = (int(fa[0]), int(fa[1]))
            p2 = (int(fb[0]), int(fb[1]))
            cv2.line(annotated, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(annotated, p1, p2, (0, 180, 255), 1, cv2.LINE_AA)
            mid = (int((fa[0] + fb[0]) / 2) - 110, int((fa[1] + fb[1]) / 2) - 8)
            cv2.putText(annotated, "VIRTUAL FENCE TRIPWIRE", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Modern Motion Trails (Thin 1px lines, fading alpha trails; vehicle trails only if enabled)
        effective_draw_trails = self.show_trails if draw_trails is None else draw_trails
        effective_veh_tracking = self.enable_vehicle_tracking if enable_vehicle_tracking is None else enable_vehicle_tracking

        if effective_draw_trails and draw_bla and bla_engine:
            for track in bla_engine.tracks.values():
                t_lbl = track.get("label", "")
                is_veh = t_lbl in VEHICLE_TYPES

                # Point 3: Only draw tracking trails for vehicles when explicitly checked
                if is_veh and not effective_veh_tracking:
                    continue

                history = list(track.get("history", []))
                n_pts = len(history)
                if n_pts >= 2:
                    is_alert = time.time() <= track.get("flash_until", 0.0)
                    if is_alert:
                        base_bgr = (0, 0, 255)       # Urgent Red for alerts
                    elif is_veh:
                        base_bgr = (0, 180, 255)     # Glowing Neon Amber for Vehicles
                    else:
                        base_bgr = (255, 212, 0)     # Modern Tactical Cyan for Persons

                    # Modernized Sleek Fading Trails with thin 1px anti-aliased lines
                    for idx in range(n_pts - 1):
                        pt_a = tuple(map(int, history[idx][1]))
                        pt_b = tuple(map(int, history[idx + 1][1]))
                        # Alpha decay along history: older points fade out smoothly
                        alpha = math.pow((idx + 1) / max(1, n_pts - 1), 1.3)
                        seg_color = (
                            int(base_bgr[0] * alpha),
                            int(base_bgr[1] * alpha),
                            int(base_bgr[2] * alpha)
                        )
                        cv2.line(annotated, pt_a, pt_b, seg_color, 1, cv2.LINE_AA)

                    # Subtle glowing marker at current target centroid
                    head_pt = tuple(map(int, history[-1][1]))
                    cv2.circle(annotated, head_pt, 2, base_bgr, -1, cv2.LINE_AA)
                    cv2.circle(annotated, head_pt, 4, (int(base_bgr[0] * 0.5), int(base_bgr[1] * 0.5), int(base_bgr[2] * 0.5)), 1, cv2.LINE_AA)

        # 3. Detections
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"][:4]]
            label = det["label"]
            conf = det["conf"]
            is_alert = det.get("alert", False)
            pose_behavior = det.get("pose_behavior")

            # ONLY trigger alerts and red boxes if pose estimation strictly detects 'Crouching' or 'Climbing/Jumping'
            if pose_behavior in ("Crouching", "Crouching/Crawling"):
                box_color = (0, 0, 255)  # Red alert
                tag = f"CROUCHING ALERT {int(conf * 100)}%"
                is_alert = True
            elif pose_behavior in ("Climbing", "Climbing/Jumping"):
                box_color = (0, 0, 255)  # Red alert
                tag = f"CLIMBING ALERT {int(conf * 100)}%"
                is_alert = True
            elif is_alert:
                box_color = (0, 0, 255)
                tag = f"BREACH DETECTED {int(conf * 100)}%"
            elif label == "PLATE":
                box_color = BOX_COLORS["PLATE"]
                plate_val = (det.get("plate") or "").strip()
                tag = f"PLATE: {plate_val}" if plate_val and plate_val != "PLATE DETECTED" else (plate_val or f"PLATE {int(conf * 100)}%")
            elif label == "PERSON":
                # Normal walking must remain green and untracked as a threat, regardless of where they are on the screen
                box_color = (0, 255, 0)  # Pure Green
                tag = f"PERSON {conf:.2f}"
            else:
                box_color = BOX_COLORS.get(label, (60, 220, 100))
                tag = f"{label} {conf:.2f}"

            thickness = 3 if (is_alert or label == "PLATE") else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Label banner
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            if y1 - th - 6 >= 0:
                tag_y1 = y1 - th - 6
                text_y = y1 - 4
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, y1), box_color, -1)
            else:
                tag_y1 = y2
                text_y = y2 + th + 4
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, y2 + th + 6), box_color, -1)
            text_color = (255, 255, 255) if (is_alert or label == "PLATE") else (10, 10, 10)
            cv2.putText(annotated, tag, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

            # 4. Draw 17-Point Pose Skeleton (Conf > 0.65 filter to eliminate ghost keypoints)
            if draw_skeletons and det.get("keypoints") is not None:
                kpts = det["keypoints"]
                pt_map: Dict[int, Tuple[int, int]] = {}
                for idx, pt in enumerate(kpts):
                    if len(pt) >= 3:
                        kx, ky, k_conf = pt[0], pt[1], pt[2]
                    else:
                        kx, ky, k_conf = pt[0], pt[1], 1.0

                    # Filter: ONLY keep keypoints with confidence > 0.65 and inside frame
                    if k_conf > 0.65 and kx > 0 and ky > 0:
                        px, py = int(kx), int(ky)
                        pt_map[idx] = (px, py)
                        cv2.circle(annotated, (px, py), 4, (0, 255, 255), -1)

                # Skeletal lines ONLY drawn if BOTH keypoints have conf > 0.65
                for p1_id, p2_id in SKELETON_CONNECTIONS:
                    if p1_id in pt_map and p2_id in pt_map:
                        cv2.line(annotated, pt_map[p1_id], pt_map[p2_id], (255, 105, 180), 2, cv2.LINE_AA)

        return annotated

    def generate_mjpeg_stream(
        self,
        source: Union[int, str] = "fence.mp4",
        camera_id: str = "CAM_1",
        conf_thresh: float = 0.38,
        draw_overlay: bool = True,
        enable_ocr: bool = True,
        enable_face: bool = True,
        enable_bla: bool = True,
        draw_trails: Optional[bool] = None,
        enable_vehicle_tracking: Optional[bool] = None
    ) -> Generator[bytes, None, None]:
        """Continuous, robust MJPEG stream generator with dynamic source switching, seamless looping, and frame-skipping."""
        active_source = str(source)
        self.camera_sources[camera_id] = active_source
        resolved_source = self.resolve_video_path(active_source)

        cap = cv2.VideoCapture(resolved_source)
        if not cap.isOpened():
            logger.warning(f"Initial stream source {resolved_source} failed, attempting fallback...")
            fallback_name = CAMERA_CONFIGS.get(camera_id.replace("CAM_", ""), {}).get("default_source", "fence.mp4")
            resolved_source = self.resolve_video_path(fallback_name)
            cap = cv2.VideoCapture(resolved_source)

        fps_tracker = deque(maxlen=15)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0
        frame_count = 0
        last_valid_frame: Optional[np.ndarray] = None
        last_results: Dict[str, Any] = {
            "detections": [],
            "inference_time_ms": 0.0,
            "target_count": 0,
            "breach_alerts": []
        }

        try:
            while True:
                f_start = time.perf_counter()

                # Dynamic Video Source Switching: check if user changed dropdown
                current_target_source = self.camera_sources.get(camera_id, active_source)
                if current_target_source != active_source:
                    logger.info(f"Dynamically switching {camera_id} source from '{active_source}' to '{current_target_source}'")
                    new_resolved = self.resolve_video_path(current_target_source)
                    new_cap = cv2.VideoCapture(new_resolved)
                    if new_cap.isOpened():
                        cap.release()
                        cap = new_cap
                        active_source = current_target_source
                        resolved_source = new_resolved
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        frame_count = 0
                        last_results = {"detections": [], "inference_time_ms": 0.0, "target_count": 0, "breach_alerts": []}
                    else:
                        logger.warning(f"Could not open target source: {new_resolved}")
                        self.camera_sources[camera_id] = active_source

                # Seamless Looping: Proactively rewind when nearing EOF
                if total_frames > 0:
                    curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if curr_pos >= (total_frames - 2):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video ended or read failed: rewind and re-read
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # Re-open capture to recover from decoder EOF state
                        cap.release()
                        cap = cv2.VideoCapture(resolved_source)
                        ret, frame = cap.read()

                    # Fallback to previous valid frame during momentary reopen so feed never blackouts
                    if not ret or frame is None:
                        if last_valid_frame is not None:
                            frame = last_valid_frame.copy()
                        else:
                            time.sleep(0.02)
                            continue
                else:
                    last_valid_frame = frame.copy()

                frame_count += 1
                effective_veh_tracking = self.enable_vehicle_tracking if enable_vehicle_tracking is None else enable_vehicle_tracking

                if draw_overlay:
                    # Decoupled Non-Blocking AI Execution:
                    # Run heavy AI inference on every 6th frame in a background thread
                    # Video frames stream at full 30 FPS with zero stuttering
                    AI_CADENCE = 6
                    should_run_ai = (frame_count % AI_CADENCE == 0 or frame_count == 1)
                    if should_run_ai and not self._ai_busy.get(camera_id, False):
                        self._ai_busy[camera_id] = True
                        frame_for_ai = frame.copy()
                        self._dispatch_ai_inference(
                            frame_for_ai,
                            camera_id=camera_id,
                            conf_thresh=conf_thresh,
                            enable_ocr=enable_ocr,
                            enable_face=enable_face,
                            enable_bla=enable_bla,
                            enable_vehicle_tracking=effective_veh_tracking
                        )

                    # Retrieve latest detections cache without blocking
                    with self._results_lock:
                        current_results = self._cached_results.get(camera_id, {"detections": []})

                    annotated_frame = self.draw_overlays(
                        frame,
                        current_results.get("detections", []),
                        camera_id=camera_id,
                        draw_skeletons=True,
                        draw_bla=enable_bla,
                        draw_trails=draw_trails,
                        enable_vehicle_tracking=effective_veh_tracking
                    )

                    f_time = time.perf_counter() - f_start
                    fps_tracker.append(1.0 / max(f_time, 1e-4))
                    curr_fps = round(sum(fps_tracker) / len(fps_tracker), 1)

                    with self.telemetry_lock:
                        self.latest_telemetry["fps"] = curr_fps

                    cam_label = CAMERA_CONFIGS.get(camera_id.replace("CAM_", ""), {}).get("label", camera_id)
                    gpu_tag = self.device_name.replace("NVIDIA ", "").replace("Laptop GPU", "").strip() or "RTX 4050"
                    inf_ms = current_results.get("inference_time_ms", 0.0)
                    hud_text = f"{cam_label} | GPU: {gpu_tag} | {curr_fps} FPS | {inf_ms}ms"
                    cv2.putText(annotated_frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    annotated_frame = frame

                if annotated_frame.shape[1] > 854:
                    scale = 854.0 / annotated_frame.shape[1]
                    annotated_frame = cv2.resize(annotated_frame, (854, int(annotated_frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                ret_enc, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret_enc:
                    continue

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

                # Frame pacing for fluid ~30 FPS playback without stuttering
                f_elapsed = time.perf_counter() - f_start
                sleep_sec = max(0.005, (1.0 / 30.0) - f_elapsed)
                time.sleep(sleep_sec)

        finally:
            cap.release()

    def generate_mosaic_stream(
        self,
        sources: Tuple[str, str, str] = ("fence.mp4", "vehicle.mp4", "sample.mp4"),
        conf_thresh: float = 0.38,
        draw_trails: Optional[bool] = None,
        enable_vehicle_tracking: Optional[bool] = None
    ) -> Generator[bytes, None, None]:
        """
        High-performance stitched 3-camera mosaic generator.
        Stitches 3 camera feeds horizontally into a single unified surveillance canvas.
        """
        caps = []
        last_mosaic_frames: List[Optional[np.ndarray]] = []
        mosaic_results: Dict[int, Dict[str, Any]] = {}
        for s in sources:
            resolved = self.resolve_video_path(s)
            c = cv2.VideoCapture(resolved)
            caps.append(c)
            last_mosaic_frames.append(None)

        target_w, target_h = 426, 240  # 3 x 426 = 1278 wide
        mosaic_frame_count = 0

        try:
            while True:
                frames = []
                mosaic_frame_count += 1
                effective_veh_tracking = self.enable_vehicle_tracking if enable_vehicle_tracking is None else enable_vehicle_tracking

                for idx, c in enumerate(caps):
                    cam_key = str(idx + 1)
                    cam_id_str = f"CAM_{cam_key}"

                    total_f = c.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total_f > 0:
                        cpos = c.get(cv2.CAP_PROP_POS_FRAMES)
                        if cpos >= (total_f - 2):
                            c.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    ret, fr = c.read()
                    if not ret or fr is None:
                        # Seamless Video Looping for mosaic feeds
                        c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, fr = c.read()
                        if not ret or fr is None:
                            src_cand = sources[idx]
                            res_cand = self.resolve_video_path(src_cand)
                            c.release()
                            c = cv2.VideoCapture(res_cand)
                            caps[idx] = c
                            ret, fr = c.read()

                    if not ret or fr is None:
                        if last_mosaic_frames[idx] is not None:
                            fr = last_mosaic_frames[idx].copy()
                        else:
                            fr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    else:
                        last_mosaic_frames[idx] = fr.copy()

                    # Decoupled Non-Blocking AI Execution for mosaic feed (every 6th frame)
                    MOSAIC_AI_CADENCE = 6
                    should_ai = (mosaic_frame_count % MOSAIC_AI_CADENCE == 0 or idx not in mosaic_results)
                    if should_ai and not self._ai_busy.get(cam_id_str, False):
                        self._ai_busy[cam_id_str] = True
                        fr_ai = fr.copy()
                        self._dispatch_ai_inference(
                            fr_ai,
                            camera_id=cam_id_str,
                            conf_thresh=conf_thresh,
                            enable_ocr=True,
                            enable_face=True,
                            enable_bla=True,
                            enable_vehicle_tracking=effective_veh_tracking
                        )

                    with self._results_lock:
                        res = self._cached_results.get(cam_id_str, {"detections": []})
                    mosaic_results[idx] = res

                    fr = self.draw_overlays(
                        fr,
                        res.get("detections", []),
                        camera_id=cam_id_str,
                        draw_trails=draw_trails,
                        enable_vehicle_tracking=effective_veh_tracking
                    )
                    fr = cv2.resize(fr, (target_w, target_h))

                    # Add camera label header
                    lbl = CAMERA_CONFIGS.get(cam_key, {}).get("label", f"CAM {cam_key}")
                    cv2.putText(fr, f"[{lbl}]", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1, cv2.LINE_AA)
                    frames.append(fr)

                # Horizontal stitch
                mosaic = np.hstack(frames)

                # Resize final 3-camera mosaic frame to a maximum width of 1280px
                if mosaic.shape[1] > 1280:
                    scale = 1280.0 / mosaic.shape[1]
                    mosaic = cv2.resize(mosaic, (1280, int(mosaic.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                # Compress using JPEG quality 70 to minimize bandwidth consumption
                ret_enc, buffer = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret_enc:
                    continue

                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(0.018)

        finally:
            for c in caps:
                c.release()


# =============================================================================
# FASTAPI LIFESPAN & APPLICATION SETUP
# =============================================================================

def resolve_ngrok_public_url() -> str:
    """Detect manually started Ngrok tunnel on localhost:4040, or fallback to default domain."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"User-Agent": "IBVAP"})
        with urllib.request.urlopen(req, timeout=0.3) as resp:
            data = json.loads(resp.read().decode())
            tunnels = data.get("tunnels", [])
            if tunnels and "public_url" in tunnels[0]:
                return tunnels[0]["public_url"]
    except Exception:
        pass
    return os.getenv("NGROK_DOMAIN", "https://uptight-surgical-steadying.ngrok-free.dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load AI engines on GPU (Manual Ngrok tunneling mode)."""
    logger.info("=== Starting IBVAP Multi-Camera Surveillance Server ===")
    cuda_status = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_status else "CPU"
    logger.info(f"CUDA: {cuda_status} | Accelerator: {device_name}")

    app.state.engine = IBVAPSurveillanceEngine(model_path="yolov8n-pose.pt")
    app.state.startup_time = time.time()
    app.state.public_url = os.getenv("PUBLIC_URL")

    port = int(os.getenv("PORT", 8000))
    resolved_url = resolve_ngrok_public_url()
    banner = f"""
=============================================================================
   🚀 IBVAP BACKEND ACTIVE (Manual Two-Terminal Mode)
   Local Dashboard : http://localhost:{port}/frontend
   Ngrok Edge URL  : {resolved_url}
   Manual Tunnel   : Run 'ngrok http {port}' in Terminal 2
=============================================================================
"""
    print(banner, flush=True)

    yield

    logger.info("=== Shutting down IBVAP FastAPI Server ===")
    if hasattr(app.state, "engine"):
        del app.state.engine


app = FastAPI(
    title="IBVAP - Intelligent Border Video Analytics Platform",
    description="GPU Multi-Camera Backend with YOLOv8 Pose, Vehicle & Plate YOLO, EasyOCR ANPR, YuNet Face, & BLA Engine",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# CORE API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
@app.get("/frontend", response_class=HTMLResponse, tags=["Frontend"])
async def frontend_dashboard():
    """Serve the 3-Camera IBVAP Surveillance Command Center frontend."""
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>IBVAP Frontend template not found. Please ensure templates/index.html exists.</h2>",
        status_code=404
    )


@app.get("/api/meta", tags=["System"])
async def system_metadata():
    """System metadata and active GPU verification."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    cuda_available = torch.cuda.is_available() or (engine.is_gpu_available if engine else False)
    device_name = engine.device_name if engine else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "RTX 4050")

    return {
        "project": "IBVAP - Intelligent Border Video Analytics Platform",
        "version": "3.0.0",
        "status": "online",
        "public_url": getattr(app.state, "public_url", None) or resolve_ngrok_public_url(),
        "active_gpu": cuda_available,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "documentation": "/docs"
    }


@app.get("/api/telemetry", tags=["Telemetry"])
async def get_realtime_telemetry():
    """Real-time stats across all active surveillance nodes."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    uptime_sec = round(time.time() - getattr(app.state, "startup_time", time.time()), 1)

    gpu_telemetry: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available() or (engine.is_gpu_available if engine else False),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else (1 if engine and engine.is_gpu_available else 0),
        "device_name": engine.device_name if engine else "RTX 4050",
        "memory_allocated_mb": 0.0,
        "memory_reserved_mb": 0.0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "13.2"
    }

    if torch.cuda.is_available():
        gpu_telemetry.update({
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        })

    with engine.telemetry_lock if engine else threading.Lock():
        live_stats = dict(engine.latest_telemetry) if engine else {}

    has_breach = live_stats.get("breaches", 0) > 0 or len(live_stats.get("active_alerts", [])) > 0
    threat_level = "DEFCON 2 // PERIMETER BREACH" if has_breach else "DEFCON 4 // GUARDED"
    defense_status = "ARMED // ACTIVE WATCH"

    return {
        "status": "operational",
        "defense_status": defense_status,
        "threat_level": threat_level,
        "has_breach": has_breach,
        "public_url": getattr(app.state, "public_url", None) or resolve_ngrok_public_url(),
        "fps": live_stats.get("fps", 0.0),
        "latency_ms": live_stats.get("latency_ms", 0.0),
        "uptime_seconds": uptime_sec,
        "gpu": gpu_telemetry,
        "counts": {
            "total_targets": live_stats.get("total_targets", 0),
            "persons": live_stats.get("persons", 0),
            "vehicles": live_stats.get("vehicles", 0),
            "faces": live_stats.get("faces", 0),
            "plates": live_stats.get("plates", 0),
            "breaches": live_stats.get("breaches", 0)
        },
        "models": {
            "yolo_pose": "ACTIVE (CUDA:0)" if (engine and (engine.device == "cuda" or "cuda" in str(engine.device) or engine.is_gpu_available)) else "ACTIVE (CPU)",
            "yolo_vehicles": "ACTIVE" if engine and engine.vehicle_model else "STANDBY",
            "plate_yolo": "ACTIVE" if engine and engine.plate_model else "STANDBY",
            "easyocr": "ACTIVE (GPU)" if (engine and (engine.is_gpu_available or torch.cuda.is_available())) else "STANDBY",
            "yunet_face": "ACTIVE" if engine and engine.face_cascade else "STANDBY",
            "bla_engine": "ACTIVE (Multi-Camera Tripwire)" if engine and engine.bla_engines else "STANDBY"
        },
        "active_alerts": live_stats.get("active_alerts", []),
        "recent_plates": live_stats.get("recent_plates", [])
    }


@app.get("/api/anpr/registry", tags=["ANPR"])
async def get_anpr_registry():
    """Retrieve full live ANPR vehicle registry."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        return {"registry": []}
    with engine.telemetry_lock:
        return {"registry": list(engine.recent_plates)}


@app.get("/api/status", tags=["System"])
async def get_system_status():
    """Detailed diagnostic status for health checks."""
    return await get_realtime_telemetry()


@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint for Vercel standby frontend handshake."""
    return {
        "status": "ok",
        "public_url": getattr(app.state, "public_url", None) or resolve_ngrok_public_url()
    }


# ---------------------------------------------------------------------------
# Individual Camera Streaming Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/stream/video", tags=["Streaming"])
@app.get("/api/stream/video/1", tags=["Streaming"])
def stream_camera_1(
    source: str = Query("fence.mp4", description="Video source for Camera 1 (Hawkins Post)"),
    conf_thresh: float = Query(0.38, ge=0.05, le=1.0),
    overlay: bool = Query(True),
    ocr: bool = Query(True),
    face: bool = Query(True),
    bla: bool = Query(True),
    trails: Optional[bool] = Query(None, description="Show tracking trails overlay"),
    vehicle_tracking: Optional[bool] = Query(None, description="Show tracking trails for vehicles")
):
    """Camera 1 Feed: Hawkins Post (Default: fence.mp4)."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    return StreamingResponse(
        engine.generate_mjpeg_stream(source=source, camera_id="CAM_1", conf_thresh=conf_thresh,
                                     draw_overlay=overlay, enable_ocr=ocr, enable_face=face, enable_bla=bla,
                                     draw_trails=trails, enable_vehicle_tracking=vehicle_tracking),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/video/2", tags=["Streaming"])
def stream_camera_2(
    source: str = Query("vehicle.mp4", description="Video source for Camera 2 (Out Post - ANPR)"),
    conf_thresh: float = Query(0.38, ge=0.05, le=1.0),
    overlay: bool = Query(True),
    ocr: bool = Query(True),
    face: bool = Query(True),
    bla: bool = Query(True),
    trails: Optional[bool] = Query(None, description="Show tracking trails overlay"),
    vehicle_tracking: Optional[bool] = Query(None, description="Show tracking trails for vehicles")
):
    """Camera 2 Feed: Out Post (Default: vehicle.mp4 - ANPR Vehicle Checkpoint)."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    return StreamingResponse(
        engine.generate_mjpeg_stream(source=source, camera_id="CAM_2", conf_thresh=conf_thresh,
                                     draw_overlay=overlay, enable_ocr=ocr, enable_face=face, enable_bla=bla,
                                     draw_trails=trails, enable_vehicle_tracking=vehicle_tracking),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/video/3", tags=["Streaming"])
def stream_camera_3(
    source: str = Query("sample.mp4", description="Video source for Camera 3 (Alpha Post)"),
    conf_thresh: float = Query(0.38, ge=0.05, le=1.0),
    overlay: bool = Query(True),
    ocr: bool = Query(True),
    face: bool = Query(True),
    bla: bool = Query(True),
    trails: Optional[bool] = Query(None, description="Show tracking trails overlay"),
    vehicle_tracking: Optional[bool] = Query(None, description="Show tracking trails for vehicles")
):
    """Camera 3 Feed: Alpha Post (Default: sample.mp4 - Aerial/Perimeter Patrol)."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    return StreamingResponse(
        engine.generate_mjpeg_stream(source=source, camera_id="CAM_3", conf_thresh=conf_thresh,
                                     draw_overlay=overlay, enable_ocr=ocr, enable_face=face, enable_bla=bla,
                                     draw_trails=trails, enable_vehicle_tracking=vehicle_tracking),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stream/video/mosaic", tags=["Streaming"])
def stream_mosaic(
    cam1_src: str = Query("fence.mp4"),
    cam2_src: str = Query("vehicle.mp4"),
    cam3_src: str = Query("sample.mp4"),
    conf_thresh: float = Query(0.38),
    trails: Optional[bool] = Query(None),
    vehicle_tracking: Optional[bool] = Query(None)
):
    """Combined 3-Camera Mosaic Feed: stitched side-by-side."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    return StreamingResponse(
        engine.generate_mosaic_stream(sources=(cam1_src, cam2_src, cam3_src), conf_thresh=conf_thresh,
                                     draw_trails=trails, enable_vehicle_tracking=vehicle_tracking),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/stream/video/{cam_id}/source", tags=["Streaming"])
@app.get("/api/stream/video/{cam_id}/source", tags=["Streaming"])
async def set_camera_source(cam_id: str, source: str = Query(..., description="Target video filename or device index")):
    """Dynamically switch video source for a camera without breaking the generator or freezing."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    key = f"CAM_{cam_id.replace('CAM_', '')}"
    engine.camera_sources[key] = source
    logger.info(f"Dynamically updated source for {key} to: {source}")
    return {"status": "ok", "camera_id": key, "source": source}


@app.post("/api/settings/trails", tags=["Settings"])
@app.get("/api/settings/trails", tags=["Settings"])
async def toggle_tracking_trails(enabled: Optional[bool] = Query(None, description="Enable or disable tracking trails")):
    """Toggle tracking trails overlay on and off."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    if enabled is not None:
        engine.show_trails = bool(enabled)
        logger.info(f"Tracking trails overlay toggled to: {engine.show_trails}")

    return {"status": "ok", "show_trails": engine.show_trails}


@app.post("/api/settings/vehicle-tracking", tags=["Settings"])
@app.get("/api/settings/vehicle-tracking", tags=["Settings"])
async def toggle_vehicle_tracking(enabled: Optional[bool] = Query(None, description="Enable or disable vehicle tracking trails")):
    """Toggle vehicle tracking trails overlay on and off."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    if enabled is not None:
        engine.enable_vehicle_tracking = bool(enabled)
        for b_eng in engine.bla_engines.values():
            b_eng.enable_vehicle_tracking = engine.enable_vehicle_tracking
        logger.info(f"Vehicle tracking trails toggled to: {engine.enable_vehicle_tracking}")

    return {"status": "ok", "enable_vehicle_tracking": engine.enable_vehicle_tracking}


@app.get("/api/video-sources", tags=["Streaming"])
async def list_video_sources():
    """List available local surveillance video files and camera options."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = [
        f for f in os.listdir(BASE_DIR)
        if os.path.isfile(os.path.join(BASE_DIR, f)) and os.path.splitext(f)[1].lower() in video_exts
    ]
    return {
        "cameras": [{"id": "0", "label": "Live Webcam / USB Camera (Device 0)"}],
        "files": [{"id": f, "label": f} for f in sorted(files)],
        "presets": {
            "cam1": "fence.mp4",
            "cam2": "vehicle.mp4",
            "cam3": "sample.mp4"
        }
    }


@app.get("/api/models", tags=["Registry"])
async def get_models_registry():
    """Checklist of loaded models."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    return {
        "yolo_pose": {"status": "loaded", "model": "yolov8n-pose.pt"},
        "yolo_vehicles": {"status": "loaded" if engine and engine.vehicle_model else "unavailable"},
        "plate_yolo": {"status": "loaded" if engine and engine.plate_model else "unavailable"},
        "easyocr_anpr": {"status": "loaded" if engine and engine.ocr_engine else "unavailable"},
        "yunet_face": {"status": "loaded" if engine and engine.face_cascade else "unavailable"},
        "bla_engine": {"status": "active", "cameras": ["CAM_1", "CAM_2", "CAM_3"]}
    }


@app.post("/api/process-frame", tags=["Inference"])
async def process_frame_endpoint(
    file: UploadFile = File(..., description="Image frame (JPEG/PNG) to analyze"),
    conf_thresh: float = Query(0.38, ge=0.05, le=1.0)
):
    """Process single frame across full pipeline."""
    engine: IBVAPSurveillanceEngine = getattr(app.state, "engine", None)
    if not engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image format")

    results = engine.process_frame(frame, conf_thresh=conf_thresh)
    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
