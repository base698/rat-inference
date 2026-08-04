"""Shared imports and optional-dependency guards for the ratbot app.

Star-imported by the app modules so method bodies keep their original
bare-name references. Import failures degrade features exactly as the
monolithic rt_200.py did.
"""

import os
import json
import time
import threading
import io
import shutil
import yaml
from datetime import datetime
import uvicorn
import numpy as np
from collections import deque
from ratbot.vision.yolo_inference import run_inference as yolo_run_inference, extract_detections
from ratbot.vision.camera_source import CameraSource
from ratbot.vision.overlay import OverlayRenderer
from ratbot.vision.stereo_depth import StereoDepthService
from ratbot.runtime_config import parse_runtime_config
from ratbot.tracking import (
    Detection3D,
    MultiTargetTracker,
    ServoKinematicsConfig,
    TrackManagerConfig,
    TurretFrameTransformer,
    WorldTrackBeliefAdapter,
)
from ratbot.tracking.recording import TrackRecordingStore
from ratbot.robot import (
    AngularBeliefController,
    VelocityFormController,
    AngularTargetBelief,
    CrosshairAiming,
    DepthCrosshairCompensation,
    ObservationConfig,
    PitchCompensation,
    ServoBounds,
    TrackingObservationConverter,
    TrackingServoController,
    TriggerServoConfig,
    TriggerServoController,
    YawCompensation,
)
from ratbot.web import ControlApiConfig, create_control_app

# Set library path for macOS
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

# Try importing OpenCV for camera
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - camera features disabled")

# Try importing CSI camera helper
CSICameraCapture = None
try:
    from ratbot.vision.csi_camera import CSICameraCapture
    CSI_HELPER_AVAILABLE = True
except ImportError:
    CSI_HELPER_AVAILABLE = False

# Try importing GPIO library
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception as exc:
    GPIO_AVAILABLE = False
    print(f"GPIO library not available - trigger servo features disabled ({exc})")

# Try importing YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available - detection features disabled")

# Try importing Feetech servo libraries
FeetechMotorsBus = None
Motor = None
MotorNormMode = None
try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
    FEETECH_AVAILABLE = True
except Exception as exc:
    FEETECH_AVAILABLE = False
    print(f"Feetech libraries not available - servo tracking features disabled ({exc})")

