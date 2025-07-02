from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class StereoCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline: float

    @classmethod
    def from_txt(cls, path: str) -> "StereoCalib":
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        _, fx, fy, cx, cy, _ = lines[0].split()
        w, h = map(int, lines[1].split())
        baseline = float(lines[-1])
        return cls(float(fx), float(fy), float(cx), float(cy), w, h, baseline)

    def proj_mats(self):
        """Return 3x4 projection matrices P_left, P_right."""
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0,       0,       1]], dtype=np.float64)
        P_L = np.hstack([K, np.zeros((3, 1))])
        T = np.array([-self.baseline, 0, 0]).reshape(3, 1)
        P_R = np.hstack([K, K @ T])
        return P_L, P_R

                                 
# State: [x, y, z, vx, vy, vz]                      
class KalmanFilter:
    def __init__(self, dt=1 / 60, process_var=1e-2, meas_var=1e-3):
        self.dt = dt
        # State transition matrix F
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt  # position += velocity * dt
        # Measurement matrix H
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.Q = np.eye(6) * process_var # process noise
        self.R = np.eye(3) * meas_var # measurement noise
        # Internal state
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) # state covariance
        self.I = np.eye(6) # identity for updates

    def initialize(self, xyz0: np.ndarray, xyz1: np.ndarray):
        v0 = (xyz1 - xyz0) / self.dt

        self.x[0:3, 0] = xyz0
        self.x[3:6, 0] = v0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:3].ravel()

    def update(self, z: np.ndarray):
        z = z.reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x[0:3].ravel()

def triangulate(ul, vl, ur, vr, calib: StereoCalib):
    P1, P2 = calib.proj_mats()
    pts4 = cv2.triangulatePoints(P1, P2,
                                 np.array([[ul], [vl]], dtype=np.float64),
                                 np.array([[ur], [vr]], dtype=np.float64))
    pts3 = (pts4[:3] / pts4[3]).flatten()
    return pts3  # meters


# Yolo utils
def pick_banana_center(infer_result, conf_thres=0.003):
    """Pick the center of the highest confidence banana detection"""
    best_conf = 0.0
    best_det = None

    predictions = infer_result.predictions

    for p in predictions:
        if p.class_name == "banana" and p.confidence >= conf_thres:
            if p.confidence > best_conf:
                best_conf = p.confidence
                best_det = p

    if best_det is not None:
        return {'u': best_det.x, 'v': best_det.y}
    else:
        return None


