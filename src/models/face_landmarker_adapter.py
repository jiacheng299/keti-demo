"""CPU MediaPipe Tasks adapter with measured eyes and conservative continuity."""

import math

import cv2
import numpy as np

from src.schemas.face_observation import FaceObservation
from src.tracking.face_tracker import FaceTracker


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MAX_FACES = 5


def eye_aspect_ratio(points):
    points = np.asarray(points, dtype=float)
    width = np.linalg.norm(points[0] - points[3])
    if width < 1:
        return None
    return float((np.linalg.norm(points[1] - points[5]) +
                  np.linalg.norm(points[2] - points[4])) / (2 * width))


class FaceLandmarkerAdapter:
    def __init__(self, definition):
        self.definition = definition
        self._model = None
        self._mp = None
        self._tracker = FaceTracker()
        self._last_ms = -1
        self.observation = FaceObservation()
        self.observations = ()

    def load(self):
        if self._model is not None:
            return
        if not self.definition.weights.is_file():
            raise FileNotFoundError("缺少人脸模型，请运行 .venv\\Scripts\\python.exe scripts\\download_face_model.py")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("人脸依赖未安装，请使用项目虚拟环境安装 requirements.txt") from error
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            # Buffer loading avoids native Windows file APIs mangling Chinese paths.
            base_options=mp.tasks.BaseOptions(model_asset_buffer=self.definition.weights.read_bytes(),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=MAX_FACES,
            min_face_detection_confidence=self.definition.confidence,
            min_face_presence_confidence=self.definition.confidence,
            min_tracking_confidence=self.definition.confidence,
            output_facial_transformation_matrixes=True,
        )
        self._model = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def predict(self, frame, frame_id=0, *, timestamp_seconds=None):
        if timestamp_seconds is None:
            raise ValueError("人脸视频推理必须提供视频时间戳")
        self.load()
        timestamp_ms = round(timestamp_seconds * 1000)
        if timestamp_ms <= self._last_ms:
            raise ValueError("人脸视频时间戳必须严格递增")
        self._last_ms = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._model.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        faces = [self._measure(landmarks, result.facial_transformation_matrixes, i, frame.shape)
                 for i, landmarks in enumerate(result.face_landmarks)]
        self.observations = self._tracker.update(faces, timestamp_seconds)
        # Compatibility for single-face consumers. The pipeline uses observations.
        self.observation = self.observations[0] if len(self.observations) == 1 else FaceObservation(
            reason="多张人脸请读取 observations" if self.observations else "未检测到人脸")
        return []

    @staticmethod
    def _measure(landmarks, matrices, index, shape):
        h, w = shape[:2]
        points = np.asarray([(p.x * w, p.y * h) for p in landmarks])
        if len(points) <= max(LEFT_EYE + RIGHT_EYE) or not np.isfinite(points).all():
            return FaceObservation(reason="眼部关键点不可靠")
        eyes = points[list(LEFT_EYE + RIGHT_EYE)]
        left, right = eye_aspect_ratio(eyes[:6]), eye_aspect_ratio(eyes[6:])
        box = (float(max(0, points[:, 0].min())), float(max(0, points[:, 1].min())),
               float(min(w, points[:, 0].max())), float(min(h, points[:, 1].max())))
        if box[2] <= box[0] or box[3] <= box[1]:
            return FaceObservation(reason="人脸超出画面")
        if left is None or right is None:
            return FaceObservation(reason="眼部关键点不可靠", bbox_xyxy=box)
        if index >= len(matrices):
            return FaceObservation(reason="无法估计人脸朝向", bbox_xyxy=box)
        rotation = np.asarray(matrices[index])[:3, :3]
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            return FaceObservation(reason="无法估计人脸朝向", bbox_xyxy=box)
        yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
        return FaceObservation(valid=True, reason="", bbox_xyxy=box, left_ear=left,
                                            right_ear=right, yaw_degrees=yaw,
                                            eye_points=tuple(map(tuple, eyes)))

    def unload(self):
        if self._model is not None:
            self._model.close()
        self._model = None
        self._tracker = FaceTracker()
        self._last_ms = -1
        self.observation = FaceObservation()
        self.observations = ()
