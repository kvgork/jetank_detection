"""
Detection backend abstractions for the JeTank sock detector.

Stage 1: UltralyticsBackend (PyTorch, runs in pixi after pip install ultralytics)
Stage 2/3: TensorRT backends (see plan §5)

The ultralytics import is deferred to load() so this module can be imported
without torch or ultralytics installed (required for colcon build and CI).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Detection:
    """A single object detection result."""

    # Bounding box centre and size in pixels
    cx: float
    cy: float
    w: float
    h: float
    score: float
    class_id: int = 0
    label: str = "sock"


class DetectorBackend(ABC):
    """Abstract base class for detector backends."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the model from *model_path*."""
        ...

    @abstractmethod
    def infer(self, image_bgr, conf_threshold: float = 0.5) -> list:
        """
        Run inference on *image_bgr* (numpy HxWxC BGR uint8).

        Returns a list of :class:`Detection` objects with score >=
        *conf_threshold*.
        """
        ...


class UltralyticsBackend(DetectorBackend):
    """
    YOLO backend using the Ultralytics library (Stage 1 — PyTorch).

    ``ultralytics`` is NOT imported at module level so the package can be
    imported without torch installed.  Call :meth:`load` before :meth:`infer`.
    """

    def __init__(self) -> None:
        """Initialise backend without loading a model."""
        self._model = None

    def load(self, model_path: str) -> None:
        """
        Load a YOLO model from *model_path*.

        Raises :class:`RuntimeError` if ``ultralytics`` is not installed.
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415 (deferred import intentional)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics not installed — pip install ultralytics in the pixi env (Stage 1)"
            ) from exc
        self._model = YOLO(model_path)

    def infer(self, image_bgr, conf_threshold: float = 0.5) -> list:
        """Run YOLO inference and return a list of :class:`Detection` objects."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        results = self._model.predict(
            image_bgr,
            conf=conf_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # xyxy → cx, cy, w, h
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(cx=cx, cy=cy, w=w, h=h, score=score, class_id=cls_id)
                )
        return detections


def make_backend(name: str = "ultralytics") -> DetectorBackend:
    """
    Create a detector backend by name.

    Parameters
    ----------
    name:
        ``"ultralytics"`` (Stage 1, PyTorch) is the only currently
        implemented backend.  ``"tensorrt"`` and ``"subprocess"`` are
        reserved for Stage 2/3 (see plan §5).

    Raises
    ------
    NotImplementedError
        For unimplemented Stage 2/3 backends.
    ValueError
        For unknown backend names.

    """
    if name == "ultralytics":
        return UltralyticsBackend()
    elif name in ("tensorrt", "subprocess"):
        raise NotImplementedError(
            f"Backend '{name}' is Stage 2/3 — see plan §5. "
            "Use 'ultralytics' for Stage 1."
        )
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. Valid: 'ultralytics', 'tensorrt', 'subprocess'"
        )
