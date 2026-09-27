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
        Load a YOLO model from *model_path* and warm it up.

        Raises :class:`RuntimeError` if ``ultralytics`` is not installed.
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415 (deferred import intentional)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics not installed — pip install ultralytics in the pixi env (Stage 1)"
            ) from exc
        self._model = YOLO(model_path)

        # Warm up: the first predict() call triggers CUDA context creation,
        # kernel JIT and cuDNN autotune, typically 1-5 s on Jetson. Pay that
        # cost here, at on_configure, instead of inside the first DetectSocks
        # goal's timeout or the first continuous-mode frame.
        import numpy as np  # noqa: PLC0415 (deferred; numpy is a hard ultralytics dep)

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False)

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
            if boxes is None or len(boxes) == 0:
                continue
            # Pull each field for the whole batch in one device-to-host
            # transfer instead of building a per-box Boxes wrapper and doing
            # three tensor conversions per detection.
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.tolist()
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, classes):
                # xyxy → cx, cy, w, h
                detections.append(
                    Detection(
                        cx=(x1 + x2) / 2.0,
                        cy=(y1 + y2) / 2.0,
                        w=x2 - x1,
                        h=y2 - y1,
                        score=float(score),
                        class_id=int(cls_id),
                    )
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
