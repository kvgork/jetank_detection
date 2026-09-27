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

    def __init__(
        self,
        imgsz: int = 640,
        half: bool | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialise backend without loading a model.

        Parameters
        ----------
        imgsz:
            Inference input size passed to ``predict()`` (default 640, the
            ultralytics default — unchanged behaviour).
        half:
            Force FP16 inference. ``None`` (default) autodetects in
            :meth:`load`: ``True`` when the resolved device is CUDA,
            ``False`` on CPU — matching current (FP32-on-CPU) behaviour.
        device:
            Ultralytics device string (e.g. ``"cuda:0"``, ``"cpu"``).
            ``None`` (default) autodetects in :meth:`load` via
            ``torch.cuda.is_available()``.

        """
        self._model = None
        self._imgsz = imgsz
        self._half = half
        self._device = device

    def load(self, model_path: str) -> None:
        """
        Load a YOLO model from *model_path* and warm it up.

        Raises :class:`RuntimeError` if ``ultralytics`` is not installed.
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415 (deferred import intentional)
            import torch  # noqa: PLC0415 (ultralytics dep; free once YOLO is imported)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics not installed — pip install ultralytics in the pixi env (Stage 1)"
            ) from exc
        self._model = YOLO(model_path)

        # Resolve device/half once, at load time, rather than per predict()
        # call. Preserves current behaviour: FP32 unless the resolved device
        # is CUDA.
        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._half is None:
            self._half = self._device.startswith("cuda")

        # Warm up: the first predict() call triggers CUDA context creation,
        # kernel JIT and cuDNN autotune, typically 1-5 s on Jetson. Pay that
        # cost here, at on_configure, instead of inside the first DetectSocks
        # goal's timeout or the first continuous-mode frame.
        import numpy as np  # noqa: PLC0415 (deferred; numpy is a hard ultralytics dep)

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            imgsz=self._imgsz,
            half=self._half,
            device=self._device,
            verbose=False,
        )

    def infer(self, image_bgr, conf_threshold: float = 0.5) -> list:
        """Run YOLO inference and return a list of :class:`Detection` objects."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        results = self._model.predict(
            image_bgr,
            conf=conf_threshold,
            imgsz=self._imgsz,
            half=self._half,
            device=self._device,
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


def make_backend(name: str = "ultralytics", **kwargs) -> DetectorBackend:
    """
    Create a detector backend by name.

    Parameters
    ----------
    name:
        ``"ultralytics"`` (Stage 1, PyTorch) is the only currently
        implemented backend.  ``"tensorrt"`` and ``"subprocess"`` are
        reserved for Stage 2/3 (see plan §5).
    kwargs:
        Forwarded to the backend constructor (e.g. ``imgsz``, ``half``,
        ``device`` for ``"ultralytics"``).

    Raises
    ------
    NotImplementedError
        For unimplemented Stage 2/3 backends.
    ValueError
        For unknown backend names.

    """
    if name == "ultralytics":
        return UltralyticsBackend(**kwargs)
    elif name in ("tensorrt", "subprocess"):
        raise NotImplementedError(
            f"Backend '{name}' is Stage 2/3 — see plan §5. "
            "Use 'ultralytics' for Stage 1."
        )
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. Valid: 'ultralytics', 'tensorrt', 'subprocess'"
        )
