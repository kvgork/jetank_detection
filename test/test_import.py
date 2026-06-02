"""
Import-level tests for jetank_detection.

These tests verify that the package can be imported without torch/ultralytics
installed, which is the case in a fresh pixi environment (Stage 1 not yet
pip-installed).
"""

import pytest


def test_backends_importable():
    """The backends module must import without torch or ultralytics."""
    from jetank_detection import backends  # noqa: F401

    assert backends is not None


def test_detection_dataclass():
    """Detection dataclass has expected fields."""
    from jetank_detection.backends import Detection

    det = Detection(cx=10.0, cy=20.0, w=50.0, h=60.0, score=0.9)
    assert det.cx == pytest.approx(10.0)
    assert det.cy == pytest.approx(20.0)
    assert det.label == "sock"
    assert det.class_id == 0


def test_make_backend_returns_ultralytics():
    """make_backend('ultralytics') returns an UltralyticsBackend instance."""
    from jetank_detection.backends import UltralyticsBackend, make_backend

    backend = make_backend("ultralytics")
    assert isinstance(backend, UltralyticsBackend)


def test_make_backend_tensorrt_not_implemented():
    """Stage 2/3 backends raise NotImplementedError."""
    from jetank_detection.backends import make_backend

    with pytest.raises(NotImplementedError):
        make_backend("tensorrt")

    with pytest.raises(NotImplementedError):
        make_backend("subprocess")


def test_make_backend_invalid_raises_value_error():
    """Unknown backend names raise ValueError."""
    from jetank_detection.backends import make_backend

    with pytest.raises(ValueError):
        make_backend("invalid_backend")
