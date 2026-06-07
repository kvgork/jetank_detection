"""
Pure-logic tests for jetank_detection.

These tests exercise the package's own code (not ROS plumbing): the
xyxy->cx/cy/w/h box conversion in :class:`UltralyticsBackend.infer`, the
filename-resume math in :meth:`FrameCaptureNode._next_start_index`, and the
geometry/message-mapping helpers on :class:`SockDetectorNode`.

Heavy/hardware deps (rclpy, cv_bridge, cv2, sensor_msgs, the generated action
package) are stubbed when absent so the modules import in a bare env, following
the project test pattern (jetank_web_control/test/test_labels.py).  The pure
methods are then called on instances built with ``object.__new__`` so no ROS
context / node init is required.
"""

import importlib
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub infrastructure (only fills in what is missing in a bare env)
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    """Stub rclpy / cv_bridge / sensor_msgs / cv2 / action pkg if absent."""
    if 'rclpy' not in sys.modules:
        try:
            import rclpy  # noqa: F401 - prefer the real package
        except ImportError:
            rclpy_stub = _make_stub('rclpy')
            node_stub = _make_stub('rclpy.node')
            node_stub.Node = object
            rclpy_stub.node = node_stub

            action_stub = _make_stub('rclpy.action')
            for attr in ('ActionServer', 'CancelResponse', 'GoalResponse'):
                setattr(action_stub, attr, type(attr, (), {}))
            rclpy_stub.action = action_stub

            cbg_stub = _make_stub('rclpy.callback_groups')
            cbg_stub.ReentrantCallbackGroup = object
            rclpy_stub.callback_groups = cbg_stub

            exec_stub = _make_stub('rclpy.executors')
            exec_stub.MultiThreadedExecutor = object
            rclpy_stub.executors = exec_stub

            life_stub = _make_stub('rclpy.lifecycle')
            life_stub.LifecycleNode = object
            life_stub.TransitionCallbackReturn = type(
                'TransitionCallbackReturn', (), {'SUCCESS': 0})
            rclpy_stub.lifecycle = life_stub

    if 'cv_bridge' not in sys.modules:
        try:
            import cv_bridge  # noqa: F401
        except ImportError:
            cvb = _make_stub('cv_bridge')
            cvb.CvBridge = type('CvBridge', (), {})

    if 'cv2' not in sys.modules:
        try:
            import cv2  # noqa: F401
        except ImportError:
            _make_stub('cv2')

    for pkg in ('sensor_msgs', 'sensor_msgs.msg', 'vision_msgs', 'vision_msgs.msg'):
        if pkg not in sys.modules:
            try:
                importlib.import_module(pkg)
            except ImportError:
                _make_stub(pkg)
    if not hasattr(sys.modules.get('sensor_msgs.msg', object()), 'Image'):
        m = sys.modules.get('sensor_msgs.msg')
        if m is not None and not hasattr(m, 'Image'):
            m.Image = type('Image', (), {})


_install_stubs()

# Make the package importable from its source tree.
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


# ---------------------------------------------------------------------------
# backends.UltralyticsBackend.infer  (box conversion, no ultralytics needed)
# ---------------------------------------------------------------------------

from jetank_detection.backends import Detection, UltralyticsBackend  # noqa: E402


class _FakeTensor:
    """Mimics the .tolist()/float()/int() surface of an ultralytics tensor row."""

    def __init__(self, value):
        self._value = value

    def tolist(self):
        return list(self._value)

    def __float__(self):
        return float(self._value)

    def __int__(self):
        return int(self._value)


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [_FakeTensor(xyxy)]
        self.conf = [_FakeTensor(conf)]
        self.cls = [_FakeTensor(cls)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, image, conf, verbose):  # noqa: D102
        self.calls.append((conf, verbose))
        return self._results


class TestUltralyticsInfer:
    def test_infer_requires_loaded_model(self):
        with pytest.raises(RuntimeError):
            UltralyticsBackend().infer(object())

    def test_xyxy_to_cxcywh_conversion(self):
        # box (10,20)-(50,80): cx=30 cy=50 w=40 h=60
        backend = UltralyticsBackend()
        backend._model = _FakeModel([_FakeResult([_FakeBox([10, 20, 50, 80], 0.8, 0)])])
        dets = backend.infer(object(), conf_threshold=0.4)
        assert len(dets) == 1
        d = dets[0]
        assert isinstance(d, Detection)
        assert d.cx == pytest.approx(30.0)
        assert d.cy == pytest.approx(50.0)
        assert d.w == pytest.approx(40.0)
        assert d.h == pytest.approx(60.0)
        assert d.score == pytest.approx(0.8)
        assert d.class_id == 0

    def test_conf_threshold_forwarded_to_predict(self):
        backend = UltralyticsBackend()
        model = _FakeModel([_FakeResult([])])
        backend._model = model
        backend.infer(object(), conf_threshold=0.73)
        assert model.calls == [(0.73, False)]

    def test_none_boxes_result_skipped(self):
        backend = UltralyticsBackend()
        backend._model = _FakeModel([_FakeResult(None)])
        assert backend.infer(object()) == []

    def test_multiple_boxes_and_class_id(self):
        backend = UltralyticsBackend()
        backend._model = _FakeModel([_FakeResult([
            _FakeBox([0, 0, 2, 2], 0.5, 1),
            _FakeBox([4, 4, 6, 10], 0.9, 3),
        ])])
        dets = backend.infer(object())
        assert [d.class_id for d in dets] == [1, 3]
        assert dets[1].cx == pytest.approx(5.0)
        assert dets[1].cy == pytest.approx(7.0)
        assert dets[1].w == pytest.approx(2.0)
        assert dets[1].h == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# capture_frames.FrameCaptureNode._next_start_index  (filename resume math)
# ---------------------------------------------------------------------------

cf = importlib.import_module('jetank_detection.capture_frames')
FrameCaptureNode = cf.FrameCaptureNode


def _bare_capture(output_dir, domain='sim'):
    """Build a FrameCaptureNode with only the attrs _next_start_index reads."""
    node = object.__new__(FrameCaptureNode)
    node._output_dir = str(output_dir)
    node._domain = domain
    return node


class TestNextStartIndex:
    def test_empty_dir_starts_at_zero(self, tmp_path):
        assert _bare_capture(tmp_path)._next_start_index() == 0

    def test_resumes_after_highest_index(self, tmp_path):
        for n in (0, 1, 5):
            (tmp_path / f'sock_sim_{n:06d}.jpg').write_text('x')
        assert _bare_capture(tmp_path)._next_start_index() == 6

    def test_ignores_other_domain(self, tmp_path):
        (tmp_path / 'sock_real_000009.jpg').write_text('x')
        (tmp_path / 'sock_sim_000002.jpg').write_text('x')
        assert _bare_capture(tmp_path, 'sim')._next_start_index() == 3

    def test_ignores_non_jpg_and_non_numeric(self, tmp_path):
        (tmp_path / 'sock_sim_000004.png').write_text('x')   # wrong ext
        (tmp_path / 'sock_sim_abc.jpg').write_text('x')       # non-numeric stem
        (tmp_path / 'unrelated.jpg').write_text('x')          # wrong prefix
        assert _bare_capture(tmp_path, 'sim')._next_start_index() == 0

    def test_non_numeric_only_returns_zero(self, tmp_path):
        # Matches prefix+ext but stem is not all digits -> no usable index.
        (tmp_path / 'sock_sim_v2.jpg').write_text('x')
        assert _bare_capture(tmp_path, 'sim')._next_start_index() == 0


# ---------------------------------------------------------------------------
# sock_detector_node helpers  (geometry + Detection->message mapping)
# ---------------------------------------------------------------------------

sdn = importlib.import_module('jetank_detection.sock_detector_node')
SockDetectorNode = sdn.SockDetectorNode

# These message-mapping tests need the real vision_msgs types (not stubs): a
# real Detection2DArray() is constructible and carries a populated bbox tree.
try:
    _probe = sdn.Detection2DArray()
    _probe.header  # noqa: B018 - real msg exposes a header field
    _have_vision_msgs = True
except Exception:
    _have_vision_msgs = False


def _bare_detector():
    return object.__new__(SockDetectorNode)


class TestDrawDetectionsGeometry:
    """_draw_detections converts cx/cy/w/h to corner coords for cv2.rectangle."""

    def test_corner_math_via_recorded_rectangles(self, monkeypatch):
        rects = []

        class _CV2Spy:
            FONT_HERSHEY_SIMPLEX = 0

            def rectangle(self, img, p1, p2, *a, **k):
                rects.append((p1, p2))

            def putText(self, *a, **k):
                pass

        # _draw_detections does `import cv2` locally; patch the module entry.
        monkeypatch.setitem(sys.modules, 'cv2', _CV2Spy())

        class _Img:
            def copy(self):
                return self

        det = Detection(cx=100.0, cy=50.0, w=40.0, h=20.0, score=0.9)
        out = _bare_detector()._draw_detections(_Img(), [det])
        assert out is not None
        # x1=cx-w/2=80, y1=cy-h/2=40, x2=cx+w/2=120, y2=cy+h/2=60
        assert rects == [((80, 40), (120, 60))]

    def test_truncates_to_int(self, monkeypatch):
        rects = []
        monkeypatch.setitem(
            sys.modules, 'cv2',
            type('C', (), {
                'FONT_HERSHEY_SIMPLEX': 0,
                'rectangle': lambda self, img, p1, p2, *a, **k: rects.append((p1, p2)),
                'putText': lambda self, *a, **k: None,
            })(),
        )

        class _Img:
            def copy(self):
                return self

        det = Detection(cx=10.6, cy=10.6, w=3.0, h=3.0, score=0.5)
        _bare_detector()._draw_detections(_Img(), [det])
        # int() truncates toward zero: 10.6-1.5=9.1 -> 9 ; 10.6+1.5=12.1 -> 12
        assert rects == [((9, 9), (12, 12))]


@pytest.mark.skipif(not _have_vision_msgs, reason='vision_msgs not available')
class TestBuildDetectionArray:
    def test_maps_detection_fields_to_message(self):
        header = sdn.Detection2DArray().header  # a real std_msgs Header instance
        dets = [
            Detection(cx=1.0, cy=2.0, w=3.0, h=4.0, score=0.7, label='sock'),
            Detection(cx=5.0, cy=6.0, w=7.0, h=8.0, score=0.3, label='sock'),
        ]
        arr = _bare_detector()._build_detection_array(dets, header)
        assert len(arr.detections) == 2
        d0 = arr.detections[0]
        assert d0.bbox.center.position.x == pytest.approx(1.0)
        assert d0.bbox.center.position.y == pytest.approx(2.0)
        assert d0.bbox.size_x == pytest.approx(3.0)
        assert d0.bbox.size_y == pytest.approx(4.0)
        assert len(d0.results) == 1
        assert d0.results[0].hypothesis.class_id == 'sock'
        assert d0.results[0].hypothesis.score == pytest.approx(0.7)

    def test_empty_detections_empty_array(self):
        header = sdn.Detection2DArray().header
        arr = _bare_detector()._build_detection_array([], header)
        assert list(arr.detections) == []
