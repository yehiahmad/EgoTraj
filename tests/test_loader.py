"""Loader tests against a small synthetic file built with the release's layout."""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from egotraj import iter_sessions, list_sessions, load_annotations, load_session  # noqa: E402
from egotraj.loader import _main  # noqa: E402

N = 40


def _write_session(f, name, with_video=True):
    g = f.create_group(name)
    t = 1.76e9 + np.arange(N) / 30.0
    g["pose/timestamp"] = t
    g["pose/position"] = np.stack([np.linspace(0, 4, N), np.full(N, 1.7), np.linspace(0, 2, N)], 1)
    g["pose/rotation"] = np.tile([0.995, 0.0, 0.1, 0.0], (N, 1))
    g["pose/velocity"] = np.ones((N, 3))
    g["pose/angular_velocity"] = np.zeros((N, 3))
    g["gaze/origin"] = np.zeros((N, 3))
    g["gaze/direction"] = np.tile([0.0, 0.0, 1.0], (N, 1))
    if with_video:
        g["video/segment"] = np.zeros(N, dtype=np.int32)
        g["video/frame"] = np.arange(N, dtype=np.int32)
        g["video/has_video"] = np.arange(N) % 5 != 0
    g.attrs["from"] = "Gates Dell Complex"
    g.attrs["to"] = b"Union Building"
    g.attrs["waypoint_start"] = np.array([0.0, 0.0])


@pytest.fixture
def h5(tmp_path):
    path = tmp_path / "egotraj_dataset.h5"
    with h5py.File(path, "w") as f:
        _write_session(f, "20251020_163423")
        _write_session(f, "20251019_101500", with_video=False)
    return path


def test_list_sessions_is_sorted(h5):
    assert list_sessions(h5) == ["20251019_101500", "20251020_163423"]


def test_load_session_returns_arrays_as_stored(h5):
    s = load_session(h5, "20251020_163423")
    assert len(s) == N
    assert s.position.shape == (N, 3) and s.rotation.shape == (N, 4)
    np.testing.assert_allclose(s.rotation[0], [0.995, 0.0, 0.1, 0.0])
    assert s.has_video.dtype == bool and s.has_video.sum() == N - N // 5
    assert s.ground_xz.shape == (N, 2)
    np.testing.assert_allclose(s.ground_xz[:, 1], s.position[:, 2])
    assert s.duration == pytest.approx((N - 1) / 30.0)
    assert s.attrs["from"] == "Gates Dell Complex" and s.attrs["to"] == "Union Building"
    assert s.attrs["waypoint_start"] == [0.0, 0.0]


def test_video_pointers_are_optional(h5):
    s = load_session(h5, "20251019_101500")
    assert s.video_segment is None and s.video_frame is None and s.has_video is None


def test_unknown_session_names_alternatives(h5):
    with pytest.raises(KeyError, match="2 sessions available"):
        load_session(h5, "nope")


def test_iter_sessions(h5):
    assert [s.name for s in iter_sessions(h5)] == list_sessions(h5)


def test_annotations_filter_and_join(h5, tmp_path):
    ann = tmp_path / "egotraj_annotations.json"
    ann.write_text(json.dumps([
        {"session": "20251020_163423", "pose_idx": 30, "second": 1, "annotation": "b"},
        {"session": "20251019_101500", "pose_idx": 0, "second": 0, "annotation": "x"},
        {"session": "20251020_163423", "pose_idx": 0, "second": 0, "annotation": "a"},
    ]), encoding="utf-8")
    rows = load_annotations(ann, session="20251020_163423")
    assert [r["annotation"] for r in rows] == ["a", "b"]
    s = load_session(h5, rows[1]["session"])
    np.testing.assert_allclose(s.position[rows[1]["pose_idx"]], s.position[30])
    assert len(load_annotations(ann)) == 3


def test_dataframe(h5):
    pd = pytest.importorskip("pandas")
    df = load_session(h5, "20251020_163423").to_dataframe()
    assert len(df) == N and {"position_x", "rotation_3", "has_video"} <= set(df.columns)
    assert isinstance(df, pd.DataFrame)


def test_cli(h5, capsys):
    assert _main([str(h5)]) == 0
    assert "2 sessions" in capsys.readouterr().out
