"""Read EgoTraj's ``egotraj_dataset.h5`` and ``egotraj_annotations.json``.

The loader returns the arrays exactly as stored. It does not resample, crop, rotate or
split anything, so the numbers you get are the numbers in the release.

Layout of ``egotraj_dataset.h5``: one group per session, named by capture timestamp
(for example ``20251020_163423``), holding

    pose/position          (N, 3)  metres, session-relative, Y is up
    pose/rotation          (N, 4)  head orientation quaternion
    pose/velocity          (N, 3)
    pose/angular_velocity  (N, 3)
    pose/timestamp         (N,)    epoch seconds
    gaze/origin            (N, 3)
    gaze/direction         (N, 3)  unit vectors
    video/segment          (N,)    optional, index of the video part file
    video/frame            (N,)    optional, frame within that part
    video/has_video        (N,)    optional, False where no frame matches

plus the group attributes ``waypoint_start``, ``waypoint_end``, ``from`` and ``to``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

PathLike = Union[str, Path]

_POSE = ("position", "rotation", "velocity", "angular_velocity", "timestamp")
_GAZE = ("origin", "direction")
_VIDEO = ("segment", "frame", "has_video")


@dataclass
class Session:
    """One EgoTraj recording. Every per-sample array has the same length ``N``."""

    name: str
    timestamp: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    gaze_origin: np.ndarray
    gaze_direction: np.ndarray
    video_segment: Optional[np.ndarray] = None
    video_frame: Optional[np.ndarray] = None
    has_video: Optional[np.ndarray] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.timestamp.shape[0])

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return float(self.timestamp[-1] - self.timestamp[0]) if len(self) else 0.0

    @property
    def ground_xz(self) -> np.ndarray:
        """Trajectory on the ground plane, ``(N, 2)`` as ``(x, z)``; Y is the up axis."""
        return self.position[:, [0, 2]]

    def to_dataframe(self):
        """The session as a pandas DataFrame, one row per pose sample. Needs pandas."""
        import pandas as pd

        cols: Dict[str, np.ndarray] = {"timestamp": self.timestamp}
        for name, arr, labels in (
            ("position", self.position, "xyz"),
            ("velocity", self.velocity, "xyz"),
            ("angular_velocity", self.angular_velocity, "xyz"),
            ("gaze_origin", self.gaze_origin, "xyz"),
            ("gaze_direction", self.gaze_direction, "xyz"),
        ):
            for i, axis in enumerate(labels):
                cols[f"{name}_{axis}"] = arr[:, i]
        for i in range(self.rotation.shape[1]):
            cols[f"rotation_{i}"] = self.rotation[:, i]
        for name in ("video_segment", "video_frame", "has_video"):
            arr = getattr(self, name)
            if arr is not None:
                cols[name] = arr
        return pd.DataFrame(cols)


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("EgoTraj needs h5py: pip install h5py") from exc
    return h5py


def _attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def list_sessions(h5_path: PathLike) -> List[str]:
    """Session names in the file, sorted, which is also chronological order."""
    with _h5py().File(h5_path, "r") as f:
        return sorted(f.keys())


def load_session(h5_path: PathLike, session: str) -> Session:
    """Load one session's pose, gaze and video pointers into memory."""
    with _h5py().File(h5_path, "r") as f:
        if session not in f:
            available = sorted(f.keys())
            raise KeyError(
                f"session {session!r} not found; {len(available)} sessions available, "
                f"e.g. {available[:3]}"
            )
        g = f[session]
        pose = {k: g[f"pose/{k}"][:] for k in _POSE}
        gaze = {k: g[f"gaze/{k}"][:] for k in _GAZE}
        video = {k: (g[f"video/{k}"][:] if f"video/{k}" in g else None) for k in _VIDEO}
        attrs = {k: _attr(v) for k, v in g.attrs.items()}
    if video["has_video"] is not None:
        video["has_video"] = video["has_video"].astype(bool)
    return Session(
        name=session,
        timestamp=pose["timestamp"],
        position=pose["position"],
        rotation=pose["rotation"],
        velocity=pose["velocity"],
        angular_velocity=pose["angular_velocity"],
        gaze_origin=gaze["origin"],
        gaze_direction=gaze["direction"],
        video_segment=video["segment"],
        video_frame=video["frame"],
        has_video=video["has_video"],
        attrs=attrs,
    )


def iter_sessions(h5_path: PathLike) -> Iterator[Session]:
    """Load every session in turn, oldest first."""
    for name in list_sessions(h5_path):
        yield load_session(h5_path, name)


def load_annotations(json_path: PathLike, session: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scene annotations (1 per second), sorted by session then ``pose_idx``.

    Each record has ``session``, ``pose_idx``, ``timestamp``, ``second``, ``gaze_dot`` and
    ``annotation``. ``pose_idx`` indexes the session's arrays, so
    ``load_session(h5, r["session"]).position[r["pose_idx"]]`` is where the wearer was.
    """
    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)
    if session is not None:
        records = [r for r in records if r.get("session") == session]
    return sorted(records, key=lambda r: (r.get("session", ""), r.get("pose_idx", 0)))


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m egotraj", description=__doc__.split("\n")[0])
    ap.add_argument("h5", help="path to egotraj_dataset.h5")
    ap.add_argument("--session", help="describe one session instead of listing all")
    a = ap.parse_args(argv)
    if a.session:
        s = load_session(a.h5, a.session)
        print(f"{s.name}: {len(s)} samples, {s.duration:.1f} s, "
              f"video={'yes' if s.video_segment is not None else 'no'}, attrs={s.attrs}")
        return 0
    names = list_sessions(a.h5)
    print(f"{len(names)} sessions")
    for name in names:
        s = load_session(a.h5, name)
        print(f"  {name}  {len(s):7d} samples  {s.duration:8.1f} s")
    return 0
