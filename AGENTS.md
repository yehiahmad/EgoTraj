# AGENTS.md

Guidance for coding agents working with EgoTraj.

## What this repository is

EgoTraj is a real-world egocentric human trajectory dataset recorded with Meta Quest Pro
headsets in urban environments ([ECCV 2026](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/14919.pdf)).
Each session has 6-DoF head pose, 3D eye gaze, privacy-blurred RGB video and per-second scene
annotations. This repository holds the loader, the EgoViz dashboard, the paper and its figures.
The data itself is downloaded separately.

## Getting the data

Download from [UT Box](https://utexas.box.com/s/kszvh58csvk8duu3qywqxp05a9fhsvph):

- `egotraj_dataset.h5`: pose, gaze and video pointers, one HDF5 group per session
- `egotraj_annotations.json`: scene annotations, one record per second
- `egotraj_videos_part1.zip` to `egotraj_videos_part5.zip`: video; all five parts are needed

Data files are git-ignored (`*.h5`, `*.csv`). Never commit them.

**Licenses differ:** the code in this repository is MIT, but the dataset is
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Do not use the data
commercially, and do not attempt to identify participants.

## Loading it

```bash
pip install git+https://github.com/yehiahmad/EgoTraj     # or: pip install -e ".[test]"
python -m egotraj egotraj_dataset.h5                     # list sessions
```

```python
from egotraj import list_sessions, load_session, load_annotations
s = load_session("egotraj_dataset.h5", list_sessions("egotraj_dataset.h5")[0])
s.position, s.rotation, s.gaze_direction, s.timestamp   # numpy arrays, length N
```

## Conventions that matter

- **Y is the up axis.** The ground plane is `(x, z)`; `Session.ground_xz` returns it.
- **Positions are session-relative**, in metres, with a local origin per session. They are
  not geographic coordinates, so sessions cannot be overlaid on a map or on each other.
- `pose/timestamp` is epoch seconds.
- `pose/rotation` is a 4-component head-orientation quaternion, returned as stored.
- `video/has_video` is False where no video frame matches a pose sample; ignore
  `video/segment` and `video/frame` there.
- An annotation's `pose_idx` indexes its session's arrays:
  `load_session(h5, r["session"]).position[r["pose_idx"]]`.
- The loader never resamples, crops or splits. The paper's prediction setting
  (1.5 s observed, 3.5 s predicted) is applied by downstream code, not by the loader.

## Not in this repository

Train/validation/test split files, the ADE/FDE evaluation script and the EgoCast model
weights are not released here yet. Do not reconstruct the paper's splits by guessing; results
computed on ad-hoc splits are not comparable with the paper's tables.

## Working on the code

- Tests: `pip install -e ".[test,pandas]"` then `pytest -q`. They build a small synthetic HDF5
  file, so they need no download.
- `ego_viz.py` is a standalone Tkinter dashboard: `pip install -r requirements.txt`, then
  `python ego_viz.py --h5 egotraj_dataset.h5 --session <name> --videos-root videos/
  --annotations-json egotraj_annotations.json`.

## Citing

Use [`CITATION.cff`](CITATION.cff) or the BibTeX entry in the README.
