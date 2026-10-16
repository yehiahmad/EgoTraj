"""EgoTraj: Real-World Egocentric Human Trajectory Dataset for Multimodal Prediction (ECCV 2026).

    from egotraj import list_sessions, load_session, load_annotations
"""
from .loader import Session, iter_sessions, list_sessions, load_annotations, load_session

__all__ = ["Session", "list_sessions", "load_session", "iter_sessions", "load_annotations"]
__version__ = "1.0.0"
