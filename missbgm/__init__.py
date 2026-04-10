"""Standalone MissBGM package."""

__all__ = []

try:
    from .models import BGM, MissBGM

    __all__ += ["BGM", "MissBGM"]
except Exception:
    # Allow lightweight utilities such as dataset staging to be imported in non-TensorFlow envs.
    BGM = None
    MissBGM = None
