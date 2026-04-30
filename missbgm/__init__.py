"""Standalone MissBGM package."""

from ._version import __version__

__all__ = []

try:
    from .models import BGM, MissBGM

    __all__ += ["MissBGM", "__version__"]
except Exception:
    # Allow lightweight utilities such as dataset staging to be imported in non-TensorFlow envs.
    BGM = None
    MissBGM = None

    __all__ += ["__version__"]
