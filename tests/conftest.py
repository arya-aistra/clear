"""Shared pytest fixtures + capability gating.

Tests are split into three tiers:
  * pure (numpy/scipy only)  -> run anywhere, always.
  * requires_torch           -> skipped if PyTorch missing.
  * requires_silero          -> skipped if the silero-vad teacher can't be loaded
                                (missing package OR no network for the weights).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the package importable when running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _silero_loadable() -> bool:
    if not _torch_available():
        return False
    try:
        from clearvad.model.silero_compat import SileroVAD

        SileroVAD(onnx=False)
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_TORCH = _torch_available()
HAVE_SILERO = _silero_loadable()


def pytest_collection_modifyitems(config, items):
    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    skip_silero = pytest.mark.skip(reason="silero-vad teacher not loadable (package/network)")
    for item in items:
        if "requires_silero" in item.keywords and not HAVE_SILERO:
            item.add_marker(skip_silero)
        if "requires_torch" in item.keywords and not HAVE_TORCH:
            item.add_marker(skip_torch)


@pytest.fixture(scope="session")
def silero_vad():
    from clearvad.model.silero_compat import SileroVAD

    return SileroVAD(onnx=False)
