"""ClearVAD model package.

Phase 0 ships only the Silero compatibility shim (the teacher wrapper).
The native ClearVAD blocks (frontend, encoder, gssm, head) arrive in Phases 1-2.
"""

from clearvad.model.silero_compat import SileroVAD, load_silero  # noqa: F401

__all__ = ["SileroVAD", "load_silero"]
