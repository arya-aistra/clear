"""Phase 0 tests for the Silero teacher wrapper.

Marked ``requires_silero`` — skipped automatically if the teacher can't be loaded.
Covers: output shapes, probability range, batch, state reset/carry, streaming
equivalence (vs the model's own internal full-audio path), and a 5s smoke test.
"""

import numpy as np
import pytest

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE

pytestmark = pytest.mark.requires_silero


def _noise(n, scale=0.05, seed=0):
    return (np.random.default_rng(seed).standard_normal(n).astype(np.float32) * scale)


def test_forward_shape_single(silero_vad):
    out = silero_vad.forward(_noise(CHUNK_SAMPLES))
    assert tuple(out.shape) == (1,)
    assert 0.0 <= float(out[0]) <= 1.0


def test_forward_shape_batch(silero_vad):
    import torch

    batch = torch.from_numpy(np.stack([_noise(CHUNK_SAMPLES, seed=i) for i in range(4)]))
    out = silero_vad.forward(batch)
    assert tuple(out.shape) == (4,)
    silero_vad.reset_states(1)  # restore default batch for later tests


def test_reset_determinism(silero_vad):
    wav = _noise(SAMPLE_RATE, seed=1)
    p1 = silero_vad.probabilities(wav, reset=True)
    p2 = silero_vad.probabilities(wav, reset=True)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_state_carry_changes_state(silero_vad):
    silero_vad.reset_states(1)
    s0 = silero_vad.get_state()
    silero_vad.forward(_noise(CHUNK_SAMPLES, seed=2))
    s1 = silero_vad.get_state()
    # tensors present in both snapshots with matching shape
    comparable = [
        k for k in s1
        if hasattr(s1[k], "shape") and k in s0 and hasattr(s0[k], "shape")
        and tuple(s1[k].shape) == tuple(s0[k].shape)
    ]
    if not comparable:
        # Backend (e.g. v5 JIT) does not expose its recurrent state as introspectable
        # tensors. State carry is still proven behaviorally by test_streaming_equivalence
        # (manual chunk streaming == the model's internal full-audio path).
        pytest.skip("backend hides recurrent state; carry covered by streaming-equivalence")
    changed = any(not np.allclose(np.asarray(s1[k]), np.asarray(s0[k])) for k in comparable)
    assert changed, "expected carried state to change after a forward pass"


def test_reset_clears_state(silero_vad):
    wav = _noise(SAMPLE_RATE, seed=3)
    silero_vad.probabilities(wav, reset=True)
    silero_vad.reset_states(1)
    after_reset = silero_vad.get_state()
    # a freshly reset state should be (near) zero where it's a tensor
    for k, v in after_reset.items():
        if hasattr(v, "shape") and "context" not in k:
            assert float(np.abs(np.asarray(v)).sum()) < 1e-3 or v.size == 0


def test_streaming_equivalence(silero_vad):
    """Manual 512-chunk streaming must match the model's internal full-audio path."""
    wav = _noise(SAMPLE_RATE * 2, seed=4)
    manual = silero_vad.probabilities(wav, reset=True)

    model = silero_vad._model
    if not hasattr(model, "audio_forward"):
        pytest.skip("backend has no audio_forward; determinism covered elsewhere")
    import torch

    silero_vad.reset_states(1)
    with torch.no_grad():
        full = torch.as_tensor(
            model.audio_forward(torch.from_numpy(wav).unsqueeze(0), SAMPLE_RATE)
        ).reshape(-1).cpu().numpy()
    n = min(len(manual), len(full))
    assert n > 0
    np.testing.assert_allclose(manual[:n], full[:n], atol=5e-3)


def test_smoke_5s(silero_vad):
    wav = _noise(SAMPLE_RATE * 5, seed=5)
    probs = silero_vad.probabilities(wav, reset=True)
    assert len(probs) == (SAMPLE_RATE * 5) // CHUNK_SAMPLES
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_describe_has_backend(silero_vad):
    info = silero_vad.describe()
    assert "backend" in info and info["sample_rate"] == SAMPLE_RATE
