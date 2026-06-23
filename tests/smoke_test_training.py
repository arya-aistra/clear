"""Phase 3 smoke test — short DFKD run must reduce the loss.

Not collected by default `pytest -q` (filename isn't test_*.py); run explicitly:
    pytest tests/smoke_test_training.py -v -s
or:
    python tests/smoke_test_training.py

Marked requires_silero (needs the teacher) + slow. Uses tiny settings so it finishes on CPU.
"""

import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.requires_silero, pytest.mark.slow]
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.distill.teacher import SileroTeacher  # noqa: E402
from clearvad.distill.trainer import DFKDTrainer  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402


def _run(steps=60):
    torch.manual_seed(0)
    model = ClearVADModel()
    teacher = SileroTeacher()
    gen = SyntheticAudioGenerator()
    out = tempfile.mkdtemp(prefix="dfkd_smoke_")
    trainer = DFKDTrainer(model, teacher, gen, device="cpu", out_dir=out)
    cfg = {
        "steps": steps, "batch_size": 8, "chunks_per_sample": 16, "lr": 1e-3,
        "weight_decay": 1e-4, "pool_size": 64, "refresh_every": 10_000,  # no mid-run refresh
        "label_batch": 64, "log_every": 5, "agree_every": 10_000, "ckpt_every": 0,
        "lambda_soft": 1.0, "lambda_boundary": 2.0, "lambda_smooth": 0.5, "seed": 0,
    }
    trainer.run_stage(cfg, stage_name="smoke")
    return trainer.history


def test_smoke_training_loss_decreases():
    history = _run(steps=60)
    assert len(history) >= 4
    totals = [r["total"] for r in history]
    first = totals[0]
    best_late = min(totals[-3:])
    print(f"\nsmoke loss: first={first:.4f} best_late={best_late:.4f}")
    assert best_late < first, f"loss did not decrease: {first:.4f} -> {best_late:.4f}"
    assert all(t == t for t in totals)  # no NaN


if __name__ == "__main__":
    h = _run(steps=60)
    print("history tail:", h[-3:])
