#!/usr/bin/env python
"""Phase 3 — run Data-Free Knowledge Distillation (Stage 1 then Stage 2).

Run (on the GPU server):
    python scripts/train_dfkd.py \
        --model-config configs/model/clearvad_base.yaml \
        --stage1 configs/train/dfkd_stage1.yaml \
        --stage2 configs/train/dfkd_stage2.yaml \
        [--real-audio path/to/real_speech.wav]   # optional transfer eval (honest-reframe)

Outputs: checkpoints/stage{1,2}_final.pt, reports/phase3/train_history.csv/json,
         reports/phase3/dfkd_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.distill.teacher import SileroTeacher  # noqa: E402
from clearvad.distill.trainer import DFKDTrainer  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402
from clearvad.utils.config import load_yaml, set_global_seed  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="configs/model/clearvad_base.yaml")
    ap.add_argument("--stage1", default="configs/train/dfkd_stage1.yaml")
    ap.add_argument("--stage2", default="configs/train/dfkd_stage2.yaml")
    ap.add_argument("--real-audio", default=None, help="optional real-speech wav for transfer eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--skip-stage2", action="store_true")
    ap.add_argument("--stage1-steps", type=int, default=None, help="override stage1 steps")
    ap.add_argument("--stage2-steps", type=int, default=None, help="override stage2 steps")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch_size (both stages)")
    args = ap.parse_args()

    set_global_seed(1234)
    model_cfg = load_yaml(args.model_config)
    model = ClearVADModel.from_config(model_cfg)
    LOG.info("Model params by module: %s", model.count_by_module())

    teacher = SileroTeacher()
    gen = SyntheticAudioGenerator()
    trainer = DFKDTrainer(model, teacher, gen, device=args.device, out_dir=args.out_dir)

    summary = {"model_params": model.count_by_module()}

    s1_cfg = load_yaml(args.stage1)
    if args.stage1_steps is not None:
        s1_cfg["steps"] = args.stage1_steps
    if args.batch_size is not None:
        s1_cfg["batch_size"] = args.batch_size
    summary["stage1"] = trainer.run_stage(s1_cfg, stage_name="stage1")

    if not args.skip_stage2:
        s2_cfg = load_yaml(args.stage2)
        if args.stage2_steps is not None:
            s2_cfg["steps"] = args.stage2_steps
        if args.batch_size is not None:
            s2_cfg["batch_size"] = args.batch_size
        init = str(Path(args.out_dir) / "stage1_final.pt")
        summary["stage2"] = trainer.run_stage(s2_cfg, stage_name="stage2", init_from=init)

    # Optional real-audio transfer eval (Claim 2 honesty check)
    if args.real_audio:
        from clearvad.utils.audio import load_audio
        wav = load_audio(args.real_audio)
        summary["transfer_real"] = trainer.evaluate_transfer_real(
            torch.from_numpy(wav).unsqueeze(0))
        LOG.info("Real-audio transfer: %s", summary["transfer_real"])

    trainer.dump_history("reports/phase3/train_history.csv", "reports/phase3/train_history.json")
    write_json(summary, "reports/phase3/dfkd_summary.json")
    LOG.info("DFKD summary: %s", summary)


if __name__ == "__main__":
    main()
