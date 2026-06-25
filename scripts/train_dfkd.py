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
    # real-audio source (recommended). Without these, training is pure synthetic.
    ap.add_argument("--use-real", action="store_true", help="use real speech for distillation")
    ap.add_argument("--speech-source", default="torchaudio",
                    choices=["torchaudio", "hf", "local"],
                    help="torchaudio LibriSpeech (default, most reliable), hf, or local")
    ap.add_argument("--ls-url", default="dev-clean", help="torchaudio LibriSpeech subset")
    ap.add_argument("--ls-root", default="data/librispeech")
    ap.add_argument("--hf-dataset", default="openslr/librispeech_asr")
    ap.add_argument("--hf-config", default="clean")
    ap.add_argument("--hf-split", default="train.clean.100")
    ap.add_argument("--local-speech-dir", default=None, help="use a local wav/flac dir")
    ap.add_argument("--buffer-seconds", type=float, default=1800.0)
    # FRAME-ACCURATE LABELS (Flag-1 fix): force-align training speech so intra-speech pauses are
    # labeled silence -> trains a real frame-level VAD, not a speech-region detector.
    ap.add_argument("--aligned-labels", action="store_true",
                    help="force-align training speech for frame-accurate labels (constructed mode)")
    ap.add_argument("--align-min-silence-ms", type=float, default=100.0,
                    help="fill interior silence gaps shorter than this (co-articulation, not pauses)")
    ap.add_argument("--real-fraction", type=float, default=None, help="override real_fraction")
    ap.add_argument("--pos-weight", type=float, default=None, help="override speech-class pos_weight (both stages)")
    ap.add_argument("--no-amp", action="store_true")
    # distillation teacher (for synthetic/distill data mode + transfer-eval reference)
    ap.add_argument("--teacher", default="silero",
                    choices=["silero", "nemo", "multi-nemo", "multi-firered"],
                    help="teacher for distill mode (constructed mode uses ground-truth labels)")
    ap.add_argument("--silero-weight", type=float, default=0.5, help="ensemble Silero weight")
    ap.add_argument("--second-weight", type=float, default=0.5, help="ensemble 2nd-teacher weight")
    # ACCURACY TRACK: constructed ground-truth-labeled data (real speech + real silence gaps)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "constructed"],
                    help="'constructed' = supervised on true labels from construction (beats Silero)")
    ap.add_argument("--aux-teacher", default="none", choices=["none", "silero", "nemo", "firered"],
                    help="optional soft-label teacher blended with true labels (constructed mode)")
    ap.add_argument("--true-weight", type=float, default=1.0,
                    help="constructed mode: weight on TRUE labels (1.0=pure supervised; <1 blends aux teacher)")
    ap.add_argument("--label-smooth", type=float, default=0.03, help="label smoothing on true labels")
    # REAL-WORLD ROBUSTNESS: real noise (MUSAN) mixed into constructed clips (speech-in-noise)
    ap.add_argument("--noise-source", default="none", choices=["none", "musan", "local", "hf"],
                    help="real noise for constructed data: MUSAN (OpenSLR), local dir, or HF repo")
    ap.add_argument("--noise-dir", default=None, help="local noise wav/flac dir")
    ap.add_argument("--noise-hf-repo", default="voice-biomarkers/DEMAND-acoustic-noise",
                    help="HF noise dataset id when --noise-source hf")
    ap.add_argument("--noise-buffer-seconds", type=float, default=1800.0)
    args = ap.parse_args()

    set_global_seed(1234)
    model_cfg = load_yaml(args.model_config)
    model = ClearVADModel.from_config(model_cfg)
    LOG.info("Model params by module: %s", model.count_by_module())

    if args.teacher == "nemo":
        from clearvad.distill.nemo_teacher import NeMoMarbleTeacher
        LOG.info("Teacher: NeMo MarbleNet (alone)")
        teacher = NeMoMarbleTeacher()
    elif args.teacher == "multi-nemo":
        from clearvad.distill.multi_teacher import MultiTeacher
        LOG.info("Multi-teacher: Silero(%.2f) + NeMo(%.2f)", args.silero_weight, args.second_weight)
        teacher = MultiTeacher.silero_nemo(args.silero_weight, args.second_weight)
    elif args.teacher == "multi-firered":
        from clearvad.distill.multi_teacher import MultiTeacher
        LOG.info("Multi-teacher: Silero(%.2f) + FireRedVAD(%.2f)", args.silero_weight, args.second_weight)
        teacher = MultiTeacher.silero_firered(args.silero_weight, args.second_weight)
    else:
        teacher = SileroTeacher()
    gen = SyntheticAudioGenerator()

    real_source = None
    if args.use_real or args.local_speech_dir or args.aligned_labels:
        from clearvad.distill.real_data import RealSpeechSource
        source = "local" if args.local_speech_dir else args.speech_source
        real_source = RealSpeechSource(
            source=source, local_dir=args.local_speech_dir,
            ls_url=args.ls_url, ls_root=args.ls_root,
            hf_dataset=args.hf_dataset, hf_config=args.hf_config, hf_split=args.hf_split,
            buffer_seconds=args.buffer_seconds,
            aligned=args.aligned_labels, align_min_silence_ms=args.align_min_silence_ms,
            align_device=args.device)

    trainer = DFKDTrainer(model, teacher, gen, device=args.device, out_dir=args.out_dir,
                          real_source=real_source, use_amp=not args.no_amp)

    summary = {"model_params": model.count_by_module(), "real_source": bool(real_source)}

    def _apply_overrides(cfg, steps_override):
        if steps_override is not None:
            cfg["steps"] = steps_override
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        if args.real_fraction is not None:
            cfg["real_fraction"] = args.real_fraction
        if args.pos_weight is not None:
            cfg["pos_weight"] = args.pos_weight
        if real_source is None:
            cfg["real_fraction"] = 0.0  # no source -> pure synthetic
        return cfg

    # ACCURACY TRACK: constructed ground-truth-labeled pool (real speech + real silence gaps).
    make_pool = (lambda cfg: None)
    if args.data == "constructed":
        if real_source is None:
            raise SystemExit("--data constructed requires real speech: pass --use-real (or --local-speech-dir)")
        aux = None
        if args.aux_teacher == "silero":
            aux = teacher if isinstance(teacher, SileroTeacher) else SileroTeacher()
        elif args.aux_teacher == "nemo":
            from clearvad.distill.nemo_teacher import NeMoMarbleTeacher
            aux = NeMoMarbleTeacher()
        elif args.aux_teacher == "firered":
            from clearvad.distill.firered_teacher import FireRedVADTeacher
            aux = FireRedVADTeacher()
        noise_source = None
        if args.noise_source != "none" or args.noise_dir:
            from clearvad.distill.real_noise import RealNoiseSource
            if args.noise_dir:
                noise_source = RealNoiseSource(source="local", local_dir=args.noise_dir,
                                               buffer_seconds=args.noise_buffer_seconds)
            elif args.noise_source == "hf":
                noise_source = RealNoiseSource(source="hf", hf_repo=args.noise_hf_repo,
                                               buffer_seconds=args.noise_buffer_seconds)
            else:
                noise_source = RealNoiseSource(source="openslr",
                                               buffer_seconds=args.noise_buffer_seconds)
        from clearvad.distill.constructed_data import ConstructedDataPool
        LOG.info("DATA=constructed (true_weight=%.2f, aux_teacher=%s, label_smooth=%.3f, noise=%s)",
                 args.true_weight, args.aux_teacher, args.label_smooth,
                 args.noise_dir or args.noise_source)

        def make_pool(cfg):
            return ConstructedDataPool(
                real_source, gen, pool_size=int(cfg.get("pool_size", 2048)),
                clip_chunks=int(cfg.get("chunks_per_sample", 64)), teacher=aux,
                true_weight=args.true_weight, label_smooth=args.label_smooth,
                noise_source=noise_source)

    s1_cfg = _apply_overrides(load_yaml(args.stage1), args.stage1_steps)
    summary["stage1"] = trainer.run_stage(s1_cfg, stage_name="stage1", pool=make_pool(s1_cfg))

    if not args.skip_stage2:
        s2_cfg = _apply_overrides(load_yaml(args.stage2), args.stage2_steps)
        init = str(Path(args.out_dir) / "stage1_final.pt")
        summary["stage2"] = trainer.run_stage(s2_cfg, stage_name="stage2", init_from=init,
                                              pool=make_pool(s2_cfg))

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
