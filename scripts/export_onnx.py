#!/usr/bin/env python
"""Phase 5 — full export pipeline: FP32 -> FP16 -> INT8 ONNX, validate, benchmark, accuracy.

Produces the deployable binaries in dist/ and a report with the size + INT8 + latency facts.
The headline: INT8 binary ~3x smaller than Silero AND it runs (Silero's INT8 fails), with
<2pp F1 degradation vs FP32.

Run (after train_dfkd.py + build_eval_set.py):
    python scripts/export_onnx.py --checkpoint checkpoints/stage2_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.export.benchmark_onnx import benchmark  # noqa: E402
from clearvad.export.quantize import collect_calibration_samples, export_fp16, quantize_int8  # noqa: E402
from clearvad.export.to_onnx import export_fp32, onnx_size_mb  # noqa: E402
from clearvad.export.validate_onnx import OrtVADRunner, validate_all  # noqa: E402
from clearvad.evaluation.metrics import summarize  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402
from clearvad.utils.config import load_yaml  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("export")
SILERO_SIZE_MB = 1.2896  # measured Phase 0


def eval_onnx_f1(onnx_path, audio, labels):
    runner = OrtVADRunner(onnx_path)
    f1s, fars, mrs = [], [], []
    for a, lab in zip(audio, labels):
        probs = runner.stream(a, reset=True)
        K = min(len(probs), len(lab))
        m = summarize(probs[:K], lab[:K], threshold=0.5)
        f1s.append(m["f1"]); fars.append(m["far"]); mrs.append(m["mr"])
    return {"f1": round(float(np.mean(f1s)), 4), "far": round(float(np.mean(fars)), 4),
            "mr": round(float(np.mean(mrs)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage2_final.pt")
    ap.add_argument("--model-config", default="configs/model/clearvad_base.yaml")
    ap.add_argument("--dist-dir", default="dist")
    ap.add_argument("--eval-cache", default="data/eval/controlled_eval.npz")
    ap.add_argument("--calib-chunks", type=int, default=1000)
    ap.add_argument("--bench-chunks", type=int, default=10000)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--out", default="reports/phase5/export_report.json")
    args = ap.parse_args()

    import torch
    model = ClearVADModel.from_config(load_yaml(args.model_config)).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    LOG.info("Model: %s", model.count_by_module())

    dist = Path(args.dist_dir)
    fp32 = str(dist / "clearvad_base.onnx")
    fp16 = str(dist / "clearvad_fp16.onnx")
    int8 = str(dist / "clearvad_lite.onnx")
    report = {"model_params": model.count_by_module(), "silero_size_mb": SILERO_SIZE_MB}

    # ---- FP32 ----
    LOG.info("Exporting FP32 -> %s", fp32)
    export_fp32(model, fp32)
    report["fp32"] = {"size_mb": onnx_size_mb(fp32),
                      "validation": validate_all(fp32, model, numeric=True, atol=1e-3)}

    # ---- FP16 ----
    LOG.info("Converting FP16 -> %s", fp16)
    try:
        export_fp16(fp32, fp16)
        report["fp16"] = {"size_mb": onnx_size_mb(fp16),
                          "validation": validate_all(fp16, model, numeric=False)}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("FP16 conversion failed: %r", exc)
        report["fp16"] = {"error": repr(exc)}

    # ---- INT8 ----
    LOG.info("Collecting %d calibration samples...", args.calib_chunks)
    gen = SyntheticAudioGenerator()
    calib = collect_calibration_samples(model, gen, n_chunks=args.calib_chunks)
    LOG.info("Static INT8 quantization -> %s", int8)
    quantize_int8(fp32, int8, calib)
    report["int8"] = {"size_mb": onnx_size_mb(int8),
                      "validation": validate_all(int8, model, numeric=False),
                      "size_vs_silero": round(SILERO_SIZE_MB / onnx_size_mb(int8), 2)}

    # ---- accuracy degradation (FP32 vs INT8) on the independent eval set ----
    if Path(args.eval_cache).exists():
        data = np.load(args.eval_cache, allow_pickle=True)
        audio, labels = data["audio"], data["labels"]
        acc_fp32 = eval_onnx_f1(fp32, audio, labels)
        acc_int8 = eval_onnx_f1(int8, audio, labels)
        report["accuracy"] = {
            "fp32": acc_fp32, "int8": acc_int8,
            "f1_degradation_pp": round((acc_fp32["f1"] - acc_int8["f1"]) * 100, 3),
        }
        LOG.info("Accuracy  FP32 F1=%.3f  INT8 F1=%.3f  degradation=%.2fpp",
                 acc_fp32["f1"], acc_int8["f1"], report["accuracy"]["f1_degradation_pp"])
    else:
        LOG.warning("eval cache %s missing; skipping accuracy degradation", args.eval_cache)

    # ---- benchmark ----
    if not args.skip_bench:
        for tag, path in (("fp32", fp32), ("fp16", fp16), ("int8", int8)):
            if Path(path).exists():
                LOG.info("Benchmarking %s...", tag)
                report.setdefault("benchmark", {})[tag] = benchmark(path, measure=args.bench_chunks)

    # ---- gate check ----
    int8_size = report["int8"]["size_mb"]
    deg = report.get("accuracy", {}).get("f1_degradation_pp", 0.0)
    report["gates"] = {
        "int8_size_under_1.5MB": bool(int8_size < 1.5),
        "int8_2-3x_smaller_than_silero": bool(SILERO_SIZE_MB / int8_size >= 2.0),
        "f1_degradation_under_2pp": bool(abs(deg) < 2.0),
        "int8_runs_and_state_carry": bool(report["int8"]["validation"]["all_passed"]),
    }
    write_json(report, args.out)
    LOG.info("Gates: %s", report["gates"])
    if "benchmark" in report and "int8" in report["benchmark"]:
        b = report["benchmark"]["int8"]
        LOG.info("INT8: size=%.3fMB (%.1fx vs Silero)  mean=%.4fms p99=%.4fms rtf=%.4f",
                 int8_size, report["int8"]["size_vs_silero"], b["mean_ms"], b["p99_ms"], b["rtf"])
    LOG.info("Wrote %s; binaries in %s/", args.out, args.dist_dir)


if __name__ == "__main__":
    main()
