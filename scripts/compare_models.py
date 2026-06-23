#!/usr/bin/env python
"""Head-to-head: ClearVAD vs Silero on the controlled INDEPENDENT eval set.

Loads the cached eval set (construction-based ground truth, NOT Silero's labels) and scores
both models identically: frame F1/FAR/MR, onset/endpoint latency, short-silence detection,
plus params + FP32 size. This is the first concrete ClearVAD-vs-Silero comparison.

Run (after build_eval_set.py + train_dfkd.py):
    python scripts/compare_models.py --checkpoint checkpoints/stage2_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES  # noqa: E402
from clearvad.evaluation.eval_dataset import short_silence_detection  # noqa: E402
from clearvad.evaluation.metrics import probs_to_labels, summarize  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.utils.config import load_yaml  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("compare")
WIN = CONTEXT_SAMPLES + CHUNK_SAMPLES  # 576


def build_windows(audio_1d: np.ndarray) -> torch.Tensor:
    a = torch.from_numpy(np.ascontiguousarray(audio_1d, dtype=np.float32))
    K = len(a) // CHUNK_SAMPLES
    padded = torch.cat([a.new_zeros(CONTEXT_SAMPLES), a])
    windows = torch.stack([padded[i * CHUNK_SAMPLES:i * CHUNK_SAMPLES + WIN] for i in range(K)])
    return windows.unsqueeze(0)  # [1, K, 576]


@torch.no_grad()
def clearvad_probs(model, audio_1d):
    windows = build_windows(audio_1d)
    probs = model.forward_sequence(windows, return_logit=False, scan_mode="loop")  # [1,K]
    return probs[0].cpu().numpy()


def aggregate(dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and d[k] == d[k]]
        out[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return out


def eval_model(prob_fn, seqs_audio, seqs_labels, seqs_gaps, thr=0.5):
    per_seq, gap_det = [], []
    for audio, labels, gaps in zip(seqs_audio, seqs_labels, seqs_gaps):
        probs = prob_fn(audio)
        K = min(len(probs), len(labels))
        per_seq.append(summarize(probs[:K], labels[:K], threshold=thr))
        gap_det.append(short_silence_detection(probs_to_labels(probs[:K], thr),
                                               [(s, e, d) for (s, e, d) in gaps]))
    agg = aggregate(per_seq, ["f1", "far", "mr", "auc",
                              "onset_latency_mean_ms", "endpoint_latency_mean_ms"])
    durs = sorted({d for gd in gap_det for d in gd})
    ss = {f"{int(d)}ms": round(float(np.mean([gd[d] for gd in gap_det if d in gd])), 4)
          for d in durs}
    return agg, ss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage2_final.pt")
    ap.add_argument("--model-config", default="configs/model/clearvad_base.yaml")
    ap.add_argument("--cache", default="data/eval/controlled_eval.npz")
    ap.add_argument("--out", default="reports/phase8/comparison.json")
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    audio, labels = data["audio"], data["labels"]
    gaps = json.loads(str(data["gaps"]))
    LOG.info("Eval set: %d sequences", len(audio))

    # Silero
    LOG.info("Scoring Silero...")
    silero = SileroVAD(onnx=False)
    sil_agg, sil_ss = eval_model(lambda a: silero.probabilities(a, reset=True),
                                 audio, labels, gaps)

    # ClearVAD
    LOG.info("Scoring ClearVAD (%s)...", args.checkpoint)
    model = ClearVADModel.from_config(load_yaml(args.model_config)).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    cv_agg, cv_ss = eval_model(lambda a: clearvad_probs(model, a), audio, labels, gaps)

    cv_params = model.parameter_count()
    cv_size_mb = round(cv_params * 4 / 1e6, 4)   # FP32
    result = {
        "silero": {"metrics": sil_agg, "short_silence": sil_ss,
                   "params": 309633, "fp32_size_mb": 1.2896},
        "clearvad_fp32": {"metrics": cv_agg, "short_silence": cv_ss,
                          "params": cv_params, "fp32_size_mb": cv_size_mb},
    }
    write_json(result, args.out)

    # human-readable table
    md = ["# ClearVAD vs Silero — independent eval set\n",
          "| metric | Silero | ClearVAD (FP32) |",
          "|--------|--------|-----------------|"]
    for k in ["f1", "far", "mr", "auc", "onset_latency_mean_ms", "endpoint_latency_mean_ms"]:
        md.append(f"| {k} | {sil_agg.get(k)} | {cv_agg.get(k)} |")
    md.append(f"| params | 309,633 | {cv_params:,} |")
    md.append(f"| FP32 size (MB) | 1.29 | {cv_size_mb} |")
    md.append("\n## Short-silence detection (fraction of gaps detected)\n")
    md.append("| gap | Silero | ClearVAD |")
    md.append("|-----|--------|----------|")
    for d in sorted(set(sil_ss) | set(cv_ss)):
        md.append(f"| {d} | {sil_ss.get(d)} | {cv_ss.get(d)} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    LOG.info("Silero   : F1=%.3f FAR=%.3f MR=%.3f endpoint=%.1fms params=309,633 size=1.29MB",
             sil_agg["f1"], sil_agg["far"], sil_agg["mr"], sil_agg["endpoint_latency_mean_ms"])
    LOG.info("ClearVAD : F1=%.3f FAR=%.3f MR=%.3f endpoint=%.1fms params=%d size=%.2fMB",
             cv_agg["f1"], cv_agg["far"], cv_agg["mr"], cv_agg["endpoint_latency_mean_ms"],
             cv_params, cv_size_mb)
    LOG.info("Wrote %s (+ .md)", args.out)


if __name__ == "__main__":
    main()
