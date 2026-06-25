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
    from clearvad.evaluation.metrics import pr_auc, roc_auc, tpr_at_fpr
    per_seq, gap_det = [], []
    pooled_p, pooled_l = [], []     # pool all frames for threshold-free metrics (standard practice)
    for audio, labels, gaps in zip(seqs_audio, seqs_labels, seqs_gaps):
        probs = prob_fn(audio)
        K = min(len(probs), len(labels))
        per_seq.append(summarize(probs[:K], labels[:K], threshold=thr))
        gap_det.append(short_silence_detection(probs_to_labels(probs[:K], thr),
                                               [(s, e, d) for (s, e, d) in gaps]))
        pooled_p.append(np.asarray(probs[:K])); pooled_l.append(np.asarray(labels[:K]))
    agg = aggregate(per_seq, ["f1", "far", "mr", "onset_latency_mean_ms",
                              "endpoint_latency_mean_ms"])
    pp = np.concatenate(pooled_p); pl = np.concatenate(pooled_l).astype(bool)
    # threshold-free metrics pooled over all frames (comparable to published VAD numbers)
    agg["auroc"] = round(roc_auc(pp, pl), 4)
    agg["pr_auc"] = round(pr_auc(pp, pl), 4)
    agg["tpr_at_fpr0.315"] = round(tpr_at_fpr(pp, pl, 0.315), 4)
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
    ap.add_argument("--webrtc", action="store_true", help="add WebRTC VAD baseline (pip webrtcvad)")
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
    models = {"silero": sil_agg, "clearvad": cv_agg}
    sslist = {"silero": sil_ss, "clearvad": cv_ss}

    # optional WebRTC baseline (binary; AUROC/PR-AUC undefined for hard 0/1 output)
    if args.webrtc:
        try:
            LOG.info("Scoring WebRTC VAD baseline...")
            from clearvad.evaluation.webrtc_baseline import webrtc_probs
            w_agg, w_ss = eval_model(lambda a: webrtc_probs(a), audio, labels, gaps)
            models["webrtc"] = w_agg
            sslist["webrtc"] = w_ss
        except Exception as exc:  # noqa: BLE001
            LOG.warning("WebRTC baseline skipped: %r", exc)

    result = {"clearvad_params": cv_params, "clearvad_fp32_size_mb": cv_size_mb,
              "silero_params": 309633, "silero_fp32_size_mb": 1.2896,
              "models": {m: {"metrics": models[m], "short_silence": sslist[m]} for m in models}}
    write_json(result, args.out)

    # human-readable multi-model table
    cols = list(models.keys())
    metric_keys = ["f1", "auroc", "pr_auc", "tpr_at_fpr0.315", "far", "mr",
                   "onset_latency_mean_ms", "endpoint_latency_mean_ms"]
    md = ["# VAD comparison — independent eval set\n",
          "| metric | " + " | ".join(cols) + " |",
          "|" + "---|" * (len(cols) + 1)]
    for k in metric_keys:
        md.append(f"| {k} | " + " | ".join(str(models[m].get(k)) for m in cols) + " |")
    md.append("\n## Short-silence detection (fraction of true gaps detected)\n")
    md.append("| gap | " + " | ".join(cols) + " |")
    md.append("|" + "---|" * (len(cols) + 1))
    all_durs = sorted({d for s in sslist.values() for d in s})
    for d in all_durs:
        md.append(f"| {d} | " + " | ".join(str(sslist[m].get(d)) for m in cols) + " |")
    Path(args.out).with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    LOG.info("Silero   : F1=%.3f FAR=%.3f MR=%.3f endpoint=%.1fms params=309,633 size=1.29MB",
             sil_agg["f1"], sil_agg["far"], sil_agg["mr"], sil_agg["endpoint_latency_mean_ms"])
    LOG.info("ClearVAD : F1=%.3f FAR=%.3f MR=%.3f endpoint=%.1fms params=%d size=%.2fMB",
             cv_agg["f1"], cv_agg["far"], cv_agg["mr"], cv_agg["endpoint_latency_mean_ms"],
             cv_params, cv_size_mb)
    LOG.info("Wrote %s (+ .md)", args.out)


if __name__ == "__main__":
    main()
