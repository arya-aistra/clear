#!/usr/bin/env python
"""Final cross-model VAD benchmark.

Scores every AVAILABLE VAD on the SAME frame-accurate eval cache(s) across the standard metrics
(AUROC, PR-AUC, TPR@FPR=0.315, F1, FAR, MR, onset/endpoint latency) + short-silence detection.
Models: clearvad (ours), silero, webrtc, pyannote, tenvad, fireredvad, nemo. Any model that is not
installed (or lacks an HF token) is reported in an "unavailable" section, not fatal.

Run:
  python scripts/benchmark_all.py \
    --checkpoint checkpoints_cfc_20h/stage2_final.pt --model-config configs/model/liquidvad_l2.yaml \
    --caches clean:data/eval/aligned_eval_pad40.npz noisy:data/eval/aligned_eval_noisy.npz \
    --models clearvad silero webrtc pyannote tenvad nemo fireredvad
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
from clearvad.evaluation.external_vads import REGISTRY  # noqa: E402
from clearvad.evaluation.metrics import (pr_auc, probs_to_labels, roc_auc,  # noqa: E402
                                         summarize, tpr_at_fpr)
from clearvad.model.factory import build_model  # noqa: E402
from clearvad.utils.config import load_yaml  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("benchmark")
WIN = CONTEXT_SAMPLES + CHUNK_SAMPLES
PARAMS = {"silero": 309633}
METRIC_KEYS = ["auroc", "pr_auc", "f1", "tpr_at_fpr0.315", "far", "mr",
               "onset_latency_mean_ms", "endpoint_latency_mean_ms"]


def clearvad_prob_fn(model):
    @torch.no_grad()
    def fn(audio):
        a = torch.from_numpy(np.ascontiguousarray(audio, np.float32))
        K = len(a) // CHUNK_SAMPLES
        padded = torch.cat([a.new_zeros(CONTEXT_SAMPLES), a])
        win = torch.stack([padded[i * CHUNK_SAMPLES:i * CHUNK_SAMPLES + WIN]
                           for i in range(K)]).unsqueeze(0)
        return model.forward_sequence(win, return_logit=False, scan_mode="loop")[0].cpu().numpy()
    return fn


def aggregate(dicts, keys):
    out = {}
    for k in keys:
        v = [d[k] for d in dicts if k in d and d[k] == d[k]]
        out[k] = round(float(np.mean(v)), 4) if v else float("nan")
    return out


def eval_model(prob_fn, audio, labels, gaps, thr=0.5):
    per_seq, gap_det, pooled_p, pooled_l = [], [], [], []
    for a, lab, gg in zip(audio, labels, gaps):
        probs = np.asarray(prob_fn(np.asarray(a, np.float32)), dtype=np.float32)
        K = min(len(probs), len(lab))
        per_seq.append(summarize(probs[:K], np.asarray(lab[:K]), threshold=thr))
        gap_det.append(short_silence_detection(probs_to_labels(probs[:K], thr),
                                               [(s, e, d) for (s, e, d) in gg]))
        pooled_p.append(probs[:K]); pooled_l.append(np.asarray(lab[:K]))
    agg = aggregate(per_seq, ["f1", "far", "mr", "onset_latency_mean_ms", "endpoint_latency_mean_ms"])
    P = np.concatenate(pooled_p); L = np.concatenate(pooled_l).astype(bool)
    agg["auroc"] = round(roc_auc(P, L), 4)
    agg["pr_auc"] = round(pr_auc(P, L), 4)
    agg["tpr_at_fpr0.315"] = round(tpr_at_fpr(P, L, 0.315), 4)
    durs = sorted({d for gd in gap_det for d in gd})
    ss = {f"{int(d)}ms": round(float(np.mean([gd[d] for gd in gap_det if d in gd])), 4) for d in durs}
    return agg, ss


def build_prob_fns(args):
    fns, notes = {}, {}
    want = args.models
    if "clearvad" in want:
        m = build_model(load_yaml(args.model_config)).eval()
        m.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        fns["clearvad"] = clearvad_prob_fn(m)
        PARAMS["clearvad"] = m.parameter_count()
    if "silero" in want:
        try:
            from clearvad.model.silero_compat import SileroVAD
            sv = SileroVAD(onnx=False)
            fns["silero"] = lambda a, sv=sv: sv.probabilities(a, reset=True)
        except Exception as exc:  # noqa: BLE001
            notes["silero"] = repr(exc)
    if "webrtc" in want:
        try:
            from clearvad.evaluation.webrtc_baseline import webrtc_probs
            fns["webrtc"] = lambda a: webrtc_probs(a)
        except Exception as exc:  # noqa: BLE001
            notes["webrtc"] = repr(exc)
    for name in want:
        if name in REGISTRY:
            try:
                ad = REGISTRY[name]()
                ad.load()
                fns[name] = (lambda ad: (lambda a: ad.probs(a)))(ad)
                LOG.info("model %s: loaded", name)
            except Exception as exc:  # noqa: BLE001
                notes[name] = repr(exc)
                LOG.warning("model %s UNAVAILABLE: %r", name, exc)
    return fns, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_cfc_20h/stage2_final.pt")
    ap.add_argument("--model-config", default="configs/model/liquidvad_l2.yaml")
    ap.add_argument("--caches", nargs="+",
                    default=["clean:data/eval/aligned_eval_pad40.npz",
                             "noisy:data/eval/aligned_eval_noisy.npz"])
    ap.add_argument("--models", nargs="+",
                    default=["clearvad", "silero", "webrtc", "pyannote", "tenvad", "nemo",
                             "fireredvad"])
    ap.add_argument("--out", default="reports/phase8/benchmark_all.json")
    ap.add_argument("--md", default="reports/phase8/benchmark_all.md")
    args = ap.parse_args()

    fns, notes = build_prob_fns(args)
    LOG.info("Available models: %s", list(fns))
    cols = list(fns)
    result = {"models_available": cols, "unavailable": notes, "params": {}, "caches": {}}
    md = ["# Final VAD benchmark — all available models on identical frame-accurate labels\n",
          "Same eval cache(s), same labels, same metrics for every model. ClearVAD = ours (CfC core)."]

    for spec in args.caches:
        cname, _, cpath = spec.partition(":")
        if not Path(cpath).exists():
            LOG.warning("cache %s missing — skipping", cpath)
            continue
        data = np.load(cpath, allow_pickle=True)
        audio, labels = data["audio"], data["labels"]
        gaps = json.loads(str(data["gaps"]))
        LOG.info("[%s] %d sequences", cname, len(audio))
        per_model, ss_model = {}, {}
        for name, fn in fns.items():
            try:
                agg, ss = eval_model(fn, audio, labels, gaps)
                per_model[name], ss_model[name] = agg, ss
                LOG.info("  %-11s AUROC=%.4f F1=%.4f FAR=%.4f MR=%.4f",
                         name, agg["auroc"], agg["f1"], agg["far"], agg["mr"])
            except Exception as exc:  # noqa: BLE001
                notes[name] = f"eval failed on {cname}: {exc!r}"
                LOG.warning("  %s eval failed: %r", name, exc)
        result["caches"][cname] = {"metrics": per_model, "short_silence": ss_model}
        avail = [c for c in cols if c in per_model]
        md.append(f"\n## {cname}  ({len(audio)} sequences)\n")
        md.append("| metric | " + " | ".join(avail) + " |")
        md.append("|" + "---|" * (len(avail) + 1))
        for k in METRIC_KEYS:
            md.append(f"| {k} | " + " | ".join(str(per_model[c].get(k)) for c in avail) + " |")
        md.append("\n_Short-silence detection (fraction of true gaps detected):_\n")
        md.append("| gap | " + " | ".join(avail) + " |")
        md.append("|" + "---|" * (len(avail) + 1))
        durs = sorted({d for c in avail for d in ss_model[c]}, key=lambda s: int(s[:-2]))
        for d in durs:
            md.append(f"| {d} | " + " | ".join(str(ss_model[c].get(d, "")) for c in avail) + " |")

    for c in cols:
        result["params"][c] = PARAMS.get(c)
    md.append("\n## Params\n")
    md.append("| model | params |")
    md.append("|---|---|")
    for c in cols:
        md.append(f"| {c} | {PARAMS.get(c, '—')} |")
    if notes:
        md.append("\n## Unavailable / errored models\n")
        for n, e in notes.items():
            md.append(f"- **{n}**: {e}")

    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    write_json(result, args.out)
    LOG.info("Wrote %s and %s", args.md, args.out)


if __name__ == "__main__":
    main()
