#!/usr/bin/env python
"""Assemble the Hugging Face upload folder: copy the final ONNX binaries into hf/, validate each
loads + runs one streaming step, and write a manifest. Run after the binaries are present in dist/.

    python scripts/prepare_hf.py
    # then: cd hf && huggingface-cli upload <user>/clearvad-cfc . .
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad.utils.logging_utils import get_logger  # noqa: E402

LOG = get_logger("prepare-hf")
WIN = 576
BINARIES = {"clearvad_lite.onnx": "int8", "clearvad_fp16.onnx": "fp16",
            "clearvad_base.onnx": "fp32"}


def _validate(path: Path) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    st = sess.get_inputs()[1].shape
    state_shape = [d if isinstance(d, int) and d > 0 else 1 for d in st]
    prob, new_state = sess.run(None, {"chunk": np.zeros((1, WIN), np.float32),
                                      "state": np.zeros(state_shape, np.float32)})
    return {"size_mb": round(path.stat().st_size / 1e6, 4),
            "state_shape": state_shape,
            "ok": bool(np.isfinite(prob).all() and np.isfinite(new_state).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--hf-dir", default="hf")
    ap.add_argument("--config", default="configs/model/liquidvad_l2.yaml")
    args = ap.parse_args()

    hf = Path(args.hf_dir)
    hf.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}, "config": Path(args.config).name}

    for fname, prec in BINARIES.items():
        src = Path(args.dist) / fname
        if not src.exists():
            LOG.warning("MISSING %s — run scripts/export_onnx.py first (or pull from server)", src)
            continue
        shutil.copy2(src, hf / fname)
        info = _validate(hf / fname)
        manifest["files"][fname] = {"precision": prec, **info}
        LOG.info("%s (%s): %.3f MB state=%s ok=%s", fname, prec, info["size_mb"],
                 info["state_shape"], info["ok"])

    cfg = Path(args.config)
    if cfg.exists():
        shutil.copy2(cfg, hf / cfg.name)

    if not manifest["files"]:
        raise SystemExit(f"No binaries copied. Export/pull them into {args.dist}/ first.")
    (hf / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("HF folder ready at %s/ : %s", hf, list(manifest["files"]))
    LOG.info("Upload: cd %s && huggingface-cli upload <user>/clearvad-cfc . .", hf)


if __name__ == "__main__":
    main()
