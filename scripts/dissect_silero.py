#!/usr/bin/env python
"""Phase 0.3 — instrument and document Silero VAD v5's architecture.

Discovers ground-truth tensor shapes (does NOT assume the spec's numbers) by:
  * enumerating the JIT model's named parameters (reveals LSTM weight_ih/weight_hh,
    conv front-end, encoder convs, head),
  * running a dummy 512-sample chunk and inspecting the carried state/context shapes,
  * loading the ONNX graph and listing every op with inferred I/O shapes,
  * inspecting the ONNX Runtime session inputs/outputs.

Writes:
  reports/silero_dissection.md   (human-readable)
  reports/silero_dissection.json (machine-readable, consumed by make_report.py)

Run:
    python scripts/dissect_silero.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("dissect")


def find_silero_onnx_path() -> Optional[str]:
    """Locate the silero_vad .onnx data file shipped with the pip package."""
    spec = find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        hits = sorted(glob.glob(str(Path(root) / "**" / "*.onnx"), recursive=True))
        # Prefer a 16k model if multiple are shipped.
        hits_16k = [h for h in hits if "16k" in Path(h).name.lower()]
        if hits_16k:
            return hits_16k[0]
        if hits:
            return hits[0]
    return None


def jit_parameter_table(model) -> List[Dict[str, Any]]:
    """Enumerate named parameters with shapes (JIT backend)."""
    rows: List[Dict[str, Any]] = []
    try:
        for name, p in model.named_parameters():
            rows.append({
                "name": name,
                "shape": list(p.shape),
                "numel": int(p.numel()),
                "dtype": str(p.dtype),
            })
    except Exception as exc:  # noqa: BLE001
        LOG.warning("named_parameters() failed: %r", exc)
    return rows


def onnx_graph_summary(onnx_path: str) -> Dict[str, Any]:
    """Parse the ONNX graph: op histogram + per-node I/O shapes (after shape inference)."""
    import onnx
    from onnx import shape_inference

    model = onnx.load(onnx_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("shape inference failed: %r", exc)
    graph = model.graph

    # value_info shape map
    def _vi_shape(vi) -> List[Any]:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
        return dims

    shapes: Dict[str, List[Any]] = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            shapes[vi.name] = _vi_shape(vi)
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)

    op_hist: Dict[str, int] = {}
    nodes: List[Dict[str, Any]] = []
    for node in graph.node:
        op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1
        nodes.append({
            "op": node.op_type,
            "name": node.name,
            "inputs": [{"n": i, "shape": shapes.get(i, "?")} for i in node.input],
            "outputs": [{"n": o, "shape": shapes.get(o, "?")} for o in node.output],
        })

    return {
        "opset": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
        "n_initializers": len(graph.initializer),
        "total_params": int(sum(int(np.prod(i.dims)) for i in graph.initializer)),
        "op_histogram": op_hist,
        "graph_inputs": [{"name": vi.name, "shape": _vi_shape(vi)} for vi in graph.input],
        "graph_outputs": [{"name": vi.name, "shape": _vi_shape(vi)} for vi in graph.output],
        "nodes": nodes,
    }


def run_dynamic_probe(vad: SileroVAD) -> Dict[str, Any]:
    """Feed a dummy chunk + 5s of audio; record output + state shapes."""
    import torch

    probe: Dict[str, Any] = {}
    vad.reset_states(batch_size=1)
    state_before = vad.get_state()
    dummy = torch.zeros(CHUNK_SAMPLES, dtype=torch.float32)
    out = vad.forward(dummy)
    state_after = vad.get_state()

    def _shape(v):
        return list(v.shape) if hasattr(v, "shape") else str(v)

    probe["single_chunk_input_shape"] = [CHUNK_SAMPLES]
    probe["single_chunk_output_shape"] = list(out.shape)
    probe["state_keys"] = sorted(k for k in state_after if k != "backend")
    probe["state_shapes"] = {
        k: _shape(v) for k, v in state_after.items() if k != "backend" and hasattr(v, "shape")
    }
    probe["state_present_before_first_forward"] = sorted(
        k for k in state_before if k not in ("backend",) and hasattr(state_before[k], "shape")
    )

    # 5-second streaming smoke
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(5 * SAMPLE_RATE).astype(np.float32) * 0.05
    probs = vad.probabilities(wav, reset=True)
    probe["five_second_num_chunks"] = int(len(probs))
    probe["five_second_expected_chunks"] = (5 * SAMPLE_RATE) // CHUNK_SAMPLES
    probe["five_second_prob_range"] = [float(probs.min()), float(probs.max())]
    return probe


def render_markdown(d: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Silero VAD v5 — Architecture Dissection\n")
    L.append("> Auto-generated by `scripts/dissect_silero.py`. All shapes are **measured**, "
             "not assumed. Compare against the GSD Phase 0.3 spec at the bottom.\n")

    L.append("## 1. Loaded model\n")
    desc = d.get("describe", {})
    for k, v in desc.items():
        if k in ("onnx_inputs", "onnx_outputs"):
            continue
        L.append(f"- **{k}**: `{v}`")
    L.append(f"- **JIT parameter count**: `{d.get('jit_param_count')}`\n")

    L.append("## 2. Dynamic probe (single chunk + 5s stream)\n")
    probe = d.get("dynamic_probe", {})
    for k, v in probe.items():
        L.append(f"- **{k}**: `{v}`")
    L.append("")

    rows = d.get("jit_parameters", [])
    if rows:
        L.append("## 3. JIT named parameters (per-layer weight shapes)\n")
        L.append("| name | shape | numel | dtype |")
        L.append("|------|-------|-------|-------|")
        for r in rows:
            L.append(f"| `{r['name']}` | `{r['shape']}` | {r['numel']:,} | {r['dtype']} |")
        L.append(f"\n**Total JIT params**: {sum(r['numel'] for r in rows):,}\n")

    onnx = d.get("onnx_graph")
    if onnx:
        L.append("## 4. ONNX graph\n")
        L.append(f"- **opset**: `{onnx['opset']}`")
        L.append(f"- **initializers**: {onnx['n_initializers']} "
                 f"(**~{onnx['total_params']:,}** params)")
        L.append(f"- **graph inputs**: `{onnx['graph_inputs']}`")
        L.append(f"- **graph outputs**: `{onnx['graph_outputs']}`\n")
        L.append("### Op histogram\n")
        L.append("| op | count |")
        L.append("|----|-------|")
        for op, c in sorted(onnx["op_histogram"].items(), key=lambda x: -x[1]):
            L.append(f"| {op} | {c} |")
        L.append("\n### Node-by-node I/O shapes\n")
        L.append("| # | op | inputs (shape) | outputs (shape) |")
        L.append("|---|----|----------------|-----------------|")
        for i, n in enumerate(onnx["nodes"]):
            ins = "; ".join(f"{x['shape']}" for x in n["inputs"])
            outs = "; ".join(f"{x['shape']}" for x in n["outputs"])
            L.append(f"| {i} | {n['op']} | {ins} | {outs} |")
        L.append("")

    if "onnx_inputs" in desc:
        L.append("## 5. ONNX Runtime session I/O\n")
        L.append("**Inputs:**\n")
        for i in desc["onnx_inputs"]:
            L.append(f"- `{i['name']}`  shape=`{i['shape']}`  type=`{i['type']}`")
        L.append("\n**Outputs:**\n")
        for o in desc.get("onnx_outputs", []):
            L.append(f"- `{o['name']}`  shape=`{o['shape']}`  type=`{o['type']}`")
        L.append("")

    L.append("## 6. GSD Phase 0.3 expected reference (for reconciliation)\n")
    L.append("| item | spec value | measured | match? |")
    L.append("|------|-----------|----------|--------|")
    L.append("| chunk size | 512 samples (32 ms) | see §2 | — |")
    L.append("| left context | 64 samples (4 ms) | see §2 state `context` | — |")
    L.append("| total input | 576 samples | derived | — |")
    L.append("| LSTM weight_ih | [512, 128] | see §3 | — |")
    L.append("| LSTM weight_hh | [512, 128] | see §3 | — |")
    L.append("| state h_n | [1, 1, 128] | see §2 | — |")
    L.append("| state c_n | [1, 1, 128] | see §2 | — |")
    L.append("\n> Fill the **measured** + **match?** columns from §2/§3 above. Any mismatch "
             "is a finding, not an error — reconcile before Phase 1.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default="reports/silero_dissection.md")
    ap.add_argument("--out-json", default="reports/silero_dissection.json")
    args = ap.parse_args()

    result: Dict[str, Any] = {}

    LOG.info("Loading Silero (JIT backend)...")
    vad_jit = SileroVAD(onnx=False)
    result["describe"] = vad_jit.describe()
    result["jit_param_count"] = vad_jit.parameter_count()
    result["jit_parameters"] = jit_parameter_table(vad_jit._model)
    LOG.info("Running dynamic probe...")
    result["dynamic_probe"] = run_dynamic_probe(vad_jit)

    LOG.info("Loading Silero (ONNX backend) for session I/O...")
    try:
        vad_onnx = SileroVAD(onnx=True)
        result["describe"].update({
            k: v for k, v in vad_onnx.describe().items()
            if k in ("onnx_inputs", "onnx_outputs")
        })
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ONNX backend load failed: %r", exc)

    onnx_path = find_silero_onnx_path()
    if onnx_path:
        LOG.info("Parsing ONNX graph at %s", onnx_path)
        try:
            result["onnx_path"] = onnx_path
            result["onnx_graph"] = onnx_graph_summary(onnx_path)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("ONNX graph parse failed: %r", exc)
    else:
        LOG.warning("Could not locate silero .onnx file; skipping graph dump.")

    write_json(result, args.out_json)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(result), encoding="utf-8")
    LOG.info("Wrote %s and %s", args.out_md, args.out_json)


if __name__ == "__main__":
    main()
