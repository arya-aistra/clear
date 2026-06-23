"""ONNX runtime validation: shape, FP32 numeric equivalence, streaming state carry, determinism.

An ``OrtVADRunner`` wraps a session with the deployment streaming contract (carry state +
build 64-context 576-windows). All ClearVAD ONNX variants (FP32/FP16/INT8) keep float32 I/O,
so one runner serves all.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES

WIN = CONTEXT_SAMPLES + CHUNK_SAMPLES  # 576


class OrtVADRunner:
    def __init__(self, onnx_path: str, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        meta = {i.name: i for i in self.sess.get_inputs()}
        st = meta["state"].shape
        self.state_shape = [d if isinstance(d, int) and d > 0 else 1 for d in st]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.state_shape, dtype=np.float32)

    def step(self, window: np.ndarray) -> float:
        feeds = {"chunk": window.reshape(1, WIN).astype(np.float32), "state": self.state}
        outs = self.sess.run(None, feeds)
        prob, new_state = outs[0], outs[1]
        self.state = new_state.astype(np.float32)
        return float(np.asarray(prob).reshape(-1)[0])

    def stream(self, audio: np.ndarray, reset: bool = True) -> np.ndarray:
        if reset:
            self.reset()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        K = len(audio) // CHUNK_SAMPLES
        prev = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        probs = []
        for i in range(K):
            chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            window = np.concatenate([prev, chunk])
            probs.append(self.step(window))
            prev = chunk[-CONTEXT_SAMPLES:]
        return np.asarray(probs, dtype=np.float32)


def validate_numeric(onnx_path: str, torch_model, n: int = 50, atol: float = 1e-3) -> Dict:
    """FP32 ONNX vs PyTorch on random (chunk, state) pairs."""
    import torch

    runner = OrtVADRunner(onnx_path)
    torch_model = torch_model.eval()
    rng = np.random.default_rng(0)
    max_prob_diff, max_state_diff = 0.0, 0.0
    for _ in range(n):
        chunk = rng.standard_normal((1, WIN)).astype(np.float32) * 0.1
        state = rng.standard_normal(runner.state_shape).astype(np.float32) * 0.1
        outs = runner.sess.run(None, {"chunk": chunk, "state": state})
        with torch.no_grad():
            tp, ts = torch_model(torch.from_numpy(chunk), torch.from_numpy(state))
        max_prob_diff = max(max_prob_diff, float(np.abs(outs[0] - tp.numpy()).max()))
        max_state_diff = max(max_state_diff, float(np.abs(outs[1] - ts.numpy()).max()))
    return {"max_prob_diff": max_prob_diff, "max_state_diff": max_state_diff,
            "passed": bool(max_prob_diff < atol and max_state_diff < atol)}


def validate_state_carry(onnx_path: str, torch_model, n_chunks: int = 120,
                         atol: float = 1e-3) -> Dict:
    """Streaming through ONNX (carry state) must match torch streaming."""
    import torch

    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(n_chunks * CHUNK_SAMPLES).astype(np.float32) * 0.1)
    runner = OrtVADRunner(onnx_path)
    onnx_probs = runner.stream(audio, reset=True)

    torch_model = torch_model.eval()
    prev = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    state = torch_model.reset_state(1)
    tprobs = []
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            window = np.concatenate([prev, chunk])[None, :]
            p, state = torch_model(torch.from_numpy(window), state)
            tprobs.append(float(p.reshape(-1)[0]))
            prev = chunk[-CONTEXT_SAMPLES:]
    tprobs = np.asarray(tprobs, dtype=np.float32)
    n = min(len(onnx_probs), len(tprobs))
    max_diff = float(np.abs(onnx_probs[:n] - tprobs[:n]).max())
    return {"max_prob_diff": max_diff, "passed": bool(max_diff < atol)}


def validate_determinism(onnx_path: str, n_chunks: int = 60) -> Dict:
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(n_chunks * CHUNK_SAMPLES).astype(np.float32) * 0.1
    r1, r2 = OrtVADRunner(onnx_path), OrtVADRunner(onnx_path)
    p1, p2 = r1.stream(audio), r2.stream(audio)
    return {"passed": bool(np.array_equal(p1, p2))}


def validate_all(onnx_path: str, torch_model, numeric: bool = True,
                 atol: float = 1e-3) -> Dict:
    """Run the full validation suite. Set numeric=False for INT8 (expected to differ)."""
    out: Dict = {"determinism": validate_determinism(onnx_path)}
    if numeric:
        out["numeric_equivalence"] = validate_numeric(onnx_path, torch_model, atol=atol)
    out["state_carry"] = validate_state_carry(onnx_path, torch_model,
                                              atol=atol if numeric else 1.0)
    out["all_passed"] = all(v.get("passed", True) for v in out.values())
    return out
