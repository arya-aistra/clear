# Phase 0 — Key Findings & Spec Reconciliation

Measured on the GPU server (torch 2.12.1+cu130, ORT CPU EP, single thread).
Teacher: `silero_vad_16k_op15.onnx` (opset **15**), JIT backend `RecursiveScriptModule`.

## A. Architecture reconciliation (spec vs measured)

| Item | Spec assumed | **Measured** | Verdict |
|------|--------------|--------------|---------|
| Chunk size | 512 (32 ms) | 512, output `[B,1]` | ✅ match |
| Left context | 64 (4 ms) | `context [1, 64]` | ✅ match |
| External input length | 576 | **512** fed externally; context prepended *inside* the graph (Pad/Slice nodes 22–29) → 576 internal | ✅ (feed 512, not 576) |
| LSTM `weight_ih` | `[512,128]` | `decoder.rnn.weight_ih [512,128]` | ✅ match |
| LSTM `weight_hh` | `[512,128]` | `[512,128]` | ✅ match |
| State `h_n`/`c_n` | two `[1,1,128]` | one combined `state [2,1,128]` (h,c stacked) | ✅ equivalent |
| Front-end | learned STFT `Conv1d(1,258,k=256,stride=128)` | `Conv1d(1→258, k=256)`, **fixed** (ONNX initializer 258×256=66,048; not in `named_parameters`), magnitude → **129** features | ✅ matches; note it's a *fixed* STFT basis, not learned |
| Encoder | `[129,128,64,64,128]`, depthwise-sep (our lightening) | 4× **plain** `reparam_conv` (RepVGG-style) Conv1d k=3, schedule **129→128→64→64→128**, ReLU each | ✅ schedule matches; Silero uses plain convs (our depthwise-sep is the proposed change) |
| Head | `Conv1d(128,1,k=1)+Sigmoid` | `decoder.decoder.2: Conv1d(128→1,k=1)` + Sigmoid | ✅ match |

**Param counts (measured):**
- 16k model (named params): **243,585** · ONNX initializers incl. STFT: **309,633** · binary **1.29 MB**
- Full JIT (16k + 8k): **462,594**

**ONNX I/O:** `input[B,seq]` (f32), `state[2,B,128]` (f32), `sr[]` (i64) → `output[B,1]`, `stateN[2,B,128]`.
Graph uses 3× `If` (8k/16k + batch branches) — control-flow that complicates quantization.

### ⚠️ Implication for our size target
Silero 16k is **243.5K params / 1.29 MB**, not the spec's "~1.8M." The spec's "<300K params"
target is therefore *parity*, not "smaller." To honestly claim **2–3× smaller** we must hit
**≤ ~130K params / INT8 ONNX ≤ ~0.5 MB**. The compact G-SSM (<100K, replacing the 132K LSTM) +
depthwise-separable encoder (replacing the 111K plain-conv encoder) should reach this.
**Tightened size gate: ClearVAD INT8 ≤ 0.5 MB.**

## B. Failure-mode findings (real speech, harvard.wav; teacher max prob 0.9999)

| Probe | Spec hypothesis | **Measured (raw model)** | Honest verdict |
|-------|-----------------|--------------------------|----------------|
| Endpoint delay | Silero is slow (200–400 ms) | **32 ms (1 frame)** raw release | Raw model is already crisp. The production delay is a **post-processing artifact** (default `min_silence_duration_ms=100` + `speech_pad_ms=30`), not the model. |
| Short-silence | Silero misses 100 ms silences | **Detects ALL** 50/100/150/200/300 ms (min prob 0.037 at 50 ms) | Raw model does **not** miss short silences. Production "misses" come from `min_silence_duration_ms` merging gaps — again **post-processing**. |
| INT8 quant | unstable / unsupported | **Fails to load**: `ConvInteger NOT_IMPLEMENTED` on `decoder/2/Conv_quant` | ✅ Confirmed real failure. Naive dynamic INT8 of Silero does not run on ORT CPU. |
| Threshold sweep | clean FAR/MR curve | FAR flat 0.059, MR 0.70→0.90 | **Probe artifact**: harvard.wav has intra-clip pauses but the probe labels the whole clip as speech → inflated MR. Needs frame-labeled audio (Phase 8). Do not over-interpret. |

### 🔴 The big narrative finding
**Silero's raw frame-level behavior is already fast and crisp.** Two of the headline
"failures" (endpoint delay, short-silence miss) are **post-processing artifacts**, not raw-model
deficiencies. Therefore:
- Beating them is a **post-processing** story (asymmetric hysteresis + configurable
  `min_silence`), which is **portable to Silero too** — not an architecture/distillation win.
- A fair benchmark must compare *Silero+its default pipeline* vs *ClearVAD+low-latency pipeline*,
  and disclose that the post-processor is portable.

### CPU latency finding (from phase0_baseline.json)
- Silero end-to-end: **0.0774 ms/chunk** (p99 0.0813), pure-ORT **0.0538 ms** (p99 0.0569), RTF 0.0024.
- This is ~4× under the spec's assumed 0.3 ms. **"Faster CPU than Silero" is not attainable** —
  our explicit per-timestep recurrent loop will struggle to match a fused 0.054 ms LSTM.
- (This CPU is faster than a t3.small; deployment-tier latency must be re-measured on t3 in Phase 7,
  but the *relative* model-vs-model comparison is on this fixed CPU.)
- Memory RSS 29→574 MB here includes torch + JIT + ONNX + quantizer in one process; the
  inference-only ONNX footprint (Phase 7 serving target) is far smaller.

## C. Which claims survive Phase 0

| Claim | Status after Phase 0 |
|-------|----------------------|
| Model size 2–3× smaller | **Viable** (target ≤0.5 MB INT8) |
| INT8 deployment stable (Silero can't) | **Strongly supported** (reproduced Silero's INT8 load failure) |
| Zero external training data (DFKD) | **Viable** by construction (transfer to real audio must be verified) |
| Endpoint latency 200–400 ms faster | **Only as a post-processing profile** (portable; must be framed honestly) |
| Short-silence detection Silero misses | **Only via post-processing** `min_silence` (raw Silero detects them) |
| CPU latency < Silero | **Not attainable** (Silero already 0.054–0.077 ms) |

→ Three bulletproof, architecture/deployment-grounded claims remain: **size, INT8-deployability,
data-free distillation**. The headline needs three — we have them. Endpoint/short-silence become
honest *post-processing* contributions. The CPU-speed claim should be dropped.
