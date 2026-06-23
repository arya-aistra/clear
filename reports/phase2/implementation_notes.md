# Phase 2 — Implementation Notes (full ClearVAD model)

## Goal
Assemble frontend + depthwise-separable encoder + G-SSM + sigmoid head into a streaming
`[B,576] → (prob[B,1], state[B,256,16])` model.

## Modules
| File | Component | Shape | Params (est.) |
|------|-----------|-------|---------------|
| `frontend.py` | `STFTFrontend`: Conv1d(1→258,k=256,s=128) → magnitude | [B,576]→[B,129,3] | 66,048 |
| `encoder.py` | `ConvEncoder`: 4× depthwise-sep (129→128→64→64→128), 'same' pad | [B,129,3]→[B,128,3] | ~38,900 |
| `gssm.py` | `GSSM` (Phase 1), 1 block | [B,3,128]+state → [B,3,128], state | 115,200 |
| `head.py` | `VADHead`: Conv1d(128→1,k=1)+Sigmoid+mean | [B,3,128]→[B,1] | 129 |
| `clearvad_model.py` | `ClearVADModel` assembly + `ClearVADConfig` | [B,576]→([B,1],state) | **~220,277** |

**Total ≈ 220K params < 300K gate, and < Silero's 309,633 (16k ONNX).** `count_by_module()`
prints the exact breakdown in the test.

## Key design decisions
1. **Only the G-SSM carries cross-chunk state.** Frontend + encoder run *per-chunk* on the
   576-sample window (512 + 64 left context), exactly like Silero. The encoder uses 'same'
   padding so T_enc is constant (=3 for a 576 input). This is precisely why chunked streaming
   == offline is exact: in both modes the encoder features are identical per-chunk, and the
   only temporal carry (G-SSM h) is exact by Phase 1's proven equivalence.
2. **Frontend mirrors Silero's STFT magnitude** (258 = 129 real + 129 imag → 129 mag), but is
   learnable with `load_basis()` for an optional Silero warm-start in Phase 3.
3. **Encoder is depthwise-separable** (Silero uses plain convs) — ~38.9K vs Silero's ~111K. A
   `depthwise_separable=False` plain variant exists for the Silero-parity ablation.
4. **Tensor state, not dict** — `forward` takes/returns the raw G-SSM hidden tensor
   `[B,256,16]` so the model exports to ONNX with state as a clean graph input/output. (The
   spec's `{'h': ...}` dict wrapper, if needed, lives in the Phase 4 streaming layer.)
5. **Head reduction** `mean` (matches Silero ReduceMean) default; `lite` config uses `last`
   for a strictly causal low-latency operating point.

## Configs
`configs/model/clearvad_base.yaml` (faithful), `clearvad_lite.yaml` (tied G-SSM + last-frame
head, smallest), `silero_teacher.yaml` (teacher + measured dissection facts).

## Tests (`tests/test_{frontend,encoder,full_model}.py`)
frontend shape/magnitude/basis/params · encoder shape/T-preservation/dwsep<plain/grad ·
full-model shape/state-carry/zero==None/**streaming==offline <1e-5**/params<300K/lite<base/
**ONNX export+ORT match**/from_config.
