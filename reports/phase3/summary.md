# Phase 3 — Summary (Data-Free Knowledge Distillation)

**Status: PHASE 3 PASSED** (server run 2026-06-23).

## Result — a working distilled model
| metric | Stage 1 | Stage 2 | Transfer (harvard.wav, held-out real) |
|--------|---------|---------|----------------------------------------|
| speech_f1 | 0.976 | **0.977** | **0.972** |
| recall | 0.990 | 0.986 | 0.957 |
| precision | 0.962 | 0.968 | 0.988 |
| agreement | 0.962 | 0.964 | 0.960 |

ClearVAD reproduces Silero on real held-out speech at **speech_f1 0.972 / 96% agreement**.

## Acceptance gates
| Gate | Result | Status |
|------|--------|--------|
| Training converges | soft loss 1.81→0.04 (stage1), stable stage2 | ✅ |
| Teacher-student agreement >90% (meaningful) | 0.96 on balanced holdout; speech_f1 0.977 | ✅ |
| Matches Silero on REAL audio (honest Claim 2) | transfer speech_f1 0.972 | ✅ |
| Endpoint latency / short-silence beat | deferred to Phase 6 (post-proc) + Phase 8 | ⏳ |

## How we got here (the pivot that mattered)
- **Synthetic-only DFKD capped at speech_f1≈0.36** (data ceiling: Silero's "speech" signal on
  synthetic is sparse/weak). Kept as the documented "pure DFKD" ablation.
- **Real speech (LibriSpeech dev-clean, teacher-labeled) + synthetic edge cases + class
  balancing (pos_weight)** → speech_f1 0.36 → **0.977**. Label-free (teacher-only); "zero
  external LABELS".
- **Parallel (associative) scan + bf16 AMP** → 1.9 s/step → **0.22 s/step** (~8.6×); loop
  retained for ONNX export. Full 2-stage run ~38 min.

## The bar (Silero on independent eval set)
F1=0.838, FAR=0.239, MR=0.246, endpoint=47.8 ms, onset=108.5 ms; short-silence detection
50ms→0.45, 100ms→0.74, 200ms→0.80, 300ms→0.94. (Labels from construction, not Silero.)

## Next
`compare_models.py` → ClearVAD vs Silero head-to-head (params/size win immediate; accuracy
parity expected). Then Phase 4 (prune) → Phase 5 (INT8 ONNX) banks the size + INT8 wins,
Phase 6 (hysteresis) the endpoint-latency win.
