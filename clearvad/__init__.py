"""ClearVAD — compact, CPU-deployable SSM-based VAD distilled from Silero VAD v5."""

__version__ = "0.0.1.dev0"

# Canonical audio / chunking constants. These are FROZEN for the whole project so
# that every phase (dissection, training, export, benchmark, serving) agrees.
SAMPLE_RATE = 16_000          # Hz
CHUNK_SAMPLES = 512           # samples per inference chunk (Silero v5 16k requirement)
CHUNK_MS = CHUNK_SAMPLES / SAMPLE_RATE * 1000.0   # = 32.0 ms
CONTEXT_SAMPLES = 64          # left-context samples Silero v5 prepends internally (= 4 ms)
TOTAL_INPUT_SAMPLES = CHUNK_SAMPLES + CONTEXT_SAMPLES  # 576

__all__ = [
    "__version__",
    "SAMPLE_RATE",
    "CHUNK_SAMPLES",
    "CHUNK_MS",
    "CONTEXT_SAMPLES",
    "TOTAL_INPUT_SAMPLES",
]
