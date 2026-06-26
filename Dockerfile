# Lean CPU serving image for the INT8 ClearVAD-CfC VAD (~0.46 MB model + ORT, no torch).
FROM python:3.11-slim

WORKDIR /app

# System deps for soundfile (libsndfile).
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code (serving only) + the locked INT8 binary.
COPY clearvad/serving/ clearvad/serving/
COPY dist/clearvad_lite.onnx dist/clearvad_lite.onnx

ENV CLEARVAD_MODEL=dist/clearvad_lite.onnx \
    CLEARVAD_THREADS=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "clearvad.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
