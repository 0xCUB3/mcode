FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps first (layer cached unless pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p src/mcode && touch src/mcode/__init__.py && \
    uv pip install --system '.[evalplus,datasets]'

# Copy source and reinstall (fast, deps already present)
COPY src ./src
RUN uv pip install --system --no-deps .

RUN mkdir -p /work /tmp/mcode-cache /tmp/.cache/evalplus /tmp/.cache/huggingface && \
    chmod -R 777 /work /tmp/mcode-cache /tmp/.cache

ENV MCODE_CACHE_DIR=/tmp/mcode-cache

WORKDIR /work
ENTRYPOINT ["mcode"]
CMD ["--help"]
