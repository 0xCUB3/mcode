FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system '.[evalplus,datasets]'

RUN mkdir -p /work /tmp/mcode-cache /tmp/.cache/evalplus /tmp/.cache/huggingface && \
    chmod -R 777 /work /tmp/mcode-cache /tmp/.cache

ENV MCODE_CACHE_DIR=/tmp/mcode-cache

WORKDIR /work
ENTRYPOINT ["mcode"]
CMD ["--help"]

