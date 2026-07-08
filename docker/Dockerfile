# ==========================================
# STAGE 1: Builder (Instalação e Compilação)
# ==========================================
FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-install-project

# ==========================================
# STAGE 2: Runtime (Imagem Final Otimizada)
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY . .

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "src/main.py"]