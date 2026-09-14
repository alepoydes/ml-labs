FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.12.14-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable && chmod -R a+rX /app
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /tmp
USER 65534:65534
CMD ["/app/.venv/bin/python", "-m", "coursekit.worker", "/submission", "0"]
