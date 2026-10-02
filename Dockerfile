# syntax=docker/dockerfile:1.7
FROM python:3.12.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN python -m pip install --upgrade "pip==24.3.1" \
    && python -m pip wheel --wheel-dir=/wheels ".[pytorch]"

FROM python:3.12.12-slim-bookworm AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 gliner \
    && useradd --system --uid 10001 --gid gliner --create-home gliner
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels "gliner-runner[pytorch]==0.1.0" \
    && rm -rf /wheels

USER 10001:10001
WORKDIR /home/gliner
EXPOSE 8080
VOLUME ["/home/gliner/.cache/gliner-runner", "/home/gliner/.config/gliner-runner"]
ENTRYPOINT ["gliner-runner"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
