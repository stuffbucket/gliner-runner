# syntax=docker/dockerfile:1.7
FROM node:24.21.0-bookworm-slim AS client-builder

ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH

RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,required=true \
    npm install --global --silent pnpm@9.15.1
WORKDIR /build
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/client/package.json packages/client/tsconfig.json packages/client/index.ts \
    packages/client/index.test.mjs ./packages/client/
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,required=true \
    pnpm install --frozen-lockfile --reporter=silent \
    && pnpm run check \
    && mkdir -p /out \
    && pnpm --dir packages/client pack --pack-destination /out

FROM python:3.12.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INPUT=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN --mount=type=secret,id=pip_config,target=/etc/pip.conf,required=false \
    python -m pip install --quiet --upgrade "pip==24.3.1" \
    && python -m pip wheel --quiet --wheel-dir=/wheels ".[pytorch]"

FROM python:3.12.12-slim-bookworm AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 gliner \
    && useradd --system --uid 10001 --gid gliner --create-home gliner
COPY --from=builder /wheels /wheels
COPY --from=client-builder /out /opt/gliner-runner/packages
RUN python -m pip install --no-index --find-links=/wheels "gliner-runner[pytorch]==0.1.0" \
    && rm -rf /wheels

USER 10001:10001
WORKDIR /home/gliner
EXPOSE 8080
VOLUME ["/home/gliner/.cache/gliner-runner", "/home/gliner/.config/gliner-runner"]
ENTRYPOINT ["gliner-runner"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
