FROM node:24.20.0-bookworm-slim AS node

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ARG CODEX_VERSION=0.154.0

# Keep Node on the system PATH so plugin hooks work in non-interactive shells.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install --global "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && node --version \
    && codex --version

# The base image includes Yarn's apt source, which this project does not use.
# Remove it so an expired/missing Yarn signing key cannot block Debian packages.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install --no-cache-dir poetry==1.8.5 \
    && python -m venv /opt/venv \
    && chown -R vscode:vscode /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/opt/poetry/bin:${PATH}" \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
