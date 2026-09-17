#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/healthcare-assistant
# Docker may create the cache parent as root when mounting the model cache.
# Change ownership of directory roots without traversing stored files.
sudo chown "$(id -u):$(id -g)" /home/vscode/.cache /home/vscode/.cache/huggingface /home/vscode/.codex /opt/venv
# The data volume is initialized without reading its contents.
sudo chown "$(id -u):$(id -g)" data
poetry install --with dev --no-root --no-interaction
