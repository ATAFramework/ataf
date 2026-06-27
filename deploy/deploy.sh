#!/usr/bin/env bash
#
# Deploy the ATAF server to 192.168.1.156 as a systemd *user* service on
# port 9123. Idempotent: safe to re-run to ship updates.
#
# Prereqs: SSH access to rajramani@192.168.1.156, and that host reachable
# (it was DOWN at first deploy attempt — confirm with `ping 192.168.1.156`).
#
# Usage:  ./deploy/deploy.sh
#
set -euo pipefail

REMOTE_USER=rajramani
REMOTE_HOST=192.168.1.156
REMOTE_DIR=/home/${REMOTE_USER}/ataf
SERVICE=ataf-server
REMOTE=${REMOTE_USER}@${REMOTE_HOST}

echo ">> [1/6] Sanity-check the host is reachable"
ping -c 2 -t 4 "${REMOTE_HOST}" >/dev/null 2>&1 || {
  echo "ERROR: ${REMOTE_HOST} is not reachable. Is the box powered on?"; exit 1; }

echo ">> [2/6] Sync source to ${REMOTE}:${REMOTE_DIR}"
# Ship the package + project metadata; never ship local venv / data / caches.
rsync -az --delete \
  --exclude '.venv' \
  --exclude 'ataf_data' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.git' \
  --exclude '*.egg-info' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo ">> [3/6] Create venv + install (idempotent)"
ssh "${REMOTE}" "
  set -e
  cd ${REMOTE_DIR}
  test -d .venv || python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -e .
"

echo ">> [4/6] Install the systemd user service"
ssh "${REMOTE}" "mkdir -p ~/.config/systemd/user"
scp ./deploy/${SERVICE}.service "${REMOTE}:~/.config/systemd/user/${SERVICE}.service"

echo ">> [5/6] Enable + (re)start the service"
ssh "${REMOTE}" "
  systemctl --user daemon-reload &&
  systemctl --user enable ${SERVICE} &&
  systemctl --user restart ${SERVICE} &&
  sleep 3 &&
  systemctl --user status ${SERVICE} --no-pager &&
  echo '--- default.target.wants (service MUST appear here) ---' &&
  ls ~/.config/systemd/user/default.target.wants/
"

echo ">> [6/6] Smoke-test the live API"
ssh "${REMOTE}" "curl -s -o /dev/null -w 'GET /openapi.json -> HTTP %{http_code}\n' http://127.0.0.1:9123/openapi.json"

echo ">> Done. Docs: http://${REMOTE_HOST}:9123/docs"
