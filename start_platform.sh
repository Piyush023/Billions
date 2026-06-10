#!/usr/bin/env bash
# Starts the AI Wealth Platform and (optionally) your nodemailer service together.
#
# Usage:
#   ./start_platform.sh                       # platform only
#   MAILER_DIR=~/path/to/mailer ./start_platform.sh   # platform + nodemailer
#
# Set MAILER_DIR to the folder containing your nodemailer service's package.json.
# The mailer is started with `npm start` and its URL should be set in .env as
# EMAIL_SERVICE_URL (e.g. http://localhost:3001/send).

set -euo pipefail
cd "$(dirname "$0")"

PIDS=()
cleanup() {
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

if [[ -n "${MAILER_DIR:-}" ]]; then
  echo "Starting nodemailer service from $MAILER_DIR ..."
  (cd "$MAILER_DIR" && npm start) &
  PIDS+=($!)
  sleep 2
fi

echo "Starting AI Wealth Platform on http://localhost:8000 ..."
uvicorn wealth_platform.server:app --host 0.0.0.0 --port 8000 &
PIDS+=($!)

wait
