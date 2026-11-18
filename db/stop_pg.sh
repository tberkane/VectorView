#!/usr/bin/env bash
# Stop the server started by start_pg.sh (same PG_BIN and PG_DATA_DIR variables).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PG_DATA_DIR:-${HERE}/../data/pgdata}"
if [[ -n "${PG_BIN:-}" ]]; then
  PATH="${PG_BIN}:${PATH}"
fi

if pg_ctl -D "${DATA_DIR}" status >/dev/null 2>&1; then
  pg_ctl -D "${DATA_DIR}" -m fast stop
else
  echo "[stop_pg] not running (data: ${DATA_DIR})"
fi
