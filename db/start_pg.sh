#!/usr/bin/env bash
# Start a user-space Postgres server with pgvector, create the database and apply init.sql.
#
# Needs the Postgres binaries (initdb, pg_ctl, psql, createdb) and the pgvector extension, e.g.
#   conda install -c conda-forge postgresql pgvector     or     brew install postgresql@17 pgvector
#
# Environment variables (defaults in brackets):
#   PG_BIN       directory holding pg_ctl            [found on PATH]
#   PG_DATA_DIR  Postgres data directory             [<repo>/data/pgdata]
#   PG_PORT [5432]  PG_USER [vectorview]  PG_PASSWORD [vectorview]  PG_DB [vectorview]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PG_DATA_DIR:-${HERE}/../data/pgdata}"
PORT="${PG_PORT:-5432}"
PG_ROLE="${PG_USER:-vectorview}"
PG_PASS="${PG_PASSWORD:-vectorview}"
PG_DBNAME="${PG_DB:-vectorview}"

if [[ -n "${PG_BIN:-}" ]]; then
  PATH="${PG_BIN}:${PATH}"
fi
if ! command -v pg_ctl >/dev/null 2>&1; then
  echo "[start_pg] pg_ctl not found; install Postgres + pgvector or set PG_BIN" >&2
  exit 1
fi

if [[ ! -s "${DATA_DIR}/PG_VERSION" ]]; then
  echo "[start_pg] initdb -> ${DATA_DIR}"
  mkdir -p "${DATA_DIR}"
  PWFILE="$(mktemp)"
  trap 'rm -f "${PWFILE}"' EXIT
  printf '%s' "${PG_PASS}" > "${PWFILE}"
  initdb -D "${DATA_DIR}" -U "${PG_ROLE}" --pwfile="${PWFILE}" \
    --auth-local=trust --auth-host=md5 --encoding=UTF8 >/dev/null
  cat >> "${DATA_DIR}/postgresql.conf" <<CONF
listen_addresses = '127.0.0.1'
port = ${PORT}
unix_socket_directories = '${DATA_DIR}'
CONF
fi

if pg_ctl -D "${DATA_DIR}" status >/dev/null 2>&1; then
  echo "[start_pg] already running on port ${PORT} (data: ${DATA_DIR})"
else
  echo "[start_pg] starting on port ${PORT}"
  pg_ctl -D "${DATA_DIR}" -l "${DATA_DIR}/server.log" -w start
fi

export PGPASSWORD="${PG_PASS}"
if ! psql -h 127.0.0.1 -p "${PORT}" -U "${PG_ROLE}" -d postgres -tAc \
     "SELECT 1 FROM pg_database WHERE datname='${PG_DBNAME}'" | grep -q 1; then
  echo "[start_pg] creating database ${PG_DBNAME}"
  createdb -h 127.0.0.1 -p "${PORT}" -U "${PG_ROLE}" "${PG_DBNAME}"
fi

echo "[start_pg] applying ${HERE}/init.sql"
psql -h 127.0.0.1 -p "${PORT}" -U "${PG_ROLE}" -d "${PG_DBNAME}" -v ON_ERROR_STOP=1 -q -f "${HERE}/init.sql"
echo "[start_pg] ready: postgresql://${PG_ROLE}:***@127.0.0.1:${PORT}/${PG_DBNAME}"
