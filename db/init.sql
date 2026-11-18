-- VectorView corpus schema (Postgres + pgvector).
--   panoramas        one row per Street View panorama kept
--   images           one row per (panorama, heading) view downloaded
--   tiles            one row per grid tile cut from an image
--   candidates_seen  every road point already looked up, so a restarted ingest skips it

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS panoramas (
  pano_id        TEXT PRIMARY KEY,
  region         TEXT NOT NULL,
  lat            DOUBLE PRECISION NOT NULL,
  lon            DOUBLE PRECISION NOT NULL,
  captured_at    DATE,
  road_bearing   REAL,
  key_used       TEXT,
  copyright      TEXT,
  fetched_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS panoramas_region_idx ON panoramas (region);

CREATE TABLE IF NOT EXISTS images (
  image_id     TEXT PRIMARY KEY,
  pano_id      TEXT NOT NULL REFERENCES panoramas(pano_id),
  heading      REAL NOT NULL,
  pitch        REAL DEFAULT 0,
  fov          REAL DEFAULT 90,
  width        INTEGER,
  height       INTEGER,
  cache_path   TEXT,
  region       TEXT NOT NULL,
  embedding    VECTOR(1152),
  ingested_at  TIMESTAMPTZ DEFAULT NOW(),
  encoded_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS images_pano_idx ON images (pano_id);
CREATE INDEX IF NOT EXISTS images_region_idx ON images (region);

CREATE TABLE IF NOT EXISTS tiles (
  tile_id       TEXT PRIMARY KEY,
  image_id      TEXT NOT NULL REFERENCES images(image_id),
  scale         INTEGER NOT NULL,
  overlap       REAL DEFAULT 0,
  x_px          INTEGER NOT NULL,
  y_px          INTEGER NOT NULL,
  w_px          INTEGER NOT NULL,
  h_px          INTEGER NOT NULL,
  embedding     VECTOR(1152),
  encoded_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS tiles_image_idx ON tiles (image_id);

CREATE TABLE IF NOT EXISTS candidates_seen (
  region         TEXT NOT NULL,
  lat            DOUBLE PRECISION NOT NULL,
  lon            DOUBLE PRECISION NOT NULL,
  road_bearing   REAL,
  status         TEXT NOT NULL,
  pano_id        TEXT,
  queried_at     TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (region, lat, lon)
);
