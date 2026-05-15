-- Greeting Cards: initial schema

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ---------------------------------------------------------------------------
-- Raw scraped listings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listings (
    listing_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              TEXT NOT NULL,
    source_listing_id   TEXT NOT NULL,
    url                 TEXT NOT NULL,
    title               TEXT,
    description         TEXT,
    seller_id           TEXT,
    price_minor_units   INTEGER,
    currency            CHAR(3),
    review_count        INTEGER,
    review_avg          NUMERIC(3,2),
    favourite_count     INTEGER,
    is_bestseller       BOOLEAN DEFAULT FALSE,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    listing_created_at  TIMESTAMPTZ,
    raw_metadata        JSONB,
    UNIQUE (source, source_listing_id)
);

CREATE INDEX IF NOT EXISTS idx_listings_source       ON listings (source);
CREATE INDEX IF NOT EXISTS idx_listings_seller       ON listings (seller_id);
CREATE INDEX IF NOT EXISTS idx_listings_last_seen    ON listings (last_seen_at);
CREATE INDEX IF NOT EXISTS idx_listings_bestseller   ON listings (is_bestseller) WHERE is_bestseller = TRUE;

-- ---------------------------------------------------------------------------
-- Time-series engagement snapshots
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listing_snapshots (
    listing_id        UUID NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    snapshot_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    review_count      INTEGER,
    favourite_count   INTEGER,
    price_minor_units INTEGER,
    PRIMARY KEY (listing_id, snapshot_at)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_at ON listing_snapshots (snapshot_at);

-- ---------------------------------------------------------------------------
-- Images
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listing_images (
    image_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id   UUID NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    is_primary   BOOLEAN DEFAULT FALSE,
    width        INTEGER,
    height       INTEGER,
    phash        BIT(64),
    sha256_hex   CHAR(64),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (listing_id, storage_path)
);

CREATE INDEX IF NOT EXISTS idx_images_listing ON listing_images (listing_id);
CREATE INDEX IF NOT EXISTS idx_images_phash   ON listing_images (phash);
CREATE INDEX IF NOT EXISTS idx_images_sha256  ON listing_images (sha256_hex);

-- ---------------------------------------------------------------------------
-- Derived features
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listing_features (
    listing_id          UUID PRIMARY KEY REFERENCES listings(listing_id) ON DELETE CASCADE,
    occasion            TEXT,
    occasion_confidence NUMERIC,
    occasion_multilabel JSONB,
    clip_embedding      VECTOR(768),
    extracted_text      TEXT,
    palette_lab         JSONB,
    image_complexity    NUMERIC,
    duplicate_cluster_id UUID,
    duplicate_cluster_size INTEGER,
    feature_version     TEXT NOT NULL DEFAULT 'v1',
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_features_occasion ON listing_features (occasion);
CREATE INDEX IF NOT EXISTS idx_features_cluster  ON listing_features (duplicate_cluster_id);
-- Approximate-NN index for CLIP embeddings (HNSW)
CREATE INDEX IF NOT EXISTS idx_features_clip_hnsw
    ON listing_features USING hnsw (clip_embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Saleability labels (proxy + survey-derived)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS saleability_labels (
    label_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id   UUID NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    label_source TEXT NOT NULL,
    score        NUMERIC NOT NULL,
    raw          JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (listing_id, label_source)
);

CREATE INDEX IF NOT EXISTS idx_labels_listing ON saleability_labels (listing_id);
CREATE INDEX IF NOT EXISTS idx_labels_source  ON saleability_labels (label_source);

-- ---------------------------------------------------------------------------
-- Survey ratings (Prolific)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS survey_ratings (
    rating_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id      TEXT NOT NULL,
    study_id            TEXT NOT NULL,
    listing_id          UUID REFERENCES listings(listing_id) ON DELETE SET NULL,
    generated_card_id   UUID,
    occasion_shown      TEXT,
    purchase_intent     SMALLINT,
    occasion_fit        SMALLINT,
    aesthetic           SMALLINT,
    emotional_resonance SMALLINT,
    distinctiveness     SMALLINT,
    price_acceptability SMALLINT,
    max_price_gbp       NUMERIC(6,2),
    free_text           TEXT,
    rated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_time_ms    INTEGER,
    attention_check_pass BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_ratings_participant ON survey_ratings (participant_id);
CREATE INDEX IF NOT EXISTS idx_ratings_study       ON survey_ratings (study_id);
CREATE INDEX IF NOT EXISTS idx_ratings_listing     ON survey_ratings (listing_id);
CREATE INDEX IF NOT EXISTS idx_ratings_generated   ON survey_ratings (generated_card_id);

-- ---------------------------------------------------------------------------
-- Generated cards
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS generated_cards (
    card_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_version     TEXT NOT NULL,
    condition_tag        TEXT,
    brief                JSONB NOT NULL,
    cover_path           TEXT,
    inside_message       TEXT,
    headline_text        TEXT,
    predicted_scores     JSONB,
    suggested_price_minor INTEGER,
    suggested_price_currency CHAR(3) DEFAULT 'GBP',
    seed                 BIGINT,
    parent_card_id       UUID REFERENCES generated_cards(card_id),
    generated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gen_condition ON generated_cards (condition_tag);
CREATE INDEX IF NOT EXISTS idx_gen_version   ON generated_cards (pipeline_version);

-- ---------------------------------------------------------------------------
-- Scrape job bookkeeping
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scrape_jobs (
    job_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source       TEXT NOT NULL,
    job_type     TEXT NOT NULL,           -- 'discover', 'fetch', 'snapshot'
    seed         TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    error        TEXT,
    counts       JSONB
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON scrape_jobs (status);
