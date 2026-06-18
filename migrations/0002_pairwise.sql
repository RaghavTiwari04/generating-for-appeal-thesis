-- Pairwise (2AFC) survey comparisons.
--
-- Cheaper alternative to per-card Likert ratings: participant chooses one of
-- two cards in response to a single question (e.g. "Which would you buy?").
-- Bradley-Terry scaling recovers a scalar score per card from the resulting
-- comparison graph. Birthday-only scope for the thesis (see common/occasions.py).

CREATE TABLE IF NOT EXISTS survey_pairs (
    pair_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id     TEXT NOT NULL,
    study_id           TEXT NOT NULL,

    -- The two cards being compared. Each side is either a marketplace listing
    -- OR a generated card; exactly one of (listing_id, generated_card_id) per
    -- side must be non-null. Enforced via CHECK constraints below.
    left_listing_id    UUID REFERENCES listings(listing_id)        ON DELETE SET NULL,
    left_generated_id  UUID REFERENCES generated_cards(card_id)    ON DELETE SET NULL,
    right_listing_id   UUID REFERENCES listings(listing_id)        ON DELETE SET NULL,
    right_generated_id UUID REFERENCES generated_cards(card_id)    ON DELETE SET NULL,

    occasion_shown     TEXT NOT NULL,        -- forced to ACTIVE_OCCASIONS member
    question_dim       TEXT NOT NULL,        -- 'purchase_intent' | 'aesthetic'
    winner_side        CHAR(1) NOT NULL,     -- 'L' | 'R' | 'T' (tie)

    response_time_ms   INTEGER,
    attention_check_pass BOOLEAN,
    rated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Exactly one card per side
    CHECK ( (left_listing_id IS NOT NULL)::int + (left_generated_id IS NOT NULL)::int = 1 ),
    CHECK ( (right_listing_id IS NOT NULL)::int + (right_generated_id IS NOT NULL)::int = 1 ),
    CHECK ( winner_side IN ('L','R','T') ),
    CHECK ( question_dim IN ('purchase_intent','aesthetic') )
);

CREATE INDEX IF NOT EXISTS idx_pairs_study        ON survey_pairs (study_id);
CREATE INDEX IF NOT EXISTS idx_pairs_participant  ON survey_pairs (participant_id);
CREATE INDEX IF NOT EXISTS idx_pairs_occasion     ON survey_pairs (occasion_shown);
CREATE INDEX IF NOT EXISTS idx_pairs_left_lst     ON survey_pairs (left_listing_id);
CREATE INDEX IF NOT EXISTS idx_pairs_right_lst    ON survey_pairs (right_listing_id);
CREATE INDEX IF NOT EXISTS idx_pairs_left_gen     ON survey_pairs (left_generated_id);
CREATE INDEX IF NOT EXISTS idx_pairs_right_gen    ON survey_pairs (right_generated_id);
