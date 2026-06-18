-- Add calibration dimensions to survey_pairs CHECK constraint.
-- Required for the VLM-calibration study which asks all 5 predictor dimensions.

ALTER TABLE survey_pairs DROP CONSTRAINT IF EXISTS survey_pairs_question_dim_check;
ALTER TABLE survey_pairs ADD CONSTRAINT survey_pairs_question_dim_check
    CHECK (question_dim IN (
        'purchase_intent', 'aesthetic',
        'occasion_fit', 'emotional_resonance', 'distinctiveness', 'saleability'
    ));

-- Also relax the left/right card NULL constraint for calibration pairs
-- where both sides are marketplace listings (no generated cards involved).
-- The original constraint required exactly one of (listing_id, generated_id)
-- per side; calibration only uses listings, so generated_id is always NULL.
ALTER TABLE survey_pairs DROP CONSTRAINT IF EXISTS survey_pairs_check;
ALTER TABLE survey_pairs DROP CONSTRAINT IF EXISTS survey_pairs_check1;

-- Recreate: at least one identifier per side (listing OR generated), not both.
ALTER TABLE survey_pairs ADD CONSTRAINT survey_pairs_left_check
    CHECK ( (left_listing_id IS NOT NULL) OR (left_generated_id IS NOT NULL) );
ALTER TABLE survey_pairs ADD CONSTRAINT survey_pairs_right_check
    CHECK ( (right_listing_id IS NOT NULL) OR (right_generated_id IS NOT NULL) );
