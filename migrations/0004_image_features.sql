-- Image features of arbitrary width, for predictor experiments.
--
-- `clip_embedding` is VECTOR(768) with an HNSW index that dedup searches, so
-- its width cannot change without rebuilding that index and re-running the
-- clustering the labelled pool is built from. A larger backbone (1024) or two
-- backbones concatenated (1536) therefore cannot live there.
--
-- REAL[] takes any width and needs no index: the predictor reads these rows by
-- listing_id and never searches them by similarity. Dedup keeps using
-- clip_embedding untouched.
--
-- NULL means "no variant computed" and readers fall back to clip_embedding.
ALTER TABLE listing_features
    ADD COLUMN IF NOT EXISTS image_features REAL[];

-- Which backbones produced them, so a run whose features came from a different
-- encoder stack is identifiable rather than silently mixed.
ALTER TABLE listing_features
    ADD COLUMN IF NOT EXISTS image_feature_source TEXT;
