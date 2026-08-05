-- Drop the tables for the crowdsourced rating study.
--
-- The study was replaced by an automated vision-language judge, whose output
-- lands in `listing_labels` instead. Neither table was ever written to:
-- `survey_ratings` (0001) held Likert responses, `survey_pairs` (0002, altered
-- by 0003) held the 2AFC comparisons that superseded them.
--
-- Migrations are forward-only here, so 0002 and 0003 stay in place as history
-- and this file undoes them. Editing them instead would leave an existing
-- database holding tables that a freshly built one never creates.

DROP TABLE IF EXISTS survey_pairs;
DROP TABLE IF EXISTS survey_ratings;
