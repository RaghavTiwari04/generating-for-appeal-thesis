-- Which card a visitor to the demo site actually picked.
--
-- The predictor is trained on judge labels and there is no human preference
-- data anywhere in this project to check them against. The site shows four
-- candidates from one brief, withholds the model's ranking until the visitor
-- has committed, and records what they chose. That is the comparison the
-- best-of-N null leaves open: the candidates sit inside a narrow band, which
-- is exactly where the selection problem is hardest.
--
-- Written only when GC_LOG_CHOICES is set. Off by default, pending ethics
-- approval; see app/choices.py.
--
-- Anonymous by construction. There is no IP address, user agent, referrer or
-- token column, and no column for anything the visitor typed: the request is
-- recorded as its enumerated fields only, because the free-text constraints
-- field can carry a real person's name. `session_id` is minted per browser
-- tab and dies with it, so a returning visitor is a new session and two
-- visits cannot be joined.
--
-- Append-only. Rows are events, never updated, so a download that follows a
-- choice is a second row rather than a mutation of the first.
CREATE TABLE IF NOT EXISTS demo_choice_events (
    id                bigserial PRIMARY KEY,

    -- Per tab, client-generated. Not a user identifier.
    session_id        uuid NOT NULL,
    job_id            uuid NOT NULL,

    -- choice | download_front | download_print | regenerate | message_edited
    event_type        text NOT NULL,

    -- Request, as enumerated values. No free text.
    occasion          text,
    tone              text,
    relationship      text,
    n_candidates      int,
    scorer            text,

    -- The full slate as the server ranked it, with the position each card was
    -- actually shown in. Without shown_position a later analysis cannot tell a
    -- preference for the model's favourite from a preference for the top-left
    -- tile.
    candidates        jsonb,

    chosen_display_id text,
    chosen_rank       int,
    shown_position    int,
    agreed_top1       boolean,

    -- Grid render to commit. A very short time is a click, not a judgement.
    time_to_choice_ms int,

    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS demo_choice_events_session_job_idx
    ON demo_choice_events (session_id, job_id);

-- The headline question: how often does the visitor pick what the model
-- ranked first. Ordering by time so a drifting rate is visible rather than
-- averaged away.
CREATE INDEX IF NOT EXISTS demo_choice_events_created_idx
    ON demo_choice_events (created_at);
