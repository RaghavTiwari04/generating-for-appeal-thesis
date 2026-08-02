"""Captions decide whether the LoRA learns lettering or memorises a phrase.

The training set is birthday cards, so most of them read "Happy Birthday". If
that phrase is constant across every caption, it becomes part of the style and
the LoRA fights the brief's real headline at generation time. Naming each
card's own words makes the lettering a variable the caption controls.
"""

from __future__ import annotations

import pytest

from generation.image.loras.train_lora import _card_text, _training_caption


class TestCardText:
    def test_keeps_the_headline_and_drops_the_body(self):
        raw = "Happy 30th Birthday " + " ".join(f"word{i}" for i in range(40))
        out = _card_text(raw)
        assert out.startswith("Happy 30th Birthday")
        assert len(out.split()) <= 8

    def test_strips_quotes_that_would_break_the_caption(self):
        """The caption wraps this in its own quotes."""
        assert '"' not in _card_text('Happy "Big" Birthday')

    def test_collapses_newlines_from_multi_line_ocr(self):
        assert _card_text("Happy\nBirthday\r\nMum") == "Happy Birthday Mum"

    @pytest.mark.parametrize("raw", [None, "", "   ", 42, float("nan")])
    def test_missing_or_non_text_is_empty(self, raw):
        """OCR returns nothing for a wordless card, and NULL reaches here as a float."""
        assert _card_text(raw) == ""


class TestTrainingCaption:
    def test_names_the_cards_own_words(self):
        from generation.image.headline_text import LETTERING_STYLE

        caption = _training_caption("a watercolour cake", "Happy Birthday", "birthday general")
        assert f'greeting "Happy Birthday" in {LETTERING_STYLE}' in caption
        assert caption.startswith("TOK a watercolour cake")

    def test_omits_the_greeting_clause_when_there_is_no_text(self):
        """Erased-text runs must not claim words the image no longer shows."""
        from generation.image.headline_text import LETTERING_STYLE

        caption = _training_caption("a watercolour cake", "", "birthday general")
        assert LETTERING_STYLE not in caption
        assert caption == "TOK a watercolour cake, a greeting card for birthday general"

    def test_no_blip_caption_yields_nothing_so_the_caller_falls_back(self):
        assert _training_caption("", "Happy Birthday", "birthday general") == ""

    def test_phrasing_matches_the_inference_prompt(self):
        """Training and generation must describe lettering the same way.

        The LoRA is conditioned on the words describing its training images, so
        wording used at generation but never during training asks for something
        it was never shown. Both sides read LETTERING_STYLE; this fails if
        either grows its own copy.
        """
        from generation.image.headline_text import LETTERING_STYLE, augment_prompt

        assert LETTERING_STYLE in augment_prompt("a cake", "Happy Birthday")
        assert LETTERING_STYLE in _training_caption("a cake", "Happy Birthday", "birthday general")
