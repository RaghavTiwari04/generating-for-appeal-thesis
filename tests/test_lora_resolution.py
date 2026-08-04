"""Which LoRA directory an occasion resolves to.

One LoRA is trained per occasion group, so `birthday/kids` has to find
`loras/birthday`. Getting this wrong fails silently — generation logs "no LoRA"
at debug level and produces base-Flux cards, which is condition A wearing
condition B's label.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from generation.image import diffusion


@pytest.fixture
def lora_root(tmp_path: Path):
    with patch.object(diffusion, "LORA_ROOT", tmp_path):
        yield tmp_path


@pytest.fixture
def runner():
    """A runner without constructing a pipeline; only resolution is under test."""
    return diffusion.DiffusionRunner.__new__(diffusion.DiffusionRunner)


class TestResolveLora:
    def test_group_lora_serves_every_subtype(self, runner, lora_root: Path):
        (lora_root / "birthday").mkdir()
        for subtype in ("birthday/general", "birthday/kids", "birthday/milestone"):
            assert runner.resolve_lora(subtype) == lora_root / "birthday"

    def test_subtype_lora_wins_over_the_group(self, runner, lora_root: Path):
        """Per-subtype training stays supported and takes precedence."""
        (lora_root / "birthday").mkdir()
        (lora_root / "birthday_kids").mkdir()
        assert runner.resolve_lora("birthday/kids") == lora_root / "birthday_kids"
        assert runner.resolve_lora("birthday/general") == lora_root / "birthday"

    def test_no_lora_at_all_resolves_to_none(self, runner, lora_root: Path):
        assert runner.resolve_lora("birthday/kids") is None

    def test_another_group_does_not_match(self, runner, lora_root: Path):
        (lora_root / "birthday").mkdir()
        assert runner.resolve_lora("christmas/general") is None

    @pytest.mark.parametrize("occasion", [None, ""])
    def test_missing_occasion_resolves_to_none(self, runner, lora_root: Path, occasion):
        assert runner.resolve_lora(occasion) is None

    def test_ungrouped_occasion_still_resolves(self, runner, lora_root: Path):
        """Not every occasion has a slash — thank_you, easter, new_baby."""
        (lora_root / "thank_you").mkdir()
        assert runner.resolve_lora("thank_you") == lora_root / "thank_you"


class TestPipelineReuse:
    def test_same_group_does_not_force_a_reload(self, runner, lora_root: Path):
        """Both subtypes resolve to one LoRA, so the pipeline must survive.

        Keyed on the occasion instead of the resolved LoRA, this would rebuild
        the whole Flux pipeline between two cards using identical weights.
        """
        (lora_root / "birthday").mkdir()
        runner._lora_occasion = "birthday"
        with patch.object(runner, "_free_pipeline") as freed:
            runner._ensure_occasion("birthday/kids")
        freed.assert_not_called()
        assert runner._lora_occasion == "birthday"

    def test_a_different_lora_forces_a_reload(self, runner, lora_root: Path):
        """fuse_lora is irreversible, so the pipeline has to be rebuilt."""
        (lora_root / "birthday").mkdir()
        (lora_root / "christmas").mkdir()
        runner._lora_occasion = "christmas"
        with patch.object(runner, "_free_pipeline") as freed:
            runner._ensure_occasion("birthday/kids")
        freed.assert_called_once()
        assert runner._lora_occasion is None
