"""Conversion of *pristine* upstream HF/TRL configs.

The other compat modules build their sources through fixtures that pin a
handful of fields. That hides a whole class of breakage: a field the user
never touched blocking the conversion outright. These tests construct
configs with nothing set beyond what a CPU test host requires, so an
upstream default that opaque's manifest doesn't classify shows up here.
"""

from __future__ import annotations

import pytest

trl = pytest.importorskip("trl")

from opaque.transformers.trl import DPOConfig, SFTConfig  # noqa: E402

# ``bf16`` resolves to True on a pristine TRL config even with
# ``use_cpu=True``, and opaque's precision check rejects that on a host
# without a bf16 accelerator. That check is unrelated to what these tests
# cover, so keep it out of the way.
_HOST = {"bf16": False, "use_cpu": True}

_FLAVORS = (
    pytest.param(trl.SFTConfig, SFTConfig, id="sft"),
    pytest.param(trl.DPOConfig, DPOConfig, id="dpo"),
)


@pytest.mark.parametrize(("trl_cls", "opaque_cls"), _FLAVORS)
def test_pristine_upstream_config_converts(tmp_path, trl_cls, opaque_cls):
    """A default upstream config converts — no field the user never set blocks it."""
    cfg = opaque_cls.from_trl(
        trl_cls(output_dir=str(tmp_path), **_HOST),
        privacy_noise_multiplier=0.8,
    )
    assert cfg is not None


def _fractional_warmup_config(trl_cls, tmp_path):
    """Build a config with a fractional warmup, or skip where upstream refuses.

    Transformers only started reading ``warmup_steps`` as a fraction in 5.x;
    4.x rejects a non-integer outright, so there is nothing to translate.
    """
    try:
        return trl_cls(output_dir=str(tmp_path), warmup_steps=0.05, **_HOST)
    except ValueError:
        pytest.skip("upstream treats warmup_steps as an integer step count only")


@pytest.mark.parametrize(("trl_cls", "opaque_cls"), _FLAVORS)
def test_fractional_warmup_steps_becomes_warmup_ratio(tmp_path, trl_cls, opaque_cls):
    """A fractional ``warmup_steps`` is a *ratio* upstream; opaque must keep it.

    HF resolves ``warmup_steps`` as ``int(w) if w >= 1 else ceil(total * w)``,
    so 0.05 means "5% of training". Opaque splits that across its integer
    ``warmup_steps`` and fractional ``warmup_ratio`` pair.
    """
    cfg = opaque_cls.from_trl(
        _fractional_warmup_config(trl_cls, tmp_path),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.warmup_ratio == pytest.approx(0.05)
    assert cfg.warmup_steps == 0


@pytest.mark.parametrize(("trl_cls", "opaque_cls"), _FLAVORS)
def test_absolute_warmup_steps_stay_absolute(tmp_path, trl_cls, opaque_cls):
    """``warmup_steps >= 1`` is a step count and must not become a ratio."""
    cfg = opaque_cls.from_trl(
        trl_cls(output_dir=str(tmp_path), warmup_steps=25, **_HOST),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.warmup_steps == 25
    assert cfg.warmup_ratio == 0.0


@pytest.mark.parametrize(("trl_cls", "opaque_cls"), _FLAVORS)
def test_unclassified_field_the_user_set_still_raises(tmp_path, trl_cls, opaque_cls):
    """Tolerating unset fields must not tolerate ones the user configured.

    ``warmup_ratio`` is HF's deprecated warmup alias and no opaque manifest
    bucket claims it. Left alone it must not block the conversion; set
    explicitly it must raise, because silently dropping a knob someone
    deliberately configured could invalidate the privacy accounting.
    """
    if "warmup_ratio" not in trl_cls.__dataclass_fields__:
        pytest.skip("upstream dropped the deprecated warmup_ratio alias")
    with pytest.raises(ValueError, match="not classified"):
        opaque_cls.from_trl(
            trl_cls(output_dir=str(tmp_path), warmup_ratio=0.05, **_HOST),
            privacy_noise_multiplier=0.8,
        )
