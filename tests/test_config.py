"""Tests for the experiment config layer (Week 2, Software Lead)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from qnlp_hpc.config import (
    _BINDING_MODULES,
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    apply_to_mc1_spider,
    config_from_mapping,
    load_config,
)
from qnlp_hpc.paths import CONFIG_DIR

# YAML field -> constant name in qnlp_hpc.mc1_spider.config.
CONFIG_TO_CONSTANT = {
    "seed": "SEED",
    "test_size": "TEST_SIZE",
    "validation_size_within_train": "VALIDATION_SIZE_WITHIN_TRAIN",
    "batch_size": "BATCH_SIZE",
    "epochs": "EPOCHS",
    "learning_rate": "LEARNING_RATE",
    "weight_decay": "WEIGHT_DECAY",
    "early_stopping_patience": "EARLY_STOPPING_PATIENCE",
    "noun_dim": "NOUN_DIM",
    "sentence_dim": "SENTENCE_DIM",
    "max_order": "MAX_ORDER",
    "classifier_hidden_dim": "CLASSIFIER_HIDDEN_DIM",
    "classifier_dropout": "CLASSIFIER_DROPOUT",
}


def test_baseline_config_reproduces_the_contributed_defaults() -> None:
    """configs/mc1_spider.yaml must stay in sync with mc1_spider/config.py.

    If someone changes a constant there without updating the YAML, a "baseline"
    run would silently stop being the baseline. This test is the tripwire.
    """
    from qnlp_hpc.mc1_spider import config as mc1_config

    config = load_config(CONFIG_DIR / "mc1_spider.yaml")

    for field_name, constant in CONFIG_TO_CONSTANT.items():
        assert getattr(config, field_name) == pytest.approx(
            getattr(mc1_config, constant)
        ), f"{field_name} differs from {constant}"

    assert Path(config.data_path) == mc1_config.DATA_PATH
    assert Path(config.output_dir) == mc1_config.OUTPUT_DIR


def test_smoke_config_loads() -> None:
    config = load_config(CONFIG_DIR / "mc1_spider_smoke.yaml")
    assert config.name == "mc1_spider_smoke"
    assert config.epochs == 3


def test_relative_paths_resolve_against_the_repo_root() -> None:
    config = load_config(CONFIG_DIR / "mc1_spider.yaml")
    assert config.resolved_data_path.is_absolute()
    assert config.resolved_data_path.is_file()
    assert config.resolved_output_dir.is_absolute()


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown config section"):
        config_from_mapping({"trainingg": {"seed": 1}})


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_mapping({"training": {"lr": 0.1}})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "epochs", 0),
        ("training", "batch_size", -1),
        ("training", "learning_rate", 0),
        ("training", "weight_decay", -0.5),
        ("data", "test_size", 1.0),
        ("data", "validation_size_within_train", 0.0),
        ("ansatz", "sentence_dim", 0),
        ("classifier", "dropout", 1.0),
    ],
)
def test_invalid_values_are_rejected(section: str, key: str, value: object) -> None:
    with pytest.raises(ConfigError):
        config_from_mapping({section: {key: value}})


def test_bool_is_not_accepted_as_an_integer() -> None:
    # bool is a subclass of int; epochs=True would otherwise mean one epoch.
    with pytest.raises(ConfigError, match="positive integer"):
        config_from_mapping({"training": {"epochs": True}})


def test_missing_config_file_is_reported() -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(CONFIG_DIR / "does_not_exist.yaml")


def test_overrides_coerce_command_line_strings() -> None:
    config = ExperimentConfig()
    updated = apply_overrides(config, {"seed": "7", "learning_rate": "0.01"})

    assert updated.seed == 7
    assert isinstance(updated.seed, int)
    assert updated.learning_rate == pytest.approx(0.01)
    # The original is frozen and untouched.
    assert config.seed == 2


def test_overrides_reject_unknown_fields() -> None:
    with pytest.raises(ConfigError, match="Unknown override field"):
        apply_overrides(ExperimentConfig(), {"lr": "0.01"})


def test_overrides_are_validated() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        apply_overrides(ExperimentConfig(), {"epochs": "0"})


def test_overrides_report_unparseable_values() -> None:
    with pytest.raises(ConfigError, match="Cannot parse override"):
        apply_overrides(ExperimentConfig(), {"seed": "many"})


def test_apply_refuses_once_the_pipeline_is_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constants are copied with `from ... import NAME` at import time."""
    monkeypatch.setitem(sys.modules, "qnlp_hpc.mc1_spider.experiment", object())

    with pytest.raises(ConfigError, match="must run before"):
        apply_to_mc1_spider(ExperimentConfig())


def test_apply_sets_every_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    from qnlp_hpc.mc1_spider import config as mc1_config

    # Other test modules import the pipeline at collection time, so clear the
    # guard's view of sys.modules for the duration of this test.
    for name in _BINDING_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # Restore every constant afterwards — this mutates a real module.
    for constant in [*CONFIG_TO_CONSTANT.values(), "DATA_PATH", "OUTPUT_DIR", "LOG_DIR"]:
        monkeypatch.setattr(mc1_config, constant, getattr(mc1_config, constant))

    config = apply_overrides(
        load_config(CONFIG_DIR / "mc1_spider.yaml"),
        {"seed": "11", "epochs": "4", "sentence_dim": "8"},
    )
    apply_to_mc1_spider(config)

    assert mc1_config.SEED == 11
    assert mc1_config.EPOCHS == 4
    assert mc1_config.SENTENCE_DIM == 8
    assert mc1_config.DATA_PATH.is_absolute()
    assert mc1_config.OUTPUT_DIR.is_absolute()
    assert mc1_config.LOG_DIR == mc1_config.OUTPUT_DIR / "training_logs"
