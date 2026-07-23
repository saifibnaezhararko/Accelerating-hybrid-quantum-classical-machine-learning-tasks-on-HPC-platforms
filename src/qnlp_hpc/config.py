"""Experiment configuration — YAML files in ``configs/`` instead of edited constants.

Why this exists: ``qnlp_hpc.mc1_spider.config`` holds the hyperparameters as
module-level constants, so varying a seed or an ansatz dimension means editing
tracked source. Benchmarking needs ~30 repetitions per dataset and an ansatz
comparison, which is not something to do by hand.

This module loads a validated config and *applies* it onto that constants module.
The contributed pipeline is left byte-for-byte untouched; ``configs/mc1_spider.yaml``
reproduces its current defaults exactly, so a default run is unchanged.

Ordering matters: ``mc1_spider.data``/``diagrams``/``model``/``experiment`` bind the
constants with ``from ... import NAME`` at import time, so :func:`apply_to_mc1_spider`
must run *before* they are imported. It refuses to run otherwise instead of
silently having no effect.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from qnlp_hpc.paths import resolve

# Modules that copy constants out of mc1_spider.config at import time.
_BINDING_MODULES = (
    "qnlp_hpc.mc1_spider.data",
    "qnlp_hpc.mc1_spider.diagrams",
    "qnlp_hpc.mc1_spider.model",
    "qnlp_hpc.mc1_spider.reporting",
    "qnlp_hpc.mc1_spider.experiment",
)


class ConfigError(ValueError):
    """Raised for a malformed or invalid experiment config."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Flattened view of an experiment YAML file."""

    name: str = "mc1_spider_baseline"

    # data
    data_path: str = "data/processed/MC1.txt"
    test_size: float = 0.20
    validation_size_within_train: float = 0.20

    # training
    seed: int = 2
    batch_size: int = 8
    epochs: int = 150
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 20

    # ansatz
    noun_dim: int = 4
    sentence_dim: int = 4
    max_order: int = 2

    # classifier
    classifier_hidden_dim: int = 16
    classifier_dropout: float = 0.0

    # output
    output_dir: str = "outputs/mc1_spider"

    def validate(self) -> None:
        positive_integers = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "noun_dim": self.noun_dim,
            "sentence_dim": self.sentence_dim,
            "max_order": self.max_order,
            "classifier_hidden_dim": self.classifier_hidden_dim,
        }
        for key, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(f"{key} must be a positive integer, got {value!r}.")

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ConfigError(f"seed must be an integer, got {self.seed!r}.")

        fractions = {
            "test_size": self.test_size,
            "validation_size_within_train": self.validation_size_within_train,
        }
        for key, value in fractions.items():
            if not 0.0 < float(value) < 1.0:
                raise ConfigError(f"{key} must be in (0, 1), got {value!r}.")

        if not 0.0 <= float(self.classifier_dropout) < 1.0:
            raise ConfigError(
                f"classifier_dropout must be in [0, 1), got {self.classifier_dropout!r}."
            )

        for key in ("learning_rate", "weight_decay"):
            value = float(getattr(self, key))
            if value < 0.0:
                raise ConfigError(f"{key} must be non-negative, got {value!r}.")
        if float(self.learning_rate) == 0.0:
            raise ConfigError("learning_rate must be greater than 0.")

    @property
    def resolved_data_path(self) -> Path:
        return resolve(self.data_path)

    @property
    def resolved_output_dir(self) -> Path:
        return resolve(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# YAML section -> (yaml key -> dataclass field).
_SECTIONS: dict[str, dict[str, str]] = {
    "data": {
        "path": "data_path",
        "test_size": "test_size",
        "validation_size_within_train": "validation_size_within_train",
    },
    "training": {
        "seed": "seed",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "early_stopping_patience": "early_stopping_patience",
    },
    "ansatz": {
        "noun_dim": "noun_dim",
        "sentence_dim": "sentence_dim",
        "max_order": "max_order",
    },
    "classifier": {
        "hidden_dim": "classifier_hidden_dim",
        "dropout": "classifier_dropout",
    },
    "output": {
        "dir": "output_dir",
    },
}

_FIELD_NAMES = {f.name for f in fields(ExperimentConfig)}


def config_from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
    """Build a config from a parsed YAML mapping, rejecting unknown keys."""
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}.")

    values: dict[str, Any] = {}

    if "name" in raw:
        values["name"] = str(raw["name"])

    unknown_sections = set(raw) - set(_SECTIONS) - {"name"}
    if unknown_sections:
        raise ConfigError(
            f"Unknown config section(s): {sorted(unknown_sections)}. "
            f"Valid sections: {sorted(_SECTIONS)}"
        )

    for section, key_map in _SECTIONS.items():
        block = raw.get(section, {})
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ConfigError(f"Section '{section}' must be a mapping.")
        unknown_keys = set(block) - set(key_map)
        if unknown_keys:
            raise ConfigError(
                f"Unknown key(s) in section '{section}': {sorted(unknown_keys)}. "
                f"Valid keys: {sorted(key_map)}"
            )
        for yaml_key, field_name in key_map.items():
            if yaml_key in block:
                values[field_name] = block[yaml_key]

    config = ExperimentConfig(**values)
    config.validate()
    return config


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    config_path = resolve(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        return config_from_mapping(raw)
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc


def apply_overrides(
    config: ExperimentConfig,
    overrides: dict[str, Any],
) -> ExperimentConfig:
    """Return a copy of ``config`` with ``field=value`` overrides applied.

    Used for CLI ``--set seed=7`` style sweeps, so a 30-repetition benchmark does
    not need 30 YAML files.
    """
    unknown = set(overrides) - _FIELD_NAMES
    if unknown:
        raise ConfigError(
            f"Unknown override field(s): {sorted(unknown)}. "
            f"Valid fields: {sorted(_FIELD_NAMES)}"
        )

    coerced = {**config.to_dict()}
    for key, value in overrides.items():
        current = getattr(config, key)
        coerced[key] = _coerce(value, type(current), key)

    updated = ExperimentConfig(**coerced)
    updated.validate()
    return updated


def _coerce(value: Any, target: type, key: str) -> Any:
    """Coerce a string CLI value to the field's existing type."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        if target is bool:
            return text.lower() in {"1", "true", "yes", "on"}
        if target is int:
            return int(text)
        if target is float:
            return float(text)
    except ValueError as exc:
        raise ConfigError(f"Cannot parse override {key}={value!r} as {target.__name__}.") from exc
    return text


def apply_to_mc1_spider(config: ExperimentConfig) -> None:
    """Push ``config`` onto ``qnlp_hpc.mc1_spider.config`` before the pipeline loads.

    Paths are made absolute against the repo root, so a run is independent of the
    working directory it was launched from.
    """
    already_imported = [name for name in _BINDING_MODULES if name in sys.modules]
    if already_imported:
        raise ConfigError(
            "apply_to_mc1_spider() must run before the pipeline modules are "
            f"imported; already imported: {already_imported}. They copy the "
            "constants at import time, so overrides applied now would be ignored."
        )

    from qnlp_hpc.mc1_spider import config as mc1_config

    output_dir = config.resolved_output_dir
    constants: dict[str, Any] = {
        "SEED": config.seed,
        "TEST_SIZE": config.test_size,
        "VALIDATION_SIZE_WITHIN_TRAIN": config.validation_size_within_train,
        "BATCH_SIZE": config.batch_size,
        "EPOCHS": config.epochs,
        "LEARNING_RATE": config.learning_rate,
        "WEIGHT_DECAY": config.weight_decay,
        "EARLY_STOPPING_PATIENCE": config.early_stopping_patience,
        "NOUN_DIM": config.noun_dim,
        "SENTENCE_DIM": config.sentence_dim,
        "MAX_ORDER": config.max_order,
        "CLASSIFIER_HIDDEN_DIM": config.classifier_hidden_dim,
        "CLASSIFIER_DROPOUT": config.classifier_dropout,
        "DATA_PATH": config.resolved_data_path,
        "OUTPUT_DIR": output_dir,
        "LOG_DIR": output_dir / "training_logs",
    }

    missing = [name for name in constants if not hasattr(mc1_config, name)]
    if missing:
        raise ConfigError(
            "qnlp_hpc.mc1_spider.config no longer defines "
            f"{missing}; the config mapping needs updating."
        )

    for name, value in constants.items():
        setattr(mc1_config, name, value)
