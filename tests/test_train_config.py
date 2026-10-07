"""TrainConfig loading: a typo in the YAML must not silently become a default."""

import logging

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

from brats.train import TrainConfig  # noqa: E402


def test_unknown_yaml_keys_are_reported(tmp_path, caplog):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  epochs: 3\n  lamda_cls: 0.5\n")  # typo

    with caplog.at_level(logging.WARNING, logger="brats.train"):
        cfg = TrainConfig.from_yaml(path)

    assert cfg.epochs == 3 and cfg.lambda_cls == TrainConfig().lambda_cls
    assert "lamda_cls" in caplog.text


def test_shipped_config_has_no_unknown_keys(caplog):
    with caplog.at_level(logging.WARNING, logger="brats.train"):
        TrainConfig.from_yaml("configs/segresnet_base.yaml")
    assert "unknown keys" not in caplog.text
