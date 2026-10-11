"""finetune_from_config: step-capped runs stay out of models/, finished seeds are skipped on rerun."""

import json

from src.train import finetune as ft


def _config(tmp_path):
    cfg = tmp_path / "ft.yaml"
    cfg.write_text("name: m\nseeds: [42, 43]\n", encoding="utf-8")
    return cfg


def _fake_finetune(calls):
    def run(cfg, seed, device, max_steps, models_dir):
        calls.append((seed, max_steps, models_dir))
    return run


def test_step_capped_runs_go_to_scratch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ft, "finetune", _fake_finetune(calls))
    ft.finetune_from_config(str(_config(tmp_path)), seeds=[42], max_steps=20, models_dir=tmp_path / "models")
    assert calls == [(42, 20, ft.SCRATCH_MODELS_DIR)]


def test_finished_seeds_are_skipped_unless_forced(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "m" / "seed42").mkdir(parents=True)
    (models / "m" / "seed42" / "fhc_meta.json").write_text(json.dumps({"max_steps": None}), encoding="utf-8")
    # A smoke checkpoint left at the real path does not count as trained.
    (models / "m" / "seed43").mkdir(parents=True)
    (models / "m" / "seed43" / "fhc_meta.json").write_text(json.dumps({"max_steps": 20}), encoding="utf-8")
    cfg = _config(tmp_path)

    calls = []
    monkeypatch.setattr(ft, "finetune", _fake_finetune(calls))
    ft.finetune_from_config(str(cfg), models_dir=models)
    assert [c[0] for c in calls] == [43]

    calls.clear()
    ft.finetune_from_config(str(cfg), models_dir=models, force=True)
    assert [c[0] for c in calls] == [42, 43]
