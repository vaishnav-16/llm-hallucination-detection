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


def test_fp16_checkpoint_trains_with_fp32_weights(tmp_path, monkeypatch):
    # MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli is stored in fp16 and transformers 5 loads the stored dtype.
    # Optimiser updates on fp16 weights blew them up and collapsed the first DeBERTa fine-tune to constant output.
    import torch
    import transformers
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    cfg = DebertaV2Config(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                          num_labels=3, id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    DebertaV2ForSequenceClassification(cfg).half().save_pretrained(tmp_path / "nli")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)

    _, model, _ = ft.load_model_for_training(str(tmp_path / "nli"))
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.config.num_labels == 2
