import torch

from ml.model import MODEL_CONFIGS, OptimalFNDModel, count_params


def test_forward_pass(tiny_model):
    ids = torch.zeros(2, 10, dtype=torch.long)
    mask = torch.ones(2, 10, dtype=torch.long)
    out = tiny_model(ids, mask)
    assert out.shape == (2, 2)


def test_output_finite(tiny_model):
    ids = torch.randint(0, 100, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.long)
    out = tiny_model(ids, mask)
    assert torch.isfinite(out).all()


def test_model_configs():
    for name, cfg in MODEL_CONFIGS.items():
        m = OptimalFNDModel(**cfg)
        assert count_params(m) > 0, f"Model {name} has no parameters"


def test_mean_pooling():
    model = OptimalFNDModel(
        vocab_size=100,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_len=10,
        use_cls_token=False,
    )
    ids = torch.zeros(1, 10, dtype=torch.long)
    mask = torch.ones(1, 10, dtype=torch.long)
    out = model(ids, mask)
    assert out.shape == (1, 2)
