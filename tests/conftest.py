from unittest.mock import MagicMock

import pytest
import torch

from ml.model import OptimalFNDModel


@pytest.fixture
def tiny_model():
    """Tiny model (d_model=16) for fast CPU smoke tests."""
    model = OptimalFNDModel(
        vocab_size=100,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        max_len=10,
    )
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tok
