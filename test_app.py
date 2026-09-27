import os
import torch
import pytest
from ecosort.explainability import load_explainer_model

def test_model_file_exists():
    """Test if the trained EfficientNet model file is present in the correct directory."""
    assert os.path.exists("checkpoints/efficientnet.pth"), "EfficientNet model weight file is missing!"

def test_model_loads_successfully():
    """Test if the PyTorch model successfully loads into memory without architecture mismatches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_explainer_model("checkpoints/efficientnet.pth", device)
    assert model is not None, "Model failed to load."