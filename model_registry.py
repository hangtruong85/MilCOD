"""
Model registry - centralized model definitions and creation
Provides easy import and instantiation of all available models
"""

import torch
import torch.nn as nn

# ===================== Model Imports =====================

from models.pvt import (
    PVT,
    PVT_BEM,
    PVT_CBAM,
    PVT_BEM_CBAM,
    PVT_DeepSupervision,
    PVT_DeepSupervision_BEMMulti
)

# ===================== Model Dictionary =====================

MODEL_DICT = {     
    # PVT thêm BEM và CBAM
    "PVT": PVT,
    "PVT_BEM": PVT_BEM,
    "PVT_CBAM": PVT_CBAM,
    "PVT_BEM_CBAM": PVT_BEM_CBAM,
    "PVT_DeepSupervision": PVT_DeepSupervision,
    "PVT_DeepSupervision_BEMMulti": PVT_DeepSupervision_BEMMulti
}


# ===================== Model Creation Function =====================

def create_model(model_name: str, device: str = "cuda") -> nn.Module:
    """
    Create and instantiate a model by name
    
    Args:
        model_name: Name of the model from MODEL_DICT
        device: Device to move model to ("cuda" or "cpu")
    
    Returns:
        model: Instantiated model on specified device
    
    Raises:
        ValueError: If model_name is not found in MODEL_DICT
    
    Example:
        >>> model = create_model("PVT", device="cuda")
        >>> model = create_model("PVT_BEM", device="cuda")
    """
    if model_name not in MODEL_DICT:
        available_models = "\n  - ".join(list(MODEL_DICT.keys()))
        raise ValueError(
            f"Unknown model: '{model_name}'\n"
            f"Available models:\n  - {available_models}"
        )
    
    model_class = MODEL_DICT[model_name]
    model = model_class().to(device)
    
    return model


def get_available_models() -> list:
    """
    Get list of all available model names
    
    Returns:
        list: Available model names
    """
    return list(MODEL_DICT.keys())


def print_available_models():
    """
    Print all available models in a formatted way
    """
    print("="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    
    # Categorize models
    models_by_category = {
        "(with BEM)": [k for k in MODEL_DICT.keys() if k.startswith("UNet3Plus") and "BEM" in k],
        "(with PVT-V2)": [k for k in MODEL_DICT.keys() if "PVT" in k],
    }
    
    for category, models in models_by_category.items():
        if models:
            print(f"\n{category}:")
            for model_name in sorted(models):
                print(f"  - {model_name}")
    
    print("\n" + "="*80)


# ===================== Model Info =====================

def get_model_info(model_name: str) -> dict:
    """
    Get information about a model
    
    Args:
        model_name: Name of the model
    
    Returns:
        dict: Model information (parameters count, architecture, etc.)
    """
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = create_model(model_name, device="cpu")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    has_bem = hasattr(model, 'predict_boundary') and model.predict_boundary
    
    info = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "has_boundary_prediction": has_bem,
    }
    
    return info


def print_model_info(model_name: str):
    """
    Print detailed information about a model
    
    Args:
        model_name: Name of the model
    """
    try:
        info = get_model_info(model_name)
        
        print("="*80)
        print(f"Model: {info['model_name']}")
        print("="*80)
        print(f"Total Parameters      : {info['total_parameters']:,}")
        print(f"Trainable Parameters  : {info['trainable_parameters']:,}")
        print(f"Boundary Prediction   : {info['has_boundary_prediction']}")
        print("="*80)
    
    except ValueError as e:
        print(f"Error: {e}")