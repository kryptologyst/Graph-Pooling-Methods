"""Utility functions for graph pooling methods project."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get the model size in megabytes.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def save_config(config: DictConfig, path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: OmegaConf configuration object.
        path: Path to save the configuration.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(config.pretty())


def load_config(path: str) -> DictConfig:
    """Load configuration from file.
    
    Args:
        path: Path to the configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override with.
        
    Returns:
        Merged configuration.
    """
    from omegaconf import OmegaConf
    return OmegaConf.merge(base_config, override_config)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten.
        parent_key: Parent key for recursion.
        sep: Separator for keys.
        
    Returns:
        Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten a dictionary.
    
    Args:
        d: Flattened dictionary.
        sep: Separator for keys.
        
    Returns:
        Nested dictionary.
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement.
            min_delta: Minimum change to qualify as an improvement.
            mode: "min" for minimizing, "max" for maximizing.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to potentially save weights from.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save weights from.
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
