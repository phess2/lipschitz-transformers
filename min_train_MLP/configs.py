from modula.atom import orthogonalize, hard_cap, soft_cap, pure_svd
from modula.atom import spectral_hammer, spectral_weight_decay, spectral_normalize
import jax.numpy as jnp

class Config:
    """Configuration container with attribute access."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def copy(self):
        """Return a copy of the configuration."""
        return {k: v for k, v in self.__dict__.items()}

# Project function mapping
PROJECT_FUNCTIONS = {
    "none": lambda x, **kwargs: x,
    "orthogonal": orthogonalize,
    "hard_cap": hard_cap,
    "soft_cap": soft_cap,
    "pure_svd": pure_svd,
    "spec_hammer": spectral_hammer,
    "spec_wd": spectral_weight_decay,
    "spec_normalize": spectral_normalize,
}

# Data type mapping
DTYPES = {
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float64": jnp.float64,
}

def parse_config_from_json(config_dict):
    """Convert JSON config dict to Config object with proper type conversions."""
    config = Config(**config_dict)
    
    # Set reference to mappings
    config.project_fn_map = PROJECT_FUNCTIONS
    config.dtype = DTYPES[config.model_dtype]
    config.project_dtype = DTYPES[config.project_dtype]
    
    return config
