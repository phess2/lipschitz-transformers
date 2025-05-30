from modula.compound import GPT, MLP
from modula.bond import Flatten

def create_model(config):
    """Factory function to create appropriate model based on configuration."""
    kwargs = config.copy()
    kwargs["project"] = {marker: config.project_fn_map[project] for marker, project in config.project.items()}
    kwargs["dtype"] = config.dtype
    kwargs["project_dtype"] = config.project_dtype
    kwargs['sensitive_to_wmax'] = {marker: project in ["hard_cap", "orthogonal", "spec_normalize"] 
                                  for marker, project in config.project.items()}

    if config.data in ["fineweb", "shakespeare"]:
        return GPT(**kwargs)
    elif config.data == "cifar":
        kwargs["output_dim"] = 10
        kwargs["input_dim"] = 32*32*3
        return MLP(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {config.data}")
