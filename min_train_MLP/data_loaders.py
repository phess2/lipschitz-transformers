from data.shakespeare import load_shakespeare
from data.cifar10 import load_cifar10
from data.fineweb import load_fineweb

def get_data_loader(config):
    """Create appropriate data loader based on configuration."""
    if config.data == "fineweb":
        data = load_fineweb(config.seq_len, config.batch_size)
    elif config.data == "shakespeare":
        data = load_shakespeare(config.seq_len, config.batch_size)
    elif config.data == "cifar":
        data = load_cifar10(config.batch_size, randomize_labels=config.randomize_labels, 
                          dtype=config.dtype)
    else:
        raise ValueError(f"Unknown dataset: {config.data}")
    return data["train_loader"], data["test_loader"], data["loss"]
