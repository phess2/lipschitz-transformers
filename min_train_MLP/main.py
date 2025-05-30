import json
import argparse
import jax
import numpy as np
import copy
from pathlib import Path

from configs import parse_config_from_json
from data_loaders import get_data_loader
from models import create_model
from optimizers import get_optimizer
from trainer import Trainer
from utils import Logger, save_results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Modula train script")
    parser.add_argument("--job_idx", type=int, default=-1, help="Index of the job")
    parser.add_argument("--sweep_config_path", type=str, help="Path to the sweep config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.sweep_config_path, 'r') as f:
        job_idx = args.job_idx
        config_dict = json.load(f)[job_idx]
        config_dict["job_idx"] = job_idx
        config = parse_config_from_json(config_dict)
    
    # Print configuration
    for key, value in vars(config).items():
        if key not in ["project_fn_map", "dtype", "project_dtype"]:
            print(f"{key}: {value}")
    
    # Set up experiment
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    
    # Initialize components
    train_loader, val_loader, loss_fn = get_data_loader(config)
    model = create_model(config)
    optimizer = get_optimizer(config)
    logger = Logger(config)
    
    # Initialize model and optimizer
    key, subkey = jax.random.split(key)
    params = model.initialize(subkey)
    opt_state = optimizer.init_state(params)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        logger=logger
    )
    
    # Train model
    weights_checkpoints = []
    for epoch in range(1):
        params, opt_state, key = trainer.train_epoch(params, opt_state, key)
        
        # Save checkpoint if needed
        if config.num_checkpoints > 0:
            weights_checkpoints.append(copy.deepcopy(params))
            
        # Check if training is complete
        if trainer.step >= config.steps:
            break

    # Get results and save
    results = logger.get_results()
    save_results(results, weights_checkpoints, model, config)

if __name__ == "__main__":
    main()
