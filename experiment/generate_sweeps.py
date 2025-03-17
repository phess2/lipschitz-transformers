import numpy as np
import json
from pathlib import Path

optimizer_pre_post_lr_wd = [
    ("adam", False, False, np.logspace(-3, -1, 10), np.logspace(-3, -1, 3)),
    ("muon", False, True, np.logspace(-2, 0, 10), np.logspace(-3, -1, 3)),
]
d_embeds = [128]
num_heads = [4]
project = False

blocks = 4
seq_len = 256
batch_size = 64

steps = 2001
beta1 = 0.95
beta2 = 0.99
pre_dualize = False
post_dualize = True

seeds = [0]
output_dir = "results"

# Create all combinations
combinations = []
for optimizer, pre_dualize, post_dualize, lrs, wds in optimizer_pre_post_lr_wd:
    for lr in lrs:
        for wd in wds:
            for d_embed in d_embeds:
                for nheads in num_heads:
                    for seed in seeds:
                        combinations.append({
                            'd_embed': d_embed,
                            'lr': lr,
                            'wd': wd,
                            'blocks': blocks,
                            'seq_len': seq_len,
                            'num_heads': nheads,
                            'optimizer': optimizer,
                            'pre_dualize': pre_dualize,
                            'post_dualize': post_dualize,
                            'beta1': beta1,
                            'beta2': beta2,
                            'batch_size': batch_size,
                            'project': project,
                            'steps': steps,
                            'seed': seed,
                            'output_dir': output_dir,
                        })

# Save combinations to file
path = Path('/data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/sweep_configs')
path.mkdir(exist_ok=True)
with open(path / 'parameter_grid.json', 'w') as f:
    json.dump(combinations, f, indent=2)

print(f"Generated {len(combinations)} combinations")