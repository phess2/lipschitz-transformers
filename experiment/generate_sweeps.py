import numpy as np
import json
from pathlib import Path

optimizer_pre_post_lr_wd = [
    #("adam", False, False, np.logspace(-3.25, -2, 8), [0.01]),  # Adam
    #("adam", False, True,  np.logspace(-2.5, -0.5, 8), [0.01]),
    #("adam", True, False,  np.logspace(-3, -1, 8), [0.01]),
    #("adam", True, True,   np.logspace(-2.5, -0.5, 8), [0.01]),
    #("muon", False, False, np.logspace(-2, 0, 8), [0.01]),  # SGD
    #("muon", False, True,  np.logspace(-1, 0.5, 8), [0.01]),  # Muon
    #("muon", True, False,  np.logspace(-0.5, 1.5, 8), [0.01]),
    ("muon", False, True,  np.logspace(-2.5, -0.5, 5), [0.01]),
    #("muon", True, True,   np.logspace(-1, 0.5, 8), [0.01]),
]
#d_embeds = [64, 128, 256, 512]
d_embeds = [128]
num_heads = [4]
project = [False]
manifold = True
softmax_scales = [4**i for i in range(6)]
final_scales = [4**i for i in range(6)]

blocks = 4
seq_len = 256
batch_size = 64

steps = 2001
beta1 = 0.95
beta2 = 0.99

seeds = [0]
output_dir = "results"

# Create all combinations
combinations = []
for optimizer, pre, post, lrs, wds in optimizer_pre_post_lr_wd:
    for lr in lrs:
        for softmax_scale in softmax_scales:
            for final_scale in final_scales:
                for wd in wds:
                    for d_embed in d_embeds:
                        for nheads in num_heads:
                            for proj in project:
                                for seed in seeds:
                                    combinations.append({
                                        'd_embed': d_embed,
                                        'lr': lr,
                                        'wd': wd,
                                        'blocks': blocks,
                                        'seq_len': seq_len,
                                        'num_heads': nheads,
                                        'softmax_scale': softmax_scale,
                                        'final_scale': final_scale,
                                        'optimizer': optimizer,
                                        'pre_dualize': pre,
                                        'post_dualize': post,
                                        'beta1': beta1,
                                        'beta2': beta2,
                                        'batch_size': batch_size,
                                        'project': proj,
                                        'manifold': manifold,
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