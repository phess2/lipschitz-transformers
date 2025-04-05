import numpy as np
import json
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

optimizer_pre_post_lr = [
    ("adam", False, False, np.logspace(-4, -2.5, 8)),
    ("muon", False, True,  np.logspace(-2.5, -0.5, 8)), 
]

d_embeds = [128]
project = [False]
manifold = False   # if true, post_dualize must be true and pre_dualize must be false

residual_scales = [1]  # (1 - a/depth) * x + (a/depth) * block(x)
softmax_scales = [1] # these get squared
final_scales = [1] # these are linear
scales_learnable = [False]

wd_base = np.array([0, 0.01, 0.001, 0.0001])
wd_and_wdlr_power = [
    (wd_base, 0),
    (wd_base * 100, 1),
    (wd_base * 10000, 2),
] # 0 means decoupled, 1 means proportional to lr, 2 means proportional to lr^2

num_heads = [4]
seq_len = 256
zero_init = True

steps = 4001
beta1 = 0.9
beta2 = 0.95
schedules = ["linear", "cosine", "none"]   # linear or none

seeds = [0]
data = "cifar"
output_dir = "results"

blocks = 3 if data == "shakespeare" else 3

batch_size = 64 if data == "shakespeare" else 128
assert not (data == "cifar" and zero_init == False)

# Create all combinations
combinations = []
for optimizer, pre, post, lrs in optimizer_pre_post_lr:
    assert not manifold or (post and not pre), "manifold optimization requires post_dualize = True and pre_dualize = False"
    for lr in lrs:
        for softmax_scale in softmax_scales:
            for final_scale in final_scales:
                for residual_scale in residual_scales:
                    for scale_learnable in scales_learnable:
                        for wds, wd_lr_power in wd_and_wdlr_power:
                            for wd in wds:
                                for d_embed in d_embeds:
                                    for nheads in num_heads:
                                        for proj in project:
                                            for schedule in schedules:
                                                for seed in seeds:
                                                    combinations.append({
                                                        'd_embed': d_embed,
                                                        'lr': lr,
                                                        'wd': wd,
                                                        'wd_lr_power': wd_lr_power,
                                                        'blocks': blocks,
                                                        'seq_len': seq_len,
                                                        'num_heads': nheads,
                                                        'softmax_scale': softmax_scale,
                                                        'final_scale': final_scale,
                                                        'residual_scale': residual_scale,
                                                        'scales_learnable': scale_learnable,
                                                        'optimizer': optimizer,
                                                        'pre_dualize': pre,
                                                        'post_dualize': post,
                                                        'beta1': beta1,
                                                        'beta2': beta2,
                                                        'batch_size': batch_size,
                                                        'zero_init': zero_init,
                                                        'project': proj,
                                                        'manifold': manifold,
                                                        'steps': steps,
                                                        'schedule': schedule,
                                                        'data': data,
                                                        'seed': seed,
                                                        'output_dir': output_dir,
                                                    })

# Save combinations to file
root_path = os.getenv('ROOT_PATH')
path = Path(root_path) / 'experiment' / 'sweep_configs'
path.mkdir(exist_ok=True)
with open(path / 'parameter_grid.json', 'w') as f:
    json.dump(combinations, f, indent=2)

print(f"Generated {len(combinations)} combinations")
