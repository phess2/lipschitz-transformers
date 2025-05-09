import numpy as np
import json
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

num_checkpoints = 5   # 0 saves readable results file, 1 saves final weights, n>1 saves evenly spaced checkpoints

optimizer_pre_post_lr = [
    #("adam", False, False, np.logspace(-4, -0.5, 12)),
    ("muon", False, True, np.logspace(-2, 0, 12)), 
]

d_embeds = [256] #[12*16]
project = [
    # {"default": "none"},
    #{"default": "orthogonal"},
    #{"default": "hard_cap"},
    #{"default": "soft_cap"},
    #{"default": "pure_svd"},
    #{"default": "spec_wd"},
    #{"default": "spec_hammer"},
    #{"default": "spec_normalize"},
]  # key: default or tracker string; value: none, orthogonal, hard_cap, soft_cap1, soft_cap2, soft_cap3, pure_svd
model_dtypes = ["float32"]   # options: float8_e4m3fn, bfloat16, float32, float64
project_dtypes = ["float32"]  # options: float8_e4m3fn, bfloat16, float32, float64
max_embed_inflation_factors = [16]#, 64, 256]#1, 16, 256, 4096]  # Caps the amount duality can increase each column of the embedding gradient
use_unembeds = [False]
w_max = [2]#1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4]  # only affects soft_cap -- max weight norm to enforce (adaptive weight decay coupling) -- dual_norm=False

residual_scales = [1]  # (1 - a/num_blocks) * x + (a/num_blocks) * block(x)
softmax_scale = 1
final_scale = 1 #, 4, 16, 64, 256, 1024]#1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
blocks_masses = [32]#, 16, 64]  # [1, 5, 25]
scales_learnable = [False]
layernorm_substitutes = ["none"]  # none, tanh, rmsnorm, layernorm

weight_decay = [0]#, 1/8, 1/16, 1/32]#, 0.03, 0.1]#0, 0.03, 0.1, 0.3]

seeds = [0]
data = "cifar"      # fineweb, shakespeare, cifar
output_dir = "results"
randomize_labels = [0]   # label noise fraction (0 = no noise, 1 = randomize all)

batch_size = 16 if data == "fineweb" else (64 if data == "shakespeare" else 512)
accum_steps = 8 if data == "fineweb" else 1
vocab_size = 50304 if data == "fineweb" else 65

epochs = 50
epoch_steps = 50000 // batch_size
steps = int(epochs * epoch_steps) if data == "cifar" else 2000
beta1s = [0.9]
beta2s = [0.95]
schedules = ["linear"]      # linear, cosine, or none

num_blocks = 12 if data == "fineweb" else (3 if data == "shakespeare" else 3)
seq_len = 1024 if data == "fineweb" else 256
num_heads = 12 if data == "fineweb" else 4
zero_init = True

log_interval = 10 if data == "fineweb" else (100 if data == "shakespeare" else epoch_steps // 4)
val_interval = 100 if data == "fineweb" else (100 if data == "shakespeare" else epoch_steps // 2)
val_iters = 200 if data == "fineweb" else 20

# Create all combinations
combinations = []
for proj in project:  # project must come first so parallel jobs take similar times
    for optimizer, pre, post, lrs in optimizer_pre_post_lr:
        for lr in lrs:
            for max_embed_inflation_factor in max_embed_inflation_factors:
                for use_unembed in use_unembeds:
                    for model_dtype in model_dtypes:
                        for project_dtype in project_dtypes:
                            for beta1 in beta1s:
                                for beta2 in beta2s:
                                    if beta1 >= beta2:
                                        continue
                                    #for softmax_scale in softmax_scales:
                                    #for final_scale in final_scales:
                                    for residual_scale in residual_scales:
                                        for scale_learnable in scales_learnable:
                                            for blocks_mass in blocks_masses:
                                                for layernorm_substitute in layernorm_substitutes:
                                                    for wmax in w_max:
                                                        for wd in weight_decay:
                                                            for d_embed in d_embeds:
                                                                for schedule in schedules:
                                                                    for randomize_label in randomize_labels:
                                                                        for seed in seeds:
                                                                            combinations.append({
                                                                                'd_embed': d_embed,
                                                                                'lr': lr,
                                                                                'wd': wd,
                                                                                'num_blocks': num_blocks,
                                                                                'seq_len': seq_len,
                                                                                'num_heads': num_heads,
                                                                                'softmax_scale': softmax_scale,
                                                                                'final_scale': final_scale,
                                                                                'residual_scale': residual_scale,
                                                                                'scales_learnable': scale_learnable,
                                                                                'blocks_mass': blocks_mass,
                                                                                'layernorm_substitute': layernorm_substitute,
                                                                                'optimizer': optimizer,
                                                                                'max_embed_inflation_factor': max_embed_inflation_factor,
                                                                                'use_unembed': use_unembed,
                                                                                'pre_dualize': pre,
                                                                                'post_dualize': post,
                                                                                'beta1': beta1,
                                                                                'beta2': beta2,
                                                                                'batch_size': batch_size,
                                                                                'accum_steps': accum_steps,
                                                                                'zero_init': zero_init,
                                                                                'project': proj,
                                                                                'w_max': wmax,
                                                                                'model_dtype': model_dtype,
                                                                                'project_dtype': project_dtype,
                                                                                'steps': steps,
                                                                                'schedule': schedule,
                                                                                'data': data,
                                                                                'randomize_labels': randomize_label,
                                                                                'vocab_size': vocab_size,
                                                                                'seed': seed,
                                                                                'log_interval': log_interval,
                                                                                'val_interval': val_interval,
                                                                                'val_iters': val_iters,
                                                                                'num_checkpoints': num_checkpoints,
                                                                                'output_dir': output_dir,
                                                                            })

# Save combinations to file
root_path = os.getenv('ROOT_PATH')
path = Path(root_path) / 'experiment' / 'sweep_configs'
path.mkdir(exist_ok=True)

# append to existing parameter_grid.json file
existing_combinations = []
if os.path.exists(path / 'parameter_grid.json'):
    with open(path / 'parameter_grid.json', 'r') as f:
        existing_combinations = json.load(f)
new_combinations = existing_combinations + combinations
with open(path / 'parameter_grid.json', 'w') as f:
    json.dump(new_combinations, f, indent=2)

print(f"Generated {len(combinations)} combinations")
if len(new_combinations) > len(combinations):
    print(f"\tFor a total of {len(new_combinations)} combinations")
