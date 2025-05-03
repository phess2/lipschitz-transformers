import numpy as np
import json
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

optimizer_pre_post_lr = [
    #("adam", False, False, np.logspace(-3.5, -1.5, 8)),
    #("muon", False, True,  np.logspace(-1.5, 1.5, 8)), 
    ("muon", False, True, np.logspace(-1, 2, 8)), 
]

# just for extending the range a bit
# optimizer_pre_post_lr = [
#     #("adam", False, False, np.logspace(-3.5, -1.5, 8)),
#     ("muon", False, True,  np.logspace(-3, -2, 4)), 
# ]

d_embeds = [256] #[12*16]
project = [
    #{"default": "none"},
    #{"default": "orthogonal"},
    #{"default": "hard_cap"},
    {"default": "soft_cap"},
    #{"default": "soft_cap1"},
    #{"default": "soft_cap2"},
    #{"default": "soft_cap3"},
    #{"default": "pure_svd"},
    #{"default": "orthogonal", "mlp_out": "pure_svd"},
]  # key: default or tracker string; value: none, orthogonal, hard_cap, soft_cap1, soft_cap2, soft_cap3, pure_svd
model_dtypes = ["float32"]   # options: float8_e4m3fn, bfloat16, float32, float64
project_dtypes = ["float32"]  # options: float8_e4m3fn, bfloat16, float32, float64
max_embed_inflation_factors = [2**20]#1, 16, 256, 4096]  # Caps the amount duality can increase each column of the embedding gradient
w_max = [1]#[1, 4, 16]  # only affects soft_cap -- max weight norm to enforce (adaptive weight decay coupling) -- dual_norm=False

residual_scales = [1]  # (1 - a/depth) * x + (a/depth) * block(x)
softmax_scales = [1] # these get squared
final_scales = [1, 4, 16, 64, 256, 1024]#1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
blocks_masses = [5]  # [1, 5, 25]
scales_learnable = [False]
adamw_for_embeds = [True]

wd_base = [0]#, 1/8, 1/16, 1/32]#, 0.03, 0.1]#0, 0.03, 0.1, 0.3]
wd_and_wdlr_power = [  # wdlr_power is DISABLED -- due to linear coupling for soft cap
    #(wd_base, 0),
    (wd_base, 1),
    #(wd_base, 2),
] # 0 means decoupled, 1 means proportional to lr, 2 means proportional to lr^2

seeds = [0]
data = "shakespeare"      # fineweb, shakespeare, cifar
output_dir = "results"
randomize_labels = [0]   # label noise fraction (0 = no noise, 1 = randomize all)

batch_size = 16 if data == "fineweb" else (64 if data == "shakespeare" else 512)
accum_steps = 8 if data == "fineweb" else 1

epochs = 20
epoch_steps = 50000 // batch_size
steps = int(epochs * epoch_steps) if data == "cifar" else 5000
beta1 = 0.9
beta2 = 0.95
schedules = ["linear"]      # linear, cosine, or none

num_blocks = 12 if data == "fineweb" else (3 if data == "shakespeare" else 3)
seq_len = 1024 if data == "fineweb" else 256
num_heads = [12] if data == "fineweb" else [4]
zero_init = True

log_interval = 10 if data == "fineweb" else (50 if data == "shakespeare" else epoch_steps // 4)
val_interval = 100 if data == "fineweb" else (100 if data == "shakespeare" else epoch_steps // 2)
val_iters = 200 if data == "fineweb" else 20

# Create all combinations
combinations = []
for proj in project:  # project must come first so parallel jobs take similar times
    for optimizer, pre, post, lrs in optimizer_pre_post_lr:
        for max_embed_inflation_factor in max_embed_inflation_factors:
            for lr in lrs:
                for model_dtype in model_dtypes:
                    for project_dtype in project_dtypes:
                        for softmax_scale in softmax_scales:
                            for final_scale in final_scales:
                                for residual_scale in residual_scales:
                                    for scale_learnable in scales_learnable:
                                        for blocks_mass in blocks_masses:
                                            for wmax in w_max:
                                                for wds, wd_lr_power in wd_and_wdlr_power:
                                                    for wd in wds:
                                                        for d_embed in d_embeds:
                                                            for nheads in num_heads:
                                                                for adamw_for_embed in adamw_for_embeds:
                                                                    for schedule in schedules:
                                                                        for randomize_label in randomize_labels:
                                                                            for seed in seeds:
                                                                                combinations.append({
                                                                                    'd_embed': d_embed,
                                                                                    'lr': lr,
                                                                                    'wd': wd,
                                                                                    'wd_lr_power': wd_lr_power,
                                                                                    'num_blocks': num_blocks,
                                                                                    'seq_len': seq_len,
                                                                                    'num_heads': nheads,
                                                                                    'softmax_scale': softmax_scale,
                                                                                    'final_scale': final_scale,
                                                                                    'residual_scale': residual_scale,
                                                                                    'scales_learnable': scale_learnable,
                                                                                    'blocks_mass': blocks_mass,
                                                                                    'optimizer': optimizer,
                                                                                    'max_embed_inflation_factor': max_embed_inflation_factor,
                                                                                    'adamw_for_embed': adamw_for_embed,
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
                                                                                    'seed': seed,
                                                                                    'log_interval': log_interval,
                                                                                    'val_interval': val_interval,
                                                                                    'val_iters': val_iters,
                                                                                    'output_dir': output_dir,
                                                                                })

    # Save combinations to file
    root_path = os.getenv('ROOT_PATH')
    path = Path(root_path) / 'experiment' / 'sweep_configs'
    path.mkdir(exist_ok=True)
    with open(path / 'parameter_grid.json', 'w') as f:
        json.dump(combinations, f, indent=2)

    print(f"Generated {len(combinations)} combinations")
