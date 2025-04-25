import numpy as np
import json
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

optimizer_pre_post_lr = [
    #("adam", False, False, np.logspace(-3.5, -1.5, 8)),
    #("muon", False, True,  np.logspace(-1.5, 1.5, 8)), 
    ("muon", False, True,  [32])#np.logspace(-2, 0, 6)), 
]

# just for extending the range a bit
# optimizer_pre_post_lr = [
#     #("adam", False, False, np.logspace(-3.5, -1.5, 8)),
#     ("muon", False, True,  np.logspace(-3, -2, 4)), 
# ]

d_embeds = [512]
project = [
    #{"default": "none"},
    #{"default": "orthogonal"},
    #{"default": "laker_pure_svd"},
    #{"default": "laker_approximate1"},
    #{"default": "laker_approximate2"},
    #{"default": "laker_approximate3"},
    {"default": "laker_approximate4"},
    #{"default": "laker_approximate4_float64"},  # to use this option, need to set jax.config.update("jax_enable_x64", True) in modula/atom.py
    #{"default": "laker_approximate5"},
    #{"default": "orthogonal", "mlp_out": "laker_pure_svd"},
]  # key: default or tracker string; value: none, orthogonal, laker, laker_pure_svd
model_dtypes = ["float32"]   # options: float8_e4m3fn, bfloat16, float32, float64
project_dtypes = ["bfloat16"]  # options: float8_e4m3fn, bfloat16, float32, float64

residual_scales = [1]  # (1 - a/depth) * x + (a/depth) * block(x)
softmax_scales = [1] # these get squared
final_scales = [256]#, 4, 16, 64, 256]#32, 64, 128, 256, 512] #[0.25, 1, 4, 16, 64, 96, 112, 128, 144, 160, 256, 512] # these are linear
scales_learnable = [False]

wd_base = [0]#, 0.03, 0.1]#0, 0.03, 0.1, 0.3]
wd_and_wdlr_power = [
    #(wd_base, 0),
    (wd_base, 1),
    #(wd_base, 2),
] # 0 means decoupled, 1 means proportional to lr, 2 means proportional to lr^2

seeds = [0]
data = "cifar"      # fineweb, shakespeare, cifar
output_dir = "results"
randomize_labels = True

batch_size = 16 if data == "fineweb" else (64 if data == "shakespeare" else 512)
accum_steps = 8 if data == "fineweb" else 1
vocab_size = 50304 if data == "fineweb" else 65

epochs = 50
epoch_steps = 50000 // batch_size
steps = int(epochs * epoch_steps) if data == "cifar" else 10001
beta1 = 0.9
beta2 = 0.95
schedules = ["linear"]      # linear, cosine, or none

num_blocks = 12 if data == "fineweb" else (3 if data == "shakespeare" else 3)
seq_len = 1024 if data == "fineweb" else 256
num_heads = [12] if data == "fineweb" else [4]
zero_init = True

log_interval = 10 if data == "fineweb" else (10 if data == "shakespeare" else epoch_steps // 4)
val_interval = 100 if data == "fineweb" else (100 if data == "shakespeare" else epoch_steps // 2)
val_iters = 200 if data == "fineweb" else 50

# Create all combinations
combinations = []
for proj in project:  # project must come first so parallel jobs take similar times
    for optimizer, pre, post, lrs in optimizer_pre_post_lr:
        for lr in lrs:
            for model_dtype in model_dtypes:
                for project_dtype in project_dtypes:
                    for softmax_scale in softmax_scales:
                        for final_scale in final_scales:
                            for residual_scale in residual_scales:
                                for scale_learnable in scales_learnable:
                                    for wds, wd_lr_power in wd_and_wdlr_power:
                                        for wd in wds:
                                            for d_embed in d_embeds:
                                                for nheads in num_heads:
                                                    for schedule in schedules:
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
                                                                'optimizer': optimizer,
                                                                'pre_dualize': pre,
                                                                'post_dualize': post,
                                                                'beta1': beta1,
                                                                'beta2': beta2,
                                                                'batch_size': batch_size,
                                                                'accum_steps': accum_steps,
                                                                'zero_init': zero_init,
                                                                'project': proj,
                                                                'model_dtype': model_dtype,
                                                                'project_dtype': project_dtype,
                                                                'steps': steps,
                                                                'schedule': schedule,
                                                                'data': data,
                                                                'randomize_labels': randomize_labels,
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
