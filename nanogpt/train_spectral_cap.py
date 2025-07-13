import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
#torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def orthogonalize(M):
    """Orthogonalize matrices without sending singular values above 1."""
    abc_list = [
        # (3955/1024, -8306/1024, 5008/1024),
        # (3735/1024, -6681/1024, 3463/1024),
        # (3799/1024, -6499/1024, 3211/1024),
        # (4019/1024, -6385/1024, 2906/1024),
        # (2677/1024, -3029/1024, 1162/1024),
        # (2172/1024, -1833/1024,  682/1024)
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    X = M.bfloat16()
    transpose = X.shape[-2] > X.shape[-1]
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in abc_list:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if transpose:
        X = X.mT
    return X

@torch.compile
def soft_cap(M: Tensor, alpha: float) -> Tensor:
    """Apply min(1, x) approximately to the singular values of a single matrix."""
    # Handle batched matrices by flattening batch dimensions
    orig_shape = M.shape
    if M.ndim > 2:
        M = M.reshape(-1, M.shape[-2], M.shape[-1])
    orig_dtype = M.dtype
    M = M.bfloat16()
    coeffs = [
        (1, -alpha),
        (1, alpha),
    ]
    transpose = M.shape[-1] > M.shape[-2]
    if transpose:
        M = M.mT
    for a, b in coeffs:
        A = M @ M.mT
        M = a * M + b * A @ M
    if transpose:
        M = M.mT
    
    if len(orig_shape) > 2:
        M = M.reshape(orig_shape)
    return M.to(orig_dtype)

def soft_cap_coupling(w_max: float, wd: float, max_update_norm: float) -> float:
    """Calculates the strength for soft cap that bounds singular values at w_max."""
    k = w_max * (1 - wd) + max_update_norm
    coeffs = torch.tensor([-k**9, 3 * k**7, -3 * k**5, 0.0, k - w_max], dtype=torch.float32)
    monic_coeffs = coeffs / coeffs[0]
    n = monic_coeffs.numel() - 1
    comp = torch.zeros((n, n), dtype=torch.float32)
    comp[1:, :-1] = torch.eye(n - 1)
    comp[0, :] = -monic_coeffs[1:]
    roots = torch.linalg.eigvals(comp)
    is_real = torch.abs(roots.imag) < 1e-6
    is_nonnegative = roots.real >= 0
    padded_reals = torch.where(is_real & is_nonnegative, roots.real, torch.ones_like(roots.real))
    return float(torch.min(padded_reals))

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, w_max=1, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, w_max=w_max, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    scale = torch.sqrt(torch.tensor(p_world.size(-2) / p_world.size(-1)))
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * scale)
                    
                    # Then apply soft cap projection with proper scaling
                    max_update_norm = group["lr"]   # since orthogonalize(G) * scale has unit RMS->RMS norm
                    alpha = soft_cap_coupling(group["w_max"], 0.0, max_update_norm * 1.14502 * 1.05)
                    p_world.copy_(scale * soft_cap(p_world / scale, alpha))
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = orthogonalize(g).flatten() # zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, W_max: float=1., use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.W_max = W_max
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
            # return out.reshape(*x.shape[:-1], -1) / self.W_max
        else:
            return F.linear(x, self.weight.type_as(x))
            # return F.linear(x, self.weight.type_as(x)) / self.W_max

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128, W_max: float=1.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scale = 24#head_dim  # it's just QK^T
        #self.softmax_scale = (head_dim**0.5) / (W_max**2)
        # self.softmax_scale = head_dim**0.5
        hdim = num_heads * head_dim
        self.attn_q = CastedLinear(dim, hdim, W_max=W_max)
        self.attn_k = CastedLinear(dim, hdim, W_max=W_max)
        self.attn_v = CastedLinear(dim, hdim, W_max=W_max)
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim, W_max=W_max)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.attn_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.attn_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.attn_v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = self.rotary(q), self.rotary(k)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.softmax_scale/self.head_dim).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y / 3)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, W_max: float=1.):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim, W_max=W_max)
        self.c_proj = CastedLinear(hdim, dim, W_max=W_max)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.gelu(x) / 1.1289  # 1.1289 is the max derivative of gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, num_layers: int, W_max: float=1.):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, W_max=W_max)
        self.mlp = MLP(dim, W_max=W_max)
        self.residual_scale = 1
        self.res_denom = 2*num_layers

    def forward(self, x: Tensor, block_mask: BlockMask):
        x = (1 - self.residual_scale/self.res_denom) * x + (self.residual_scale/self.res_denom) * self.attn(x, block_mask)
        x = (1 - self.residual_scale/self.res_denom) * x + (self.residual_scale/self.res_denom) * self.mlp(x)
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int,
        emb_w_max: float=1., W_max: float=1., lm_head_w_max: float=1., final_scale: float=1.,
    ):
        super().__init__()
        self.final_scale = final_scale
        self.embed = nn.Embedding(vocab_size, model_dim, max_norm=emb_w_max * model_dim**0.5)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, num_layers, W_max) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), W_max=lm_head_w_max)
                                    # use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor, return_logits_argmax: bool = False):
        assert input_seq.ndim == 1
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        if len(self.blocks) == 12:
            block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            max_act_rms_norm_list: list[Tensor | None] = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
            max_act_entry_list: list[Tensor | None]    = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        elif len(self.blocks) == 16:
            block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            max_act_rms_norm_list: list[Tensor | None] = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
            max_act_entry_list: list[Tensor | None]    = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        else:
            assert False, "Unsupported number of blocks"
        assert len(block_masks) == len(self.blocks)

        x: Tensor = self.embed(input_seq)[None]

        max_act_rms_norm_list[0] = x.norm(dim=-1).max() / (x.size(-1)**0.5)
        max_act_entry_list[0] = x.abs().max()

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, block_masks[i])
            max_act_rms_norm_list[i + 1] = x.norm(dim=-1).max() / (x.size(-1)**0.5)
            max_act_entry_list[i + 1] = x.abs().max()

        logits = self.lm_head(x).float()
        max_act_rms_norm_list[-1] = logits.norm(dim=-1).max() / (logits.size(-1)**0.5)
        max_act_entry_list[-1] = logits.abs().max()
        logits = logits * self.final_scale
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        # logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        max_logits = logits.max(dim=-1)[1] if return_logits_argmax else None
        return loss, max_logits, max_act_rms_norm_list, max_act_entry_list

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # arch configs
    emb_w_max = 1
    w_max = 1
    lm_head_w_max = 1  # equivalent to inverse temperature
    final_scale = 1.
    lm_head_muon = True
    # optimization
    num_iterations = 6200 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

import argparse
parser = argparse.ArgumentParser(description="Train a GPT model")
parser.add_argument("--head_lr", type=float, default=0.005, help="learning rate for head")
parser.add_argument("--qkv_lr", type=float, default=0.05, help="learning rate for QKV weights")
parser.add_argument("--hidden_lr", type=float, default=0.05, help="learning rate for hidden layers")
parser_args = parser.parse_args()
print(f"head_lr: {parser_args.head_lr}")
print(f"qkv_lr: {parser_args.qkv_lr}")
print(f"hidden_lr: {parser_args.hidden_lr}")

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
# model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                       emb_w_max=args.emb_w_max, W_max=args.w_max, lm_head_w_max=args.lm_head_w_max,
                       final_scale=args.final_scale,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "attn_" not in n]
qkv_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "attn_" in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# current training has no scalar parameters
assert len(scalar_params) == 0

@torch.compile
def project_embed_unembed_weights(model):
    with torch.no_grad():
        if not args.lm_head_muon:
            # Normalize head matrix rows [shape vocab_size, d_embed]
            lm_head_w_max_tensor = torch.tensor(args.lm_head_w_max, dtype=torch.float32, device=device)
            rms = torch.norm(model.lm_head.weight, dim=-1, keepdim=True) * model.lm_head.weight.shape[-1]**0.5
            model.lm_head.weight.div_(torch.maximum(rms, lm_head_w_max_tensor) / lm_head_w_max_tensor + 1e-12)
project_embed_unembed_weights(model)
# Approximately orthogonalize hidden matrix parameters at init
with torch.no_grad():
    for p in hidden_matrix_params + qkv_params:
        p.copy_(orthogonalize(p).float() * torch.sqrt(torch.tensor(p.size(0) / p.size(1))))

# init the optimizer(s)
# 
if args.lm_head_muon:
    adam_params = [dict(params=embed_params, lr=0.1)]
else:
    adam_params = [dict(params=head_params, lr=parser_args.head_lr), dict(params=embed_params, lr=0.1)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=parser_args.hidden_lr, momentum=0.95, w_max=args.w_max, rank=rank, world_size=world_size)
optimizer3 = Muon(qkv_params, lr=parser_args.qkv_lr, momentum=0.95, w_max=args.w_max, rank=rank, world_size=world_size)
if args.lm_head_muon:
    optimizer4 = Muon(head_params, lr=parser_args.head_lr, momentum=0.95, w_max=args.lm_head_w_max, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]  # optimizers[1:] must all be muon optimizers for momentum warmup
else:
    optimizers = [optimizer1, optimizer2, optimizer3]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    loss, _, _, _ = model(inputs.to(torch.int32), targets, get_window_size_blocks(0))
    loss.backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()

        # print all weight norms
        rms_to_rms_norm = lambda w: torch.linalg.norm(w.float(), ord=2)*(w.shape[1]/w.shape[0])**0.5
        if master_process:
            print0(">>> Weights:", console=True)
            for name, weight in model.named_parameters():
                weight_shape = str(tuple(weight.shape))
                if "embed" in name:
                    print0(f"{name:<40} {weight_shape:<13}:  l1->RMS:{torch.max(weight.norm(dim=-1)) / weight.shape[-1]**0.5:.4f}, RMS->RMS:{rms_to_rms_norm(weight):.4f}", console=True)
                elif "lm_head" in name:
                    print0(f"{name:<40} {weight_shape:<13}: RMS->INF:{torch.max(weight.norm(dim=-1)) * weight.shape[-1]**0.5:.4f}, RMS->RMS:{rms_to_rms_norm(weight):.4f}", console=True)
                elif len(weight.shape) == 3:
                    for i, w in enumerate(weight):
                        print0(f"{name:<37} #{i:<2} {weight_shape:<13}: RMS->RMS:{rms_to_rms_norm(w):.4f}", console=True)
                else:
                    print0(f"{name:<40} {weight_shape:<13}: RMS->RMS:{rms_to_rms_norm(weight):.4f}", console=True)

        # run val step
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_step in range(val_steps):
                inputs, targets = next(val_loader)
                loss, pred, max_act_rms_norm_list, max_act_entry_list = model(inputs, targets, get_window_size_blocks(step), return_logits_argmax=True)
                if master_process and val_step == 0:
                    print0(">>> Act RMS Norms:", console=True)
                    print0(f"Embed:     {max_act_rms_norm_list[0]:.4f}", console=True)
                    for i, max_act_rms_norm in enumerate(max_act_rms_norm_list[1:-1]):
                        print0(f"Block #{i}: {max_act_rms_norm:.4f}", console=True)
                    print0(f"Logits:    {max_act_rms_norm_list[-1]:.4f}", console=True)
                    print0(">>> Act Max Entries:", console=True)
                    print0(f"Embed:     {max_act_entry_list[0]:.4f}", console=True)
                    for i, max_entry in enumerate(max_act_entry_list[1:-1]):
                        print0(f"Block #{i}: {max_entry:.4f}", console=True)
                    print0(f"Logits:    {max_act_entry_list[-1]:.4f}", console=True)
                val_loss += loss
                val_correct += (pred == targets).sum().item()
                val_total += targets.numel()
        val_loss /= val_steps
        val_acc = val_correct / val_total
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(torch.tensor([val_correct, val_total], device="cuda"), op=dist.ReduceOp.SUM)
        val_acc = val_correct / val_total
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    loss, _, _, _ = model(inputs, targets, get_window_size_blocks(step))
    loss.backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # Print grads
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        if master_process:
            print0(">>> Grads:", console=True)
            for name, weight in model.named_parameters():
                if weight.grad is None:
                    continue
                weight_shape = str(tuple(weight.shape))
                if "embed" in name:
                    print0(f"{name:<40} {weight_shape:<13}:  l1->RMS:{torch.max(weight.grad.norm(dim=-1)) / weight.shape[-1]**0.5:.4f}, RMS->RMS:{rms_to_rms_norm(weight.grad):.4f}", console=True)
                elif "lm_head" in name:
                    print0(f"{name:<40} {weight_shape:<13}: RMS->INF:{torch.max(weight.grad.norm(dim=-1)) * weight.shape[-1]**0.5:.4f}, RMS->RMS:{rms_to_rms_norm(weight.grad):.4f}", console=True)
                else:
                    print0(f"{name:<40} {weight_shape:<13}: RMS->RMS:{rms_to_rms_norm(weight.grad):.4f}", console=True)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for opt_muon in optimizers[1:]:
        for group in opt_muon.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        opt.step()
    project_embed_unembed_weights(model)
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    if (step + 1) % 10 == 0:
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()


====================================================================================================
Running Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
Running PyTorch 2.8.0.dev20250316+cu126 compiled for CUDA 12.6
Wed May 14 00:19:28 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:19:00.0 Off |                    0 |
| N/A   27C    P0            110W /  700W |    5808MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:3B:00.0 Off |                    0 |
| N/A   26C    P0            117W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:4C:00.0 Off |                    0 |
| N/A   25C    P0            112W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:5D:00.0 Off |                    0 |
| N/A   27C    P0            110W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  |   00000000:9B:00.0 Off |                    0 |
| N/A   27C    P0            109W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:BB:00.0 Off |                    0 |
| N/A   26C    P0            110W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  |   00000000:CB:00.0 Off |                    0 |
| N/A   26C    P0            109W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  |   00000000:DB:00.0 Off |                    0 |
| N/A   24C    P0            110W /  700W |    1498MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

====================================================================================================
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:1.1172, RMS->RMS:31.1008
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1461
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1462
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0372
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1460
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0372
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1461
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0373
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1461
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1462
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1464
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0379
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1457
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1458
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0370
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1462
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1464
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1467
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0367
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0372
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1459
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0366
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1460
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1465
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1456
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0368
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1.1462
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:1.1461
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:1.1465
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:1.0367
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1.1465
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:1.1461
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:1.1458
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:1.0365
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:1.1463
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:1.1457
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:1.1464
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:1.0370
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.0000
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:0.0000, RMS->RMS:-0.0000
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9219
Block #1: 0.8438
Block #2: 0.7773
Block #3: 0.7148
Block #4: 0.6562
Block #5: 0.6016
Block #6: 0.5508
Block #7: 0.5078
Block #8: 0.4668
Block #9: 0.4277
Block #10: 0.3926
Block #11: 0.3613
Logits:    0.0000
>>> Act Max Entries:
Embed:     5.2812
Block #0: 4.8438
Block #1: 4.4375
Block #2: 4.0625
Block #3: 3.7344
Block #4: 3.4219
Block #5: 3.1406
Block #6: 2.8906
Block #7: 2.6562
Block #8: 2.4375
Block #9: 2.2344
Block #10: 2.0469
Block #11: 1.8828
Logits:    0.0000
step:0/6200 val_loss:10.8258 val_acc:0.0013 train_time:0ms step_avg:0.02ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:0.0000, RMS->RMS:-0.0000
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.0000
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.0000
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.0000
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.0000
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:145277.8906, RMS->RMS:673.8903
step:10/6200 train_time:1467ms step_avg:146.67ms
step:20/6200 train_time:2906ms step_avg:145.31ms
step:30/6200 train_time:4344ms step_avg:144.79ms
step:40/6200 train_time:5782ms step_avg:144.55ms
step:50/6200 train_time:7222ms step_avg:144.44ms
step:60/6200 train_time:8665ms step_avg:144.41ms
step:70/6200 train_time:10109ms step_avg:144.41ms
step:80/6200 train_time:11550ms step_avg:144.38ms
step:90/6200 train_time:12993ms step_avg:144.36ms
step:100/6200 train_time:14438ms step_avg:144.38ms
step:110/6200 train_time:15882ms step_avg:144.38ms
step:120/6200 train_time:17326ms step_avg:144.38ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.1562, RMS->RMS:431.1929
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9550
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9613
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9871
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9861
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9859
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9838
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9586
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9605
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9858
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9864
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9834
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9595
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9875
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9851
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9855
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9831
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9610
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9869
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9855
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9827
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9603
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9872
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9866
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9823
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9634
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9648
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9862
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9850
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9847
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9825
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9616
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9865
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9842
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9823
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9627
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9861
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9844
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9850
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9819
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9639
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9863
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9857
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9849
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9811
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9867
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9860
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9845
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9806
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9858
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9846
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9846
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9804
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9639
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9859
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9845
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9845
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9796
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:81.7822, RMS->RMS:0.5815
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9180
Block #3: 0.8906
Block #4: 0.8633
Block #5: 0.8359
Block #6: 0.8125
Block #7: 0.7891
Block #8: 0.7617
Block #9: 0.7383
Block #10: 0.7148
Block #11: 0.6914
Logits:    0.3764
>>> Act Max Entries:
Embed:     5.0312
Block #0: 4.7812
Block #1: 4.5312
Block #2: 4.2812
Block #3: 4.0625
Block #4: 3.8438
Block #5: 3.6406
Block #6: 3.4688
Block #7: 3.2969
Block #8: 3.1250
Block #9: 2.9688
Block #10: 2.8281
Block #11: 2.7031
Logits:    17.3750
step:125/6200 val_loss:6.4728 val_acc:0.1644 train_time:18082ms step_avg:144.66ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:4.8438, RMS->RMS:16.6876
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:101.5418
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:50.3937
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:129.4594
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:269.6207
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:82.5781
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1271.0750
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:132.1972
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:77.7322
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:122.6441
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:290.7558
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:84.4693
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1298.8943
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:121.8913
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:75.8357
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:137.9725
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:281.6259
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:85.4368
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1323.6447
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:87.6553
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:48.1023
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:128.9412
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:309.0467
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:87.3495
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1356.5704
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:125.0442
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:47.0605
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:140.3774
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:306.8461
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:88.2985
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1387.7982
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:120.2719
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:66.1077
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:145.1608
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:300.0239
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:89.1166
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1415.9452
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:117.7700
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:52.9017
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:147.5622
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:295.1639
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:89.1545
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1447.0704
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:143.0858
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:59.9767
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:154.3769
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:326.1632
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:89.5513
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1484.4890
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:138.3367
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:77.3466
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:156.8764
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:330.4271
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:88.9735
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1532.0320
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:163.0946
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:74.4767
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:161.6429
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:324.9488
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:89.2691
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:1569.1295
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:213.1049
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:96.5086
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:157.7081
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:357.0789
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:90.2193
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:1613.9816
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:139.7077
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:51.4300
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:162.9278
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:344.8039
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:91.1335
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:1648.3049
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:124740.4141, RMS->RMS:719.6047
step:130/6200 train_time:18797ms step_avg:144.59ms
step:140/6200 train_time:20237ms step_avg:144.55ms
step:150/6200 train_time:21682ms step_avg:144.55ms
step:160/6200 train_time:23125ms step_avg:144.53ms
step:170/6200 train_time:24571ms step_avg:144.53ms
step:180/6200 train_time:26018ms step_avg:144.54ms
step:190/6200 train_time:27465ms step_avg:144.55ms
step:200/6200 train_time:28914ms step_avg:144.57ms
step:210/6200 train_time:30363ms step_avg:144.58ms
step:220/6200 train_time:31812ms step_avg:144.60ms
step:230/6200 train_time:33262ms step_avg:144.62ms
step:240/6200 train_time:34713ms step_avg:144.64ms
step:250/6200 train_time:36163ms step_avg:144.65ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.2812, RMS->RMS:401.1797
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9617
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9643
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9888
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9629
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9888
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9893
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9890
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9848
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9886
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9842
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9897
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9885
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9886
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9836
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9890
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9849
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9827
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9887
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9850
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9827
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9743
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9894
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9888
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9819
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9754
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9886
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9838
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9813
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:109.1150, RMS->RMS:0.8919
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8438
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.5888
>>> Act Max Entries:
Embed:     5.3125
Block #0: 4.9375
Block #1: 4.5938
Block #2: 4.2812
Block #3: 3.9844
Block #4: 3.7031
Block #5: 3.4531
Block #6: 3.2344
Block #7: 3.0938
Block #8: 2.9531
Block #9: 2.8125
Block #10: 2.6875
Block #11: 2.5625
Logits:    19.8750
step:250/6200 val_loss:5.9677 val_acc:0.1763 train_time:36197ms step_avg:144.79ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:3.7656, RMS->RMS:13.2970
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:394.4517
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:176.1904
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:194.1252
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:437.0238
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:123.0086
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2165.2683
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:300.4000
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:151.1547
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:220.8792
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:458.4559
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.9067
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2264.5232
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:416.0479
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:200.3347
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:203.9233
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:456.0116
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.8828
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2376.0835
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:286.6498
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:132.5348
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:221.8595
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:524.1551
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.1022
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2488.0249
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:275.6670
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:130.6874
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:248.7476
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:541.6921
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.7971
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2596.9534
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:294.3711
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:185.4517
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:248.9659
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:524.9618
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.3407
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2692.3711
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:360.8882
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:193.6381
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:247.1531
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:499.3875
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.7214
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2773.3389
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:296.2475
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:106.9620
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:274.3388
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:568.8246
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.0980
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2854.3022
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:357.1697
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:173.7440
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:278.9572
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:582.0692
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.5389
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2949.5845
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:350.6159
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:163.9074
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:293.2673
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:564.4927
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:132.3224
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3022.9260
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:333.7596
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:212.3343
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:281.2051
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:582.9796
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:134.0137
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3104.3767
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:241.8281
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:86.1550
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:304.6897
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:607.0592
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:139.2523
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3167.5134
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:157059.3438, RMS->RMS:921.5840
step:260/6200 train_time:37649ms step_avg:144.81ms
step:270/6200 train_time:39097ms step_avg:144.80ms
step:280/6200 train_time:40545ms step_avg:144.80ms
step:290/6200 train_time:41995ms step_avg:144.81ms
step:300/6200 train_time:43446ms step_avg:144.82ms
step:310/6200 train_time:44896ms step_avg:144.83ms
step:320/6200 train_time:46347ms step_avg:144.83ms
step:330/6200 train_time:47799ms step_avg:144.85ms
step:340/6200 train_time:49251ms step_avg:144.86ms
step:350/6200 train_time:50702ms step_avg:144.86ms
step:360/6200 train_time:52154ms step_avg:144.87ms
step:370/6200 train_time:53605ms step_avg:144.88ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.5312, RMS->RMS:390.2568
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9642
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9879
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9690
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9894
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9895
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9676
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9848
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9894
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9772
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9895
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9838
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9793
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9835
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:126.6467, RMS->RMS:0.9412
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9180
Block #3: 0.8945
Block #4: 0.8633
Block #5: 0.8398
Block #6: 0.8125
Block #7: 0.7891
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6324
>>> Act Max Entries:
Embed:     5.4375
Block #0: 5.1250
Block #1: 4.8125
Block #2: 4.5312
Block #3: 4.2812
Block #4: 4.0312
Block #5: 3.8125
Block #6: 3.5938
Block #7: 3.4062
Block #8: 3.2188
Block #9: 3.0312
Block #10: 2.8594
Block #11: 2.6875
Logits:    19.5000
step:375/6200 val_loss:5.8143 val_acc:0.1834 train_time:54366ms step_avg:144.97ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:16.5000, RMS->RMS:56.9285
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:336.8956
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:116.8484
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:164.7478
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:393.5368
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:92.8290
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2231.5835
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:208.5126
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:113.3024
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:165.1136
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:453.8551
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:95.2827
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2345.7334
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:444.3611
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:268.4447
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:182.6443
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:472.4360
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:98.7159
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2471.3357
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:444.1007
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:248.8108
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:191.1839
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:478.9845
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:99.3509
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2580.3401
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:283.5868
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:156.8540
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:208.1803
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:449.6330
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:96.6431
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2691.6926
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:272.8829
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:164.2744
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:204.8974
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:506.0698
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:100.1043
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2807.7944
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:317.4619
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:163.4744
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:222.4489
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:451.1194
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:103.3884
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2904.1084
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:283.6716
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:179.6523
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:240.7531
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:551.9313
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:104.4806
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3006.5095
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:308.8495
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:151.9195
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:245.9210
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:537.4098
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:111.2119
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3107.0745
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:372.0291
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:171.7622
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:255.5467
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:565.3246
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:114.8343
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3219.2454
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:269.4424
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:168.1079
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:280.4911
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:580.4538
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:117.0404
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3327.8945
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:320.5637
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:214.5061
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:271.9505
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:536.8995
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:120.4105
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3416.4011
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:214208.8438, RMS->RMS:998.8620
step:380/6200 train_time:55079ms step_avg:144.95ms
step:390/6200 train_time:56528ms step_avg:144.94ms
step:400/6200 train_time:57980ms step_avg:144.95ms
step:410/6200 train_time:59430ms step_avg:144.95ms
step:420/6200 train_time:60883ms step_avg:144.96ms
step:430/6200 train_time:62334ms step_avg:144.96ms
step:440/6200 train_time:63786ms step_avg:144.97ms
step:450/6200 train_time:65237ms step_avg:144.97ms
step:460/6200 train_time:66690ms step_avg:144.98ms
step:470/6200 train_time:68152ms step_avg:145.00ms
step:480/6200 train_time:69612ms step_avg:145.03ms
step:490/6200 train_time:71074ms step_avg:145.05ms
step:500/6200 train_time:72536ms step_avg:145.07ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.8438, RMS->RMS:388.0355
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9653
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9668
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9764
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9782
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9733
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9778
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9791
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9833
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9855
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:134.0888, RMS->RMS:0.9459
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6423
>>> Act Max Entries:
Embed:     5.3750
Block #0: 5.0312
Block #1: 4.7188
Block #2: 4.4375
Block #3: 4.1562
Block #4: 3.9062
Block #5: 3.6719
Block #6: 3.4219
Block #7: 3.2031
Block #8: 3.0156
Block #9: 2.8594
Block #10: 2.7188
Block #11: 2.5938
Logits:    23.7500
step:500/6200 val_loss:5.7494 val_acc:0.1858 train_time:72570ms step_avg:145.14ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:11.5000, RMS->RMS:39.5439
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:708.8616
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:231.9976
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:271.8969
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:739.8022
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:152.7163
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3101.9509
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:386.0903
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:171.4317
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:294.4458
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:667.4438
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:150.7490
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3294.5156
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:894.0045
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:684.8672
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:285.2142
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:710.7521
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.8576
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3495.5745
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:460.1803
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:259.6711
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:317.2083
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:778.9888
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.4968
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3669.0786
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:668.3544
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:333.5613
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:376.3822
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:877.3344
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:144.5893
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3854.8032
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:299.5503
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:200.1262
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:348.7784
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:765.2374
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.9187
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4014.6956
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:310.5177
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:196.5257
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:367.5997
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:776.3237
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.9149
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4167.9722
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:371.6354
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:148.0629
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.1360
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:955.7603
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:153.6025
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4340.3008
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:559.8921
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:296.3478
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:413.0594
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:857.9035
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:166.4187
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4485.7671
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:399.5423
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:179.2566
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:423.3775
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:850.9310
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:173.3551
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4630.0928
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:374.2952
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:266.3625
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:420.4569
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:872.6743
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:179.4787
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4782.8481
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:333.7255
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:155.3776
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:466.9056
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1020.1375
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:195.7085
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4933.4751
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:189660.0156, RMS->RMS:1148.6680
step:510/6200 train_time:74039ms step_avg:145.17ms
step:520/6200 train_time:75498ms step_avg:145.19ms
step:530/6200 train_time:76959ms step_avg:145.20ms
step:540/6200 train_time:78416ms step_avg:145.21ms
step:550/6200 train_time:79876ms step_avg:145.23ms
step:560/6200 train_time:81335ms step_avg:145.24ms
step:570/6200 train_time:82792ms step_avg:145.25ms
step:580/6200 train_time:84253ms step_avg:145.26ms
step:590/6200 train_time:85715ms step_avg:145.28ms
step:600/6200 train_time:87177ms step_avg:145.30ms
step:610/6200 train_time:88638ms step_avg:145.31ms
step:620/6200 train_time:90098ms step_avg:145.32ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:6.7500, RMS->RMS:386.2538
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9735
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9714
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9702
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9770
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9785
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9789
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9754
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9840
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9850
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:134.9552, RMS->RMS:0.9491
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7891
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6434
>>> Act Max Entries:
Embed:     5.6875
Block #0: 5.2500
Block #1: 4.8750
Block #2: 4.5000
Block #3: 4.1562
Block #4: 3.8438
Block #5: 3.5625
Block #6: 3.2969
Block #7: 3.1250
Block #8: 2.9688
Block #9: 2.8281
Block #10: 2.6719
Block #11: 2.5312
Logits:    20.3750
step:625/6200 val_loss:5.7133 val_acc:0.1892 train_time:90863ms step_avg:145.38ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.6250, RMS->RMS:26.3303
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:689.1183
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:264.8087
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:235.8435
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:579.8777
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:117.5735
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2387.9456
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:298.5252
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:136.0242
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:213.4954
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:530.5115
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:112.4317
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2522.5425
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:770.6078
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:561.3258
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:234.8194
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:574.1525
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:108.0865
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2648.1743
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:247.9213
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:126.5480
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:220.7215
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:596.6727
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:107.9571
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2774.3518
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:553.8844
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:262.4175
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:268.9145
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:649.7418
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:109.4509
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2914.0408
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:236.8071
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:151.2229
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:263.4844
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:597.6003
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:110.3978
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3050.0642
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:282.0484
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:164.1367
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:288.3453
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:602.6740
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:109.8745
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3183.9573
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:394.7254
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:227.2723
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:305.3174
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:734.6267
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:111.1255
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3324.1152
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:437.4436
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:200.3853
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:322.5531
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:675.3901
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.1899
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3443.3125
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:387.0793
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:206.9893
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:313.7869
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:694.9755
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.7365
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3571.8491
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:359.9387
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:270.7306
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:328.2292
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:690.9536
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:124.8294
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3711.6160
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:271.4724
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:266.7726
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:346.7025
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:794.9305
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:130.7793
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3854.5298
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138941.4688, RMS->RMS:904.4501
step:630/6200 train_time:91581ms step_avg:145.37ms
step:640/6200 train_time:93043ms step_avg:145.38ms
step:650/6200 train_time:94506ms step_avg:145.39ms
step:660/6200 train_time:95966ms step_avg:145.40ms
step:670/6200 train_time:97425ms step_avg:145.41ms
step:680/6200 train_time:98886ms step_avg:145.42ms
step:690/6200 train_time:100349ms step_avg:145.43ms
step:700/6200 train_time:101813ms step_avg:145.45ms
step:710/6200 train_time:103275ms step_avg:145.46ms
step:720/6200 train_time:104738ms step_avg:145.47ms
step:730/6200 train_time:106198ms step_avg:145.48ms
step:740/6200 train_time:107659ms step_avg:145.49ms
step:750/6200 train_time:109121ms step_avg:145.49ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.3125, RMS->RMS:385.6870
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9714
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9765
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9787
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9785
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9763
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9838
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9849
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:135.8378, RMS->RMS:0.9516
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6461
>>> Act Max Entries:
Embed:     6.4375
Block #0: 5.9375
Block #1: 5.5000
Block #2: 5.0938
Block #3: 4.6875
Block #4: 4.3438
Block #5: 4.0312
Block #6: 3.7344
Block #7: 3.4531
Block #8: 3.2656
Block #9: 3.0938
Block #10: 2.9219
Block #11: 2.7656
Logits:    25.1250
step:750/6200 val_loss:5.7005 val_acc:0.1903 train_time:109154ms step_avg:145.54ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.5312, RMS->RMS:26.0795
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:974.2929
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:283.9996
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:248.4507
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:649.1651
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.7157
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2711.9324
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:383.8821
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:163.0277
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:235.7504
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:646.4838
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:121.8957
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2910.4575
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:995.5461
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:763.2978
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:263.1224
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:678.1812
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:115.0954
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3095.0061
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:539.0952
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:279.8003
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:260.2510
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:704.6088
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.7371
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3257.6614
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:839.8030
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:448.6981
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:293.5602
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:764.6558
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:124.5469
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3420.7063
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:260.8451
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:146.2684
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:303.1064
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:715.8558
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.6038
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3578.6567
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:304.3829
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:153.5449
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:342.3982
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:735.1636
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.9317
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3724.9910
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:384.2614
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:175.0196
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:340.9238
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:876.5042
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.0499
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3877.2031
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:618.3612
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:266.7419
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:378.3116
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:798.2111
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.5494
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4002.0010
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:615.3622
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:300.3644
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.3252
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:806.8322
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.2225
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4125.7231
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:358.1849
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:223.6198
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:380.3307
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:809.0817
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:166.8118
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4278.4277
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:488.7680
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:205.1912
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:403.5550
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:936.9427
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:174.3733
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4418.1772
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:127787.0781, RMS->RMS:835.6781
step:760/6200 train_time:110605ms step_avg:145.53ms
step:770/6200 train_time:112081ms step_avg:145.56ms
step:780/6200 train_time:113541ms step_avg:145.57ms
step:790/6200 train_time:115001ms step_avg:145.57ms
step:800/6200 train_time:116461ms step_avg:145.58ms
step:810/6200 train_time:117920ms step_avg:145.58ms
step:820/6200 train_time:119382ms step_avg:145.59ms
step:830/6200 train_time:120841ms step_avg:145.59ms
step:840/6200 train_time:122302ms step_avg:145.60ms
step:850/6200 train_time:123764ms step_avg:145.61ms
step:860/6200 train_time:125227ms step_avg:145.61ms
step:870/6200 train_time:126689ms step_avg:145.62ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.8438, RMS->RMS:384.1911
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9668
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9743
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9799
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9777
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9792
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9749
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9833
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9853
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:136.6913, RMS->RMS:0.9531
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6474
>>> Act Max Entries:
Embed:     6.0312
Block #0: 5.5625
Block #1: 5.1562
Block #2: 4.7812
Block #3: 4.4375
Block #4: 4.1250
Block #5: 3.8281
Block #6: 3.5938
Block #7: 3.3906
Block #8: 3.2031
Block #9: 3.0156
Block #10: 2.8438
Block #11: 2.7031
Logits:    19.5000
step:875/6200 val_loss:5.6916 val_acc:0.1895 train_time:127452ms step_avg:145.66ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.8125, RMS->RMS:30.5075
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1039.8862
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:404.3724
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:326.0980
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:874.5219
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:155.5494
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3504.7400
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:411.8891
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:167.5832
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.1002
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:783.8985
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:148.5463
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3712.4163
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1181.3811
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:949.6392
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:343.0040
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:834.7111
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:153.1725
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3911.3340
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:581.8535
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:297.9293
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:359.4984
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:887.8773
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.4967
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4074.0901
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:876.8373
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:526.9932
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:459.0784
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1015.8838
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.5218
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4260.6602
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:324.8843
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:225.8421
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:392.2179
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:865.3807
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:173.1583
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4432.2373
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:335.2507
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:191.9685
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.5304
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:885.1252
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:177.2531
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4601.7900
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:500.3138
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:182.7451
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:430.8007
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1077.1248
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:182.7906
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4784.8760
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:623.9183
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:323.5523
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:451.8464
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:937.5117
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:195.2658
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4929.8247
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:654.5444
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:334.8285
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:443.9804
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:941.9641
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:196.3811
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5085.3389
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:378.9199
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:240.4445
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:469.5322
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:941.5359
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:200.9468
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5258.5815
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:574.1125
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:227.0084
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:500.5592
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1136.3566
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:211.5605
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5424.9106
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:167586.5156, RMS->RMS:1198.8945
step:880/6200 train_time:128169ms step_avg:145.65ms
step:890/6200 train_time:129626ms step_avg:145.65ms
step:900/6200 train_time:131087ms step_avg:145.65ms
step:910/6200 train_time:132551ms step_avg:145.66ms
step:920/6200 train_time:134012ms step_avg:145.66ms
step:930/6200 train_time:135482ms step_avg:145.68ms
step:940/6200 train_time:136948ms step_avg:145.69ms
step:950/6200 train_time:138415ms step_avg:145.70ms
step:960/6200 train_time:139881ms step_avg:145.71ms
step:970/6200 train_time:141348ms step_avg:145.72ms
step:980/6200 train_time:142815ms step_avg:145.73ms
step:990/6200 train_time:144281ms step_avg:145.74ms
step:1000/6200 train_time:145746ms step_avg:145.75ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.6250, RMS->RMS:383.4655
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9876
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9760
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9773
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9786
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9785
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9790
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9769
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9843
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9856
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:137.9150, RMS->RMS:0.9542
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7461
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6484
>>> Act Max Entries:
Embed:     5.5938
Block #0: 5.1562
Block #1: 4.7500
Block #2: 4.4062
Block #3: 4.1562
Block #4: 3.9219
Block #5: 3.6875
Block #6: 3.4688
Block #7: 3.2812
Block #8: 3.1094
Block #9: 2.9375
Block #10: 2.7656
Block #11: 2.6250
Logits:    19.7500
step:1000/6200 val_loss:5.6806 val_acc:0.1914 train_time:145780ms step_avg:145.78ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.4062, RMS->RMS:19.4894
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1115.4075
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:347.8307
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:281.6653
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:729.4906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.4164
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2997.2053
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:430.6547
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:219.2707
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:261.4794
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:685.1652
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.2499
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3182.0728
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1097.8809
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:842.0436
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:271.0532
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:700.6623
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:118.6286
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3342.5696
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:500.4106
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:242.0946
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:268.5014
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:745.6702
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:122.9166
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3504.7329
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:853.8768
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:442.9234
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:309.4536
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:828.4548
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.9239
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3668.4150
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:308.5109
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:193.9718
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:318.7024
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:748.4031
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.8332
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3830.4883
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:380.1496
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:214.9648
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:348.8363
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:763.0114
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:144.3225
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3985.9517
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:496.5465
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:188.3267
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:370.4558
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:920.0029
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:149.1861
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4152.5371
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:649.8533
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:298.4348
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:403.0007
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:830.6565
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:166.1620
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4276.2622
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:806.7557
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:383.0303
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:400.0652
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:814.6309
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.6199
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4394.4917
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:540.4476
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:385.9538
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:383.5495
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:805.3510
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:177.5697
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4529.1924
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:506.4883
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:146.1103
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:420.8161
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:978.1036
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:185.3025
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4684.7744
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:186409.4062, RMS->RMS:1071.3937
step:1010/6200 train_time:147234ms step_avg:145.78ms
step:1020/6200 train_time:148718ms step_avg:145.80ms
step:1030/6200 train_time:150187ms step_avg:145.81ms
step:1040/6200 train_time:151654ms step_avg:145.82ms
step:1050/6200 train_time:153122ms step_avg:145.83ms
step:1060/6200 train_time:154591ms step_avg:145.84ms
step:1070/6200 train_time:156061ms step_avg:145.85ms
step:1080/6200 train_time:157528ms step_avg:145.86ms
step:1090/6200 train_time:158995ms step_avg:145.87ms
step:1100/6200 train_time:160464ms step_avg:145.88ms
step:1110/6200 train_time:161934ms step_avg:145.89ms
step:1120/6200 train_time:163402ms step_avg:145.89ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:6.2188, RMS->RMS:383.9922
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9675
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9879
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9876
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9750
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9701
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9775
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9783
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9743
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9796
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9792
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9758
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9845
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9855
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:137.6058, RMS->RMS:0.9552
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9180
Block #3: 0.8906
Block #4: 0.8633
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7422
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6495
>>> Act Max Entries:
Embed:     6.3750
Block #0: 6.0312
Block #1: 5.6875
Block #2: 5.3438
Block #3: 5.0312
Block #4: 4.7500
Block #5: 4.4688
Block #6: 4.2188
Block #7: 3.9688
Block #8: 3.7344
Block #9: 3.5156
Block #10: 3.3125
Block #11: 3.1094
Logits:    19.5000
step:1125/6200 val_loss:5.6745 val_acc:0.1915 train_time:164169ms step_avg:145.93ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.4375, RMS->RMS:29.2737
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:749.9316
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:293.7480
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:300.5584
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:762.8126
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.6378
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2985.7749
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:246.8006
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:139.7043
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:296.1936
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:667.2668
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.1982
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3116.2710
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:964.6991
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:805.6901
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:293.6535
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:702.5416
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.0062
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3269.0635
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:815.3456
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:491.1480
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:297.0958
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:743.0400
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:144.3386
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3397.0286
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:608.1479
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:305.0738
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:345.5044
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:844.0908
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:150.6413
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3514.3684
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:320.9160
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:155.0786
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:330.4954
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:710.3313
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.0664
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3623.5952
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:317.4765
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:181.6144
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:331.8330
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:718.9424
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:164.7919
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3726.4106
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:344.5511
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:269.9681
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:353.3602
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:879.0459
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.7203
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3841.2141
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:367.7024
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:163.3865
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:366.4321
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:757.7903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:172.9766
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3946.7795
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:570.6276
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:260.1656
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:376.2842
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:755.3771
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:178.8697
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4045.1018
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:288.4764
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:208.2590
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:384.7535
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:768.3297
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:182.9663
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4162.3774
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:446.6817
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:405.9776
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:400.4776
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:913.6249
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:188.1561
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4288.3921
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:128054.0703, RMS->RMS:900.5652
step:1130/6200 train_time:164894ms step_avg:145.92ms
step:1140/6200 train_time:166363ms step_avg:145.93ms
step:1150/6200 train_time:167833ms step_avg:145.94ms
step:1160/6200 train_time:169301ms step_avg:145.95ms
step:1170/6200 train_time:170768ms step_avg:145.96ms
step:1180/6200 train_time:172236ms step_avg:145.96ms
step:1190/6200 train_time:173702ms step_avg:145.97ms
step:1200/6200 train_time:175175ms step_avg:145.98ms
step:1210/6200 train_time:176644ms step_avg:145.99ms
step:1220/6200 train_time:178115ms step_avg:146.00ms
step:1230/6200 train_time:179584ms step_avg:146.00ms
step:1240/6200 train_time:181056ms step_avg:146.01ms
step:1250/6200 train_time:182523ms step_avg:146.02ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.2500, RMS->RMS:384.2337
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9884
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9676
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9766
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9772
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9793
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9805
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9763
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9844
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9856
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:137.5813, RMS->RMS:0.9555
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7656
Block #9: 0.7461
Block #10: 0.7188
Block #11: 0.6953
Logits:    0.6506
>>> Act Max Entries:
Embed:     6.2188
Block #0: 5.8438
Block #1: 5.5000
Block #2: 5.1875
Block #3: 4.8750
Block #4: 4.5938
Block #5: 4.3125
Block #6: 4.0625
Block #7: 3.8281
Block #8: 3.6094
Block #9: 3.3906
Block #10: 3.1875
Block #11: 3.0000
Logits:    19.5000
step:1250/6200 val_loss:5.6697 val_acc:0.1904 train_time:182557ms step_avg:146.05ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:10.5625, RMS->RMS:36.4096
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:965.9871
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:369.0428
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:281.7265
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:782.7479
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.3345
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3245.4788
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:411.1770
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:246.8334
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:318.2693
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:746.3680
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.7403
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3446.9854
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:678.2394
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:475.5911
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:322.3791
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:797.1520
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.8024
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3643.9849
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:888.4882
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:610.7939
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:328.3207
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:798.3349
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.9963
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3836.7771
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:417.7304
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:155.7310
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:342.4266
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:957.6313
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:141.7344
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4039.4617
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:507.6565
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:281.6745
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:374.1315
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:814.4327
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.0206
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4233.8896
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:403.7227
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:225.4253
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:398.1304
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:842.7062
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:155.6411
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4423.6123
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:882.5381
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:406.5603
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:447.0131
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1013.8777
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:172.6562
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4608.9595
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:352.2235
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:148.2372
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:432.8882
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:884.4013
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:185.1131
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4760.4282
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:373.0020
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:153.0357
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:427.5355
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:918.0454
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:191.4574
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4921.4487
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:358.5450
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:195.0731
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:479.1318
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:914.6835
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:206.7127
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5083.8052
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:497.8278
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:207.8834
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:472.2877
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1096.2466
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:217.8486
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5247.3032
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:153705.8594, RMS->RMS:1111.6361
step:1260/6200 train_time:184013ms step_avg:146.04ms
step:1270/6200 train_time:185479ms step_avg:146.05ms
step:1280/6200 train_time:186972ms step_avg:146.07ms
step:1290/6200 train_time:188439ms step_avg:146.08ms
step:1300/6200 train_time:189907ms step_avg:146.08ms
step:1310/6200 train_time:191373ms step_avg:146.09ms
step:1320/6200 train_time:192843ms step_avg:146.09ms
step:1330/6200 train_time:194311ms step_avg:146.10ms
step:1340/6200 train_time:195780ms step_avg:146.10ms
step:1350/6200 train_time:197246ms step_avg:146.11ms
step:1360/6200 train_time:198713ms step_avg:146.11ms
step:1370/6200 train_time:200182ms step_avg:146.12ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.2500, RMS->RMS:382.8382
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9795
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9797
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9723
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9796
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9758
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9842
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:141.2868, RMS->RMS:0.9568
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9180
Block #3: 0.8906
Block #4: 0.8672
Block #5: 0.8398
Block #6: 0.8164
Block #7: 0.7930
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7188
Block #11: 0.6992
Logits:    0.6526
>>> Act Max Entries:
Embed:     6.0312
Block #0: 5.5938
Block #1: 5.1875
Block #2: 4.8125
Block #3: 4.5000
Block #4: 4.2500
Block #5: 4.0000
Block #6: 3.7969
Block #7: 3.6094
Block #8: 3.4375
Block #9: 3.2656
Block #10: 3.1094
Block #11: 2.9688
Logits:    20.2500
step:1375/6200 val_loss:5.6633 val_acc:0.1912 train_time:200952ms step_avg:146.15ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.9375, RMS->RMS:30.8285
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1019.2060
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:358.8810
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:309.4671
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:814.6067
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.5807
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3262.3179
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:430.9494
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:257.1439
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:319.2691
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:723.6387
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.4004
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3433.7200
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1031.1348
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:819.5640
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:340.1428
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:772.2009
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.3859
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3600.7109
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:494.3166
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:228.5402
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:332.8206
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:786.0999
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.2336
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3742.1956
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:716.5028
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:385.4138
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:418.5773
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:951.8821
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:152.0520
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3902.4431
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:295.0808
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:175.1933
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:363.1031
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:786.6604
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.7563
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4060.0471
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:341.0181
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:199.2907
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:370.9621
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:817.7668
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:161.8836
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4218.8828
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:611.9037
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:320.8180
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:400.0884
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:986.8962
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.1191
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4384.7041
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:528.5562
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:274.7822
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:414.1252
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:858.1446
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:180.7726
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4519.3081
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:680.6662
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:362.4055
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:410.3792
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:852.0609
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:186.4332
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4653.6245
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:418.8546
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:292.3815
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:428.6004
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:865.3062
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:189.5414
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4806.1802
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:538.3547
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:481.6277
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:454.4634
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1061.4785
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:195.9510
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4970.9917
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:166222.8906, RMS->RMS:1126.9583
step:1380/6200 train_time:201680ms step_avg:146.15ms
step:1390/6200 train_time:203175ms step_avg:146.17ms
step:1400/6200 train_time:204670ms step_avg:146.19ms
step:1410/6200 train_time:206165ms step_avg:146.22ms
step:1420/6200 train_time:207662ms step_avg:146.24ms
step:1430/6200 train_time:209157ms step_avg:146.26ms
step:1440/6200 train_time:210654ms step_avg:146.29ms
step:1450/6200 train_time:212147ms step_avg:146.31ms
step:1460/6200 train_time:213643ms step_avg:146.33ms
step:1470/6200 train_time:215135ms step_avg:146.35ms
step:1480/6200 train_time:216635ms step_avg:146.37ms
step:1490/6200 train_time:218135ms step_avg:146.40ms
step:1500/6200 train_time:219635ms step_avg:146.42ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.7812, RMS->RMS:381.0789
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9675
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9879
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9747
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9697
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9761
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9798
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9813
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9805
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9754
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9842
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:139.6399, RMS->RMS:0.9566
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8438
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.6558
>>> Act Max Entries:
Embed:     6.2188
Block #0: 5.8125
Block #1: 5.4375
Block #2: 5.0625
Block #3: 4.7188
Block #4: 4.4062
Block #5: 4.1250
Block #6: 3.8594
Block #7: 3.6094
Block #8: 3.3750
Block #9: 3.1562
Block #10: 2.9531
Block #11: 2.7656
Logits:    19.8750
step:1500/6200 val_loss:5.6477 val_acc:0.1918 train_time:219669ms step_avg:146.45ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.1250, RMS->RMS:28.5606
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1231.0021
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:373.6611
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:379.3493
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:1031.9335
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.8260
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4216.0884
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1003.4335
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:536.8250
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:363.7645
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1015.8180
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:154.6293
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4496.6694
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1202.4752
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:834.7928
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:403.8802
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1106.2290
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:157.6823
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4742.8584
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:759.5895
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:535.7775
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:387.3282
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1179.8484
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.1962
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5001.7739
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:835.8700
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:398.6735
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:445.7614
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1197.4308
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:175.4841
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5244.0693
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:464.1851
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:188.4138
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:485.8711
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1187.5986
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:176.3521
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5509.6152
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:400.2273
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:182.8244
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:495.6749
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1234.9491
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:180.6702
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5786.8379
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1299.6842
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:714.3499
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:575.8771
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1315.5819
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:194.2388
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6042.5869
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1037.7955
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:535.0117
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:583.1583
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1321.2390
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:217.8911
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6250.5527
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:963.2214
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:601.2452
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:568.4886
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1361.4987
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:223.3243
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6476.9473
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:693.1744
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:466.6531
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:576.5419
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1277.1699
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:228.3273
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:6703.0703
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:648.9623
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:519.2548
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:621.2900
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1429.9971
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:244.3714
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:6951.9556
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:216611.7188, RMS->RMS:1437.8922
step:1510/6200 train_time:221154ms step_avg:146.46ms
step:1520/6200 train_time:222652ms step_avg:146.48ms
step:1530/6200 train_time:224172ms step_avg:146.52ms
step:1540/6200 train_time:225664ms step_avg:146.54ms
step:1550/6200 train_time:227159ms step_avg:146.55ms
step:1560/6200 train_time:228655ms step_avg:146.57ms
step:1570/6200 train_time:230154ms step_avg:146.60ms
step:1580/6200 train_time:231652ms step_avg:146.62ms
step:1590/6200 train_time:233146ms step_avg:146.63ms
step:1600/6200 train_time:234644ms step_avg:146.65ms
step:1610/6200 train_time:236141ms step_avg:146.67ms
step:1620/6200 train_time:237634ms step_avg:146.69ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.5000, RMS->RMS:382.6847
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9675
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9742
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9723
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9805
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9850
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9742
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9806
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9836
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9847
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138.6732, RMS->RMS:0.9577
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6598
>>> Act Max Entries:
Embed:     5.8438
Block #0: 5.4375
Block #1: 5.0625
Block #2: 4.7188
Block #3: 4.4062
Block #4: 4.1250
Block #5: 3.9219
Block #6: 3.7344
Block #7: 3.5469
Block #8: 3.3594
Block #9: 3.1875
Block #10: 3.0312
Block #11: 2.8750
Logits:    19.5000
step:1625/6200 val_loss:5.6386 val_acc:0.1937 train_time:238417ms step_avg:146.72ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.8125, RMS->RMS:30.4654
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:935.9924
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:333.4680
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:321.6597
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:814.2077
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:154.4185
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3281.1472
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:522.4416
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:309.6195
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.2373
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:819.8057
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:154.6891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3453.2974
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1054.2848
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:899.8552
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.0295
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:922.8239
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.9007
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3652.0459
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1275.6521
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:736.6274
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.2903
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:951.6331
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.7236
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3790.3230
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:624.7498
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:308.5889
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:377.3727
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:930.6088
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:172.3113
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3921.9939
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:416.4063
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:215.7012
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:386.9502
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:897.4405
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:175.9869
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4052.3716
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:389.1477
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:173.8525
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:384.5275
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:927.6614
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:179.4398
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4190.3066
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:343.4843
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:131.5697
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:396.7997
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:967.6114
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:185.5516
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4325.7139
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:510.6749
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:199.1857
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:438.1641
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:953.1337
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:188.4833
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4453.2319
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:784.7989
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:334.1284
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:437.0608
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:993.5961
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:193.8035
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4569.1685
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:446.5804
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:184.6347
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:427.2079
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:935.0255
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:197.3611
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4699.9067
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:543.1661
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:202.3299
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:443.1252
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1011.3784
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:202.3685
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4819.9199
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:134695.9062, RMS->RMS:935.5049
step:1630/6200 train_time:239154ms step_avg:146.72ms
step:1640/6200 train_time:240650ms step_avg:146.74ms
step:1650/6200 train_time:242144ms step_avg:146.75ms
step:1660/6200 train_time:243641ms step_avg:146.77ms
step:1670/6200 train_time:245144ms step_avg:146.79ms
step:1680/6200 train_time:246641ms step_avg:146.81ms
step:1690/6200 train_time:248139ms step_avg:146.83ms
step:1700/6200 train_time:249632ms step_avg:146.84ms
step:1710/6200 train_time:251123ms step_avg:146.86ms
step:1720/6200 train_time:252613ms step_avg:146.87ms
step:1730/6200 train_time:254110ms step_avg:146.88ms
step:1740/6200 train_time:255603ms step_avg:146.90ms
step:1750/6200 train_time:257103ms step_avg:146.92ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3125, RMS->RMS:381.8305
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9648
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9724
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9750
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9723
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9761
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9801
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9798
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9807
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9760
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9835
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138.5980, RMS->RMS:0.9573
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6588
>>> Act Max Entries:
Embed:     6.8750
Block #0: 6.4062
Block #1: 5.9688
Block #2: 5.5625
Block #3: 5.1875
Block #4: 4.8125
Block #5: 4.4688
Block #6: 4.1562
Block #7: 3.8750
Block #8: 3.6094
Block #9: 3.3594
Block #10: 3.1250
Block #11: 2.9219
Logits:    20.1250
step:1750/6200 val_loss:5.6405 val_acc:0.1909 train_time:257137ms step_avg:146.94ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.6875, RMS->RMS:21.4862
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1049.8634
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:403.2381
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:358.2434
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:955.5402
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.5283
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3818.3113
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1015.3526
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:620.0258
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:366.8439
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:955.3794
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.2547
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4085.4031
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1383.9717
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1347.1425
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:466.8552
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1041.0747
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:160.0960
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4293.3745
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:835.0040
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:477.5623
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:394.6838
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1090.2888
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.6209
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4466.0674
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:782.4595
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:440.1162
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:514.0898
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1130.0911
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:177.2506
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4660.2666
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:386.6330
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:195.9923
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:449.2161
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1063.0636
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:180.3469
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4858.4189
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:395.8773
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:164.4329
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:454.0043
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1107.5360
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:188.8559
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5055.5581
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:736.0417
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:391.3743
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:484.1597
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1165.6705
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:190.4743
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5254.5483
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:730.2875
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:353.8982
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:503.0639
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1158.7704
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:210.1729
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5417.4434
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1042.0812
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:574.5034
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:488.7499
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1172.7476
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:215.7727
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5589.9883
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:596.3101
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:395.4601
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:504.9349
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1116.2086
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:212.7230
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5773.0620
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:524.1599
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:392.0552
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:548.8779
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1258.5217
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:220.5226
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5987.4961
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:217400.2656, RMS->RMS:1304.4127
step:1760/6200 train_time:258621ms step_avg:146.94ms
step:1770/6200 train_time:260113ms step_avg:146.96ms
step:1780/6200 train_time:261641ms step_avg:146.99ms
step:1790/6200 train_time:263133ms step_avg:147.00ms
step:1800/6200 train_time:264626ms step_avg:147.01ms
step:1810/6200 train_time:266123ms step_avg:147.03ms
step:1820/6200 train_time:267622ms step_avg:147.05ms
step:1830/6200 train_time:269114ms step_avg:147.06ms
step:1840/6200 train_time:270616ms step_avg:147.07ms
step:1850/6200 train_time:272115ms step_avg:147.09ms
step:1860/6200 train_time:273615ms step_avg:147.10ms
step:1870/6200 train_time:275115ms step_avg:147.12ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.8125, RMS->RMS:381.9278
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9721
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9749
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9675
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9748
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9761
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9807
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9741
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9798
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9766
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9844
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9858
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138.7161, RMS->RMS:0.9576
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6601
>>> Act Max Entries:
Embed:     6.1875
Block #0: 5.7188
Block #1: 5.2812
Block #2: 4.9062
Block #3: 4.5625
Block #4: 4.2500
Block #5: 4.0312
Block #6: 3.8125
Block #7: 3.6094
Block #8: 3.4219
Block #9: 3.2812
Block #10: 3.1250
Block #11: 2.9844
Logits:    21.7500
step:1875/6200 val_loss:5.6322 val_acc:0.1928 train_time:275898ms step_avg:147.15ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.6562, RMS->RMS:22.2811
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:699.5958
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:257.2288
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:277.0115
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:773.4349
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.7614
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3234.5225
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:614.2154
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:418.9457
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:300.6580
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:765.7223
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:121.1587
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3457.8735
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:568.3412
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:425.8505
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:326.1856
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:846.6387
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.5338
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3657.4180
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:449.2227
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:230.5193
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:298.0701
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:895.1371
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.4224
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3867.8052
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:528.4536
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:219.0396
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.0790
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:963.3992
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:134.4594
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4097.0459
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:695.6324
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:293.0136
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:394.8258
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:930.0637
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.9164
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4330.3286
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:430.6356
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:191.5409
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.9495
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:977.2609
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.2833
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4550.7646
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1182.7540
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:710.6697
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:463.2718
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1020.0081
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:166.7534
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4747.5078
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:645.5366
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:350.8696
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:425.7234
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1042.5537
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:178.9721
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4919.4043
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:790.8875
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:581.6545
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:414.5632
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1056.4421
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:189.2355
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5111.0664
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:714.4846
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:334.2301
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:460.1124
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:990.6417
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:201.5863
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5275.8477
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:407.3807
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:170.5166
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:485.9658
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1152.2974
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:206.8208
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5492.0381
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:226024.8438, RMS->RMS:1315.8590
step:1880/6200 train_time:276637ms step_avg:147.15ms
step:1890/6200 train_time:278135ms step_avg:147.16ms
step:1900/6200 train_time:279637ms step_avg:147.18ms
step:1910/6200 train_time:281137ms step_avg:147.19ms
step:1920/6200 train_time:282636ms step_avg:147.21ms
step:1930/6200 train_time:284136ms step_avg:147.22ms
step:1940/6200 train_time:285637ms step_avg:147.24ms
step:1950/6200 train_time:287137ms step_avg:147.25ms
step:1960/6200 train_time:288638ms step_avg:147.26ms
step:1970/6200 train_time:290137ms step_avg:147.28ms
step:1980/6200 train_time:291637ms step_avg:147.29ms
step:1990/6200 train_time:293137ms step_avg:147.31ms
step:2000/6200 train_time:294637ms step_avg:147.32ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.4375, RMS->RMS:380.7159
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9735
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9721
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9803
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9806
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9758
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9843
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9857
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138.6879, RMS->RMS:0.9588
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7070
Logits:    0.6622
>>> Act Max Entries:
Embed:     6.3750
Block #0: 5.9375
Block #1: 5.5000
Block #2: 5.1250
Block #3: 4.7812
Block #4: 4.4375
Block #5: 4.1250
Block #6: 3.8750
Block #7: 3.6562
Block #8: 3.4219
Block #9: 3.2656
Block #10: 3.1094
Block #11: 2.9688
Logits:    20.3750
step:2000/6200 val_loss:5.6246 val_acc:0.1935 train_time:294671ms step_avg:147.34ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:11.5625, RMS->RMS:39.9930
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:904.8479
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:299.3507
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:294.5498
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:785.9764
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:131.2708
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3167.2278
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1035.1699
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:676.8762
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:294.5186
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:788.4375
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:135.2593
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3365.2417
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:928.9814
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:690.2872
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:301.4191
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:864.7661
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.5104
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3530.0437
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:501.5879
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:272.2206
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:297.2953
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:916.7729
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:143.0569
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3712.2168
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:567.5643
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:261.8534
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:337.3388
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:940.1398
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:145.5748
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3906.3462
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:513.9832
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:232.4113
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:356.9936
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:899.1038
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.0688
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4125.3862
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:447.3998
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:250.0776
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:345.1061
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:963.6522
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.0013
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4350.0854
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:995.5906
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:610.6284
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:465.8656
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1008.9600
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:153.9142
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4569.8027
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:689.1862
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:310.5452
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:425.4048
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1023.8130
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:163.9765
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4756.9478
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:866.1778
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:603.7280
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:442.7928
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1082.3733
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.8744
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4956.6299
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:694.9491
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:479.9532
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:433.8599
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:994.5245
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:175.3003
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5147.2720
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:559.3059
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:181.5433
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:488.6736
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1132.9631
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:182.7773
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5341.8804
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:210961.3438, RMS->RMS:1233.2667
step:2010/6200 train_time:296152ms step_avg:147.34ms
step:2020/6200 train_time:297651ms step_avg:147.35ms
step:2030/6200 train_time:299152ms step_avg:147.37ms
step:2040/6200 train_time:300692ms step_avg:147.40ms
step:2050/6200 train_time:302191ms step_avg:147.41ms
step:2060/6200 train_time:303691ms step_avg:147.42ms
step:2070/6200 train_time:305192ms step_avg:147.44ms
step:2080/6200 train_time:306691ms step_avg:147.45ms
step:2090/6200 train_time:308191ms step_avg:147.46ms
step:2100/6200 train_time:309690ms step_avg:147.47ms
step:2110/6200 train_time:311192ms step_avg:147.48ms
step:2120/6200 train_time:312691ms step_avg:147.50ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:10.6875, RMS->RMS:379.9611
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9690
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9879
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9714
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9765
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9729
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9807
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9810
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9758
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9840
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:139.0477, RMS->RMS:0.9592
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6607
>>> Act Max Entries:
Embed:     8.4375
Block #0: 7.8438
Block #1: 7.2812
Block #2: 6.7812
Block #3: 6.3125
Block #4: 5.8750
Block #5: 5.4688
Block #6: 5.0938
Block #7: 4.7500
Block #8: 4.4375
Block #9: 4.1250
Block #10: 3.8438
Block #11: 3.5938
Logits:    19.7500
step:2125/6200 val_loss:5.6238 val_acc:0.1941 train_time:313476ms step_avg:147.52ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:13.0625, RMS->RMS:45.0101
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1061.7069
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:380.8943
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:319.4185
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:788.9023
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.7743
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3230.2478
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1135.4562
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:931.4033
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:406.9728
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:905.9327
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:116.3067
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3517.1248
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1425.0984
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1257.6404
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:327.8338
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:903.7053
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:127.3124
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3691.4922
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1065.6682
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:670.8229
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:325.6048
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:966.1135
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:137.7549
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3853.3870
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:803.5045
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:452.8692
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:367.6087
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:944.1844
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.3106
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4001.7610
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:597.7031
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:212.8950
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:367.8876
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:937.5244
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.7904
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4185.6904
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:334.6991
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:135.0257
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:399.5648
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1001.7862
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.0614
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4375.2817
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:391.6716
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:157.8806
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:411.2200
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1032.7982
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:161.3686
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4570.3174
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:530.2079
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:220.2363
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:434.6758
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1047.7943
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:174.0879
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4750.7363
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1105.2181
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:533.9146
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:505.3913
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1131.1152
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:184.8497
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4897.2222
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:444.6922
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:276.1951
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:433.8243
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1042.3955
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:186.1991
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5104.4468
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:462.8780
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:227.6937
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:474.4561
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1116.2058
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:198.5929
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5287.6567
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:167110.5938, RMS->RMS:1007.1464
step:2130/6200 train_time:314216ms step_avg:147.52ms
step:2140/6200 train_time:315717ms step_avg:147.53ms
step:2150/6200 train_time:317219ms step_avg:147.54ms
step:2160/6200 train_time:318717ms step_avg:147.55ms
step:2170/6200 train_time:320218ms step_avg:147.57ms
step:2180/6200 train_time:321719ms step_avg:147.58ms
step:2190/6200 train_time:323218ms step_avg:147.59ms
step:2200/6200 train_time:324717ms step_avg:147.60ms
step:2210/6200 train_time:326219ms step_avg:147.61ms
step:2220/6200 train_time:327719ms step_avg:147.62ms
step:2230/6200 train_time:329218ms step_avg:147.63ms
step:2240/6200 train_time:330718ms step_avg:147.64ms
step:2250/6200 train_time:332218ms step_avg:147.65ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.0625, RMS->RMS:382.1285
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9796
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9731
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9786
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9782
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9745
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9819
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:138.6907, RMS->RMS:0.9590
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6583
>>> Act Max Entries:
Embed:     7.1250
Block #0: 6.6250
Block #1: 6.1562
Block #2: 5.7188
Block #3: 5.3438
Block #4: 4.9688
Block #5: 4.6250
Block #6: 4.3125
Block #7: 4.0000
Block #8: 3.7344
Block #9: 3.4688
Block #10: 3.2344
Block #11: 3.0156
Logits:    21.0000
step:2250/6200 val_loss:5.6276 val_acc:0.1939 train_time:332252ms step_avg:147.67ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3125, RMS->RMS:28.8937
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1040.7588
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:377.7813
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:290.9449
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:791.8928
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.0307
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3230.3975
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1096.6371
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:801.6556
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.4087
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:850.8579
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:123.0815
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3479.1687
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1111.5369
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:970.2203
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:301.3814
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:871.2256
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.3040
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3659.2163
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:830.3656
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:495.4278
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:318.0940
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:924.2884
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:132.3369
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3848.1802
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:747.0292
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:329.9052
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:343.0663
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:964.7398
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:137.1583
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4044.6111
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:351.5704
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:211.4294
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:361.5075
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:906.8060
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.4937
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4280.1743
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:472.3483
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:310.9389
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:356.7122
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1010.5428
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:134.4688
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4517.2505
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:876.3193
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:542.2974
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:452.2079
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1029.4421
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:144.4646
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4744.6587
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:628.5759
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:301.0836
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:463.4904
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1056.4467
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.6231
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4945.0356
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1008.0688
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:845.3231
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:472.9336
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1118.5265
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:165.4645
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5142.7373
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:695.2354
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:798.7813
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:458.1398
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1028.5394
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:173.9479
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5357.6240
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:571.3441
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:690.1056
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:509.7149
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1147.3154
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:181.6979
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5577.3789
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:253979.4531, RMS->RMS:1342.5807
step:2260/6200 train_time:333740ms step_avg:147.67ms
step:2270/6200 train_time:335241ms step_avg:147.68ms
step:2280/6200 train_time:336740ms step_avg:147.69ms
step:2290/6200 train_time:338282ms step_avg:147.72ms
step:2300/6200 train_time:339787ms step_avg:147.73ms
step:2310/6200 train_time:341307ms step_avg:147.75ms
step:2320/6200 train_time:342823ms step_avg:147.77ms
step:2330/6200 train_time:344345ms step_avg:147.79ms
step:2340/6200 train_time:345863ms step_avg:147.80ms
step:2350/6200 train_time:347379ms step_avg:147.82ms
step:2360/6200 train_time:348897ms step_avg:147.84ms
step:2370/6200 train_time:350415ms step_avg:147.85ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:10.1875, RMS->RMS:382.1902
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9648
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9747
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9771
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9804
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9804
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9848
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9811
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9858
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9849
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9841
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9859
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9848
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:139.1567, RMS->RMS:0.9591
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8750
Block #5: 0.8516
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7305
Block #11: 0.7070
Logits:    0.6621
>>> Act Max Entries:
Embed:     6.6875
Block #0: 6.2188
Block #1: 5.7812
Block #2: 5.3750
Block #3: 5.0000
Block #4: 4.6562
Block #5: 4.3438
Block #6: 4.0312
Block #7: 3.7812
Block #8: 3.5469
Block #9: 3.3281
Block #10: 3.1406
Block #11: 2.9531
Logits:    20.5000
step:2375/6200 val_loss:5.6307 val_acc:0.1929 train_time:351208ms step_avg:147.88ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.4375, RMS->RMS:32.6729
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1011.4651
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:337.0938
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:378.3386
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:973.5353
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:183.1004
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3982.7207
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:955.3207
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:725.1511
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:431.3695
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:973.5051
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.5396
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4199.8633
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1356.8997
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1190.4739
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:454.7146
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1078.0061
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:192.9448
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4368.6738
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1278.8138
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:752.5342
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.1985
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1090.3463
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:202.8898
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4518.7734
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1050.9009
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:484.4699
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:487.5252
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1111.7279
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:207.4825
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4655.6738
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:568.5593
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:228.4207
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:451.0171
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1044.0562
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:210.3488
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4805.9751
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:556.7712
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:184.2082
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:431.4120
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1106.4170
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:213.6055
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4949.5039
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:629.3306
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:270.5338
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:464.0388
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1138.9487
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:215.6346
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5123.6597
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:502.6046
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:208.3098
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:472.1840
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1115.4979
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:222.5447
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5276.7227
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:904.6339
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:348.1620
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:503.3464
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1158.3831
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:227.9906
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5411.9819
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:552.4694
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:188.9348
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:505.2821
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1085.5564
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:238.1950
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5567.0547
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:641.7117
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:277.8218
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:518.5081
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1187.2242
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:245.8349
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5707.7354
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:179736.5938, RMS->RMS:1108.7893
step:2380/6200 train_time:351956ms step_avg:147.88ms
step:2390/6200 train_time:353472ms step_avg:147.90ms
step:2400/6200 train_time:354993ms step_avg:147.91ms
step:2410/6200 train_time:356513ms step_avg:147.93ms
step:2420/6200 train_time:358034ms step_avg:147.95ms
step:2430/6200 train_time:359551ms step_avg:147.96ms
step:2440/6200 train_time:361070ms step_avg:147.98ms
step:2450/6200 train_time:362591ms step_avg:148.00ms
step:2460/6200 train_time:364108ms step_avg:148.01ms
step:2470/6200 train_time:365629ms step_avg:148.03ms
step:2480/6200 train_time:367148ms step_avg:148.04ms
step:2490/6200 train_time:368668ms step_avg:148.06ms
step:2500/6200 train_time:370191ms step_avg:148.08ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.6875, RMS->RMS:381.9931
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9641
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9742
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9758
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9748
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9761
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9731
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9804
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9741
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9819
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9771
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9846
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:139.9460, RMS->RMS:0.9597
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6599
>>> Act Max Entries:
Embed:     7.1875
Block #0: 6.6562
Block #1: 6.1562
Block #2: 5.7188
Block #3: 5.3125
Block #4: 4.9375
Block #5: 4.5938
Block #6: 4.3125
Block #7: 4.0625
Block #8: 3.8125
Block #9: 3.5781
Block #10: 3.3750
Block #11: 3.1719
Logits:    19.6250
step:2500/6200 val_loss:5.6230 val_acc:0.1942 train_time:370225ms step_avg:148.09ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.8125, RMS->RMS:30.5402
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:880.9066
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:325.3963
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:233.4644
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:655.9443
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:106.7581
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2759.5129
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:817.8423
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:609.7524
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:329.9745
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:754.2318
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:99.3712
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2960.5151
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1160.1223
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1038.7694
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:250.2346
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:726.8088
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:108.7320
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3084.9546
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1380.5586
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:811.6982
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:269.3993
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:760.7325
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.4817
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3196.9043
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:670.7947
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:314.8945
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:277.2202
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:758.2654
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.7238
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3308.3242
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:537.4368
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:184.8424
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:289.6137
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:739.3776
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:135.6814
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3423.4966
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:562.3525
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:210.5264
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:297.2220
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:784.5123
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.9431
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3522.6372
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:553.1024
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:288.6374
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:313.9360
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:796.2720
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:143.0564
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3651.4067
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:428.0824
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:195.9607
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:319.1237
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:822.6408
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.2863
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3772.0874
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:795.8973
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:388.3668
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:366.2858
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:852.6265
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.5398
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3884.3948
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:562.3284
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:312.3023
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:336.5801
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:784.2396
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:166.5809
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4010.6289
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:677.6550
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:316.0880
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:363.0060
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:837.8878
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:168.8628
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4114.8276
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:112097.8281, RMS->RMS:834.6630
step:2510/6200 train_time:371730ms step_avg:148.10ms
step:2520/6200 train_time:373250ms step_avg:148.11ms
step:2530/6200 train_time:374767ms step_avg:148.13ms
step:2540/6200 train_time:376285ms step_avg:148.14ms
step:2550/6200 train_time:377845ms step_avg:148.17ms
step:2560/6200 train_time:379364ms step_avg:148.19ms
step:2570/6200 train_time:380883ms step_avg:148.20ms
step:2580/6200 train_time:382402ms step_avg:148.22ms
step:2590/6200 train_time:383920ms step_avg:148.23ms
step:2600/6200 train_time:385442ms step_avg:148.25ms
step:2610/6200 train_time:386959ms step_avg:148.26ms
step:2620/6200 train_time:388477ms step_avg:148.27ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.2188, RMS->RMS:382.9532
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9884
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9879
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9749
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9820
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9741
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9811
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9822
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9760
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9848
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9861
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:139.9733, RMS->RMS:0.9601
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6611
>>> Act Max Entries:
Embed:     7.2812
Block #0: 6.7500
Block #1: 6.2500
Block #2: 5.8125
Block #3: 5.4062
Block #4: 5.0312
Block #5: 4.6875
Block #6: 4.3438
Block #7: 4.0312
Block #8: 3.7500
Block #9: 3.4844
Block #10: 3.2500
Block #11: 3.0469
Logits:    21.0000
step:2625/6200 val_loss:5.6176 val_acc:0.1941 train_time:389271ms step_avg:148.29ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.0000, RMS->RMS:27.7894
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:945.7495
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:368.6379
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:295.0126
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:705.9387
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:113.8946
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2885.5261
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1014.2767
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:934.5331
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:405.6388
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:822.5662
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:107.1724
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3139.1584
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1256.6049
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1267.0813
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:286.2193
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:797.8729
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.6450
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3272.4275
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1118.7561
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:601.6212
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:287.4929
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:844.7626
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.9535
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3401.4675
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:695.4578
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:340.7477
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:314.5486
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:828.5126
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.7685
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3532.2236
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:439.7349
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:164.3546
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:323.7593
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:821.3354
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:131.9080
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3701.5764
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:488.3567
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:187.2707
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.8483
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:890.1564
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.5489
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3875.1348
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:511.0147
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:263.6928
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:385.8174
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:907.1474
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.5235
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4063.2407
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:572.6152
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:313.7021
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:420.7224
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:936.4421
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:149.2658
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4217.7886
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:930.6642
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:476.7571
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:404.4490
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:997.5391
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:152.7355
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4361.7739
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:488.7498
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:307.3936
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:384.3335
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:925.4019
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:160.1005
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4550.5083
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:499.9997
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:199.6183
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:420.6335
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:982.9728
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:162.5633
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4717.0435
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:145225.4688, RMS->RMS:991.4768
step:2630/6200 train_time:390018ms step_avg:148.30ms
step:2640/6200 train_time:391536ms step_avg:148.31ms
step:2650/6200 train_time:393054ms step_avg:148.32ms
step:2660/6200 train_time:394573ms step_avg:148.34ms
step:2670/6200 train_time:396094ms step_avg:148.35ms
step:2680/6200 train_time:397608ms step_avg:148.36ms
step:2690/6200 train_time:399125ms step_avg:148.37ms
step:2700/6200 train_time:400645ms step_avg:148.39ms
step:2710/6200 train_time:402161ms step_avg:148.40ms
step:2720/6200 train_time:403682ms step_avg:148.41ms
step:2730/6200 train_time:405202ms step_avg:148.43ms
step:2740/6200 train_time:406721ms step_avg:148.44ms
step:2750/6200 train_time:408244ms step_avg:148.45ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.2812, RMS->RMS:380.5981
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9735
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9733
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9802
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9849
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9825
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9848
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:140.4200, RMS->RMS:0.9601
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6609
>>> Act Max Entries:
Embed:     6.4062
Block #0: 5.9375
Block #1: 5.5312
Block #2: 5.1875
Block #3: 4.8750
Block #4: 4.5938
Block #5: 4.3125
Block #6: 4.0625
Block #7: 3.8125
Block #8: 3.5781
Block #9: 3.3594
Block #10: 3.1562
Block #11: 2.9688
Logits:    21.1250
step:2750/6200 val_loss:5.6334 val_acc:0.1918 train_time:408278ms step_avg:148.46ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.7500, RMS->RMS:30.9079
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1538.7537
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:515.1802
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:475.6666
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:1239.9297
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:214.8495
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5151.8491
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1696.7393
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1379.6545
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:479.1526
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1347.4948
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:208.5099
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5528.7070
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2093.4568
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:2159.0076
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:660.0624
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1460.0636
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:230.8096
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5750.6367
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1510.0508
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:883.0947
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:504.2010
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1484.6284
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:244.7016
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5937.8677
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1593.4849
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:776.2630
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:665.6643
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1480.6608
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:255.0805
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6097.0581
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:735.9831
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:334.2207
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:607.8540
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1387.1204
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:256.1793
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6304.8647
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:644.6869
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:263.6778
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:597.7105
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1485.6727
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:259.9338
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6523.8389
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:733.4412
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:294.6233
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:617.3201
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1504.2153
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:269.6942
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6764.2173
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:893.1461
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:368.2434
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:662.8425
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1502.5553
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:282.7202
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6974.5161
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1582.8149
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:740.7454
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:634.9760
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1566.7124
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:287.7886
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:7151.7554
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:815.6266
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:510.9489
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:635.3562
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1458.7375
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:293.8181
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:7400.3301
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:777.5308
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:343.6498
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:696.7526
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1581.7837
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:302.5352
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:7612.9170
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:237263.1562, RMS->RMS:1378.4297
step:2760/6200 train_time:409784ms step_avg:148.47ms
step:2770/6200 train_time:411307ms step_avg:148.49ms
step:2780/6200 train_time:412828ms step_avg:148.50ms
step:2790/6200 train_time:414353ms step_avg:148.51ms
step:2800/6200 train_time:415917ms step_avg:148.54ms
step:2810/6200 train_time:417439ms step_avg:148.55ms
step:2820/6200 train_time:418965ms step_avg:148.57ms
step:2830/6200 train_time:420485ms step_avg:148.58ms
step:2840/6200 train_time:422009ms step_avg:148.59ms
step:2850/6200 train_time:423531ms step_avg:148.61ms
step:2860/6200 train_time:425058ms step_avg:148.62ms
step:2870/6200 train_time:426580ms step_avg:148.63ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3125, RMS->RMS:381.3522
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9876
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9761
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9812
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9803
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9763
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9825
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9772
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9849
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:140.5800, RMS->RMS:0.9607
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6620
>>> Act Max Entries:
Embed:     6.8125
Block #0: 6.3125
Block #1: 5.8438
Block #2: 5.4062
Block #3: 5.0000
Block #4: 4.6250
Block #5: 4.3438
Block #6: 4.0938
Block #7: 3.8438
Block #8: 3.6094
Block #9: 3.3906
Block #10: 3.1875
Block #11: 2.9844
Logits:    20.7500
step:2875/6200 val_loss:5.6248 val_acc:0.1928 train_time:427375ms step_avg:148.65ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4688, RMS->RMS:26.1859
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1207.9854
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:463.0670
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:349.3594
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:892.2550
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.9160
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3754.3962
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1323.3768
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1061.1498
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:426.2191
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:991.6997
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.1607
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4073.2288
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1401.7053
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1314.2538
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.4084
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:987.6526
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.1826
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4240.0464
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1080.1580
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:632.6010
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.0119
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1066.5839
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.0050
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4433.7686
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:954.5350
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:463.1068
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:401.7446
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1069.2224
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:163.5358
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4601.1948
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:500.3101
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:212.5494
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:417.2667
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1037.8928
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:165.7256
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4824.7402
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:518.5613
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:192.3964
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:428.7050
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1099.2472
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.6135
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5039.7803
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:952.8190
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:569.1523
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:475.8648
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1124.3726
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.9158
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5258.6338
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:785.1632
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:403.8755
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:504.5805
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1165.4926
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:194.4850
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5452.0493
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1135.2915
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:564.3899
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:513.9141
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1210.7426
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:202.7933
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5636.7891
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:610.6473
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:431.8392
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:482.4254
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1135.1124
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:204.0185
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5865.9292
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:580.4545
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:258.5851
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:524.9590
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1250.8712
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:213.8528
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:6090.7754
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:179919.9688, RMS->RMS:1284.1902
step:2880/6200 train_time:428123ms step_avg:148.65ms
step:2890/6200 train_time:429645ms step_avg:148.67ms
step:2900/6200 train_time:431167ms step_avg:148.68ms
step:2910/6200 train_time:432692ms step_avg:148.69ms
step:2920/6200 train_time:434216ms step_avg:148.70ms
step:2930/6200 train_time:435738ms step_avg:148.72ms
step:2940/6200 train_time:437259ms step_avg:148.73ms
step:2950/6200 train_time:438786ms step_avg:148.74ms
step:2960/6200 train_time:440307ms step_avg:148.75ms
step:2970/6200 train_time:441830ms step_avg:148.76ms
step:2980/6200 train_time:443354ms step_avg:148.78ms
step:2990/6200 train_time:444876ms step_avg:148.79ms
step:3000/6200 train_time:446399ms step_avg:148.80ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3750, RMS->RMS:382.2634
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9751
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9676
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9729
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9800
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9811
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9757
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9821
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9766
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9851
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9861
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:140.6543, RMS->RMS:0.9611
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6623
>>> Act Max Entries:
Embed:     6.5312
Block #0: 6.0938
Block #1: 5.6562
Block #2: 5.2812
Block #3: 4.9062
Block #4: 4.5625
Block #5: 4.2500
Block #6: 3.9688
Block #7: 3.7031
Block #8: 3.4688
Block #9: 3.2500
Block #10: 3.0469
Block #11: 2.8594
Logits:    20.3750
step:3000/6200 val_loss:5.6208 val_acc:0.1935 train_time:446432ms step_avg:148.81ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.6562, RMS->RMS:20.0051
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1276.1473
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:554.6395
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:347.8095
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:850.6896
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:143.5690
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3551.8291
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1564.7603
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1420.7178
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:443.3684
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:986.5699
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:131.2538
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3895.2041
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1726.6455
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1572.9421
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:338.5243
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:944.2001
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.5857
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4045.2271
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1576.8197
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:928.3745
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.2997
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1024.3904
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.3293
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4204.3755
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:966.0612
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:468.2109
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.5582
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:996.5210
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:164.2071
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4347.4180
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:624.8929
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:204.4591
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:395.3321
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:984.3291
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:171.8181
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4522.6055
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:500.4990
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:141.5619
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:420.7163
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1047.7109
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:176.2369
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4707.0283
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:478.9630
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:169.7984
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:423.3106
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1094.4490
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:182.5990
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4912.1230
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:606.2332
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:273.6901
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:466.0673
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1118.9906
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:195.2151
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5092.1831
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1289.7195
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:629.1365
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:533.1620
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1150.1342
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:201.6085
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5226.1211
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:630.1062
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:362.1933
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:461.2350
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1073.4886
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:206.5005
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5424.8276
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:656.6398
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:301.3927
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:490.7973
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1164.4587
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:213.5016
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5612.7861
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:132704.7344, RMS->RMS:1110.9109
step:3010/6200 train_time:447940ms step_avg:148.82ms
step:3020/6200 train_time:449461ms step_avg:148.83ms
step:3030/6200 train_time:450982ms step_avg:148.84ms
step:3040/6200 train_time:452507ms step_avg:148.85ms
step:3050/6200 train_time:454071ms step_avg:148.88ms
step:3060/6200 train_time:455595ms step_avg:148.89ms
step:3070/6200 train_time:457118ms step_avg:148.90ms
step:3080/6200 train_time:458641ms step_avg:148.91ms
step:3090/6200 train_time:460166ms step_avg:148.92ms
step:3100/6200 train_time:461690ms step_avg:148.93ms
step:3110/6200 train_time:463212ms step_avg:148.94ms
step:3120/6200 train_time:464735ms step_avg:148.95ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.9375, RMS->RMS:380.7207
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9887
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9785
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9760
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9789
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9800
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9816
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9845
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:141.2302, RMS->RMS:0.9613
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8750
Block #5: 0.8516
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7773
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7070
Logits:    0.6634
>>> Act Max Entries:
Embed:     7.1250
Block #0: 6.6562
Block #1: 6.1875
Block #2: 5.7500
Block #3: 5.3750
Block #4: 5.0000
Block #5: 4.6562
Block #6: 4.3438
Block #7: 4.0625
Block #8: 3.7969
Block #9: 3.5469
Block #10: 3.3125
Block #11: 3.0938
Logits:    20.1250
step:3125/6200 val_loss:5.6205 val_acc:0.1945 train_time:465532ms step_avg:148.97ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:12.1250, RMS->RMS:41.6935
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:906.0626
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:343.5733
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:310.4767
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:778.9880
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.0533
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3261.3557
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1006.1785
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:675.2092
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:376.5218
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:852.7589
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:133.1321
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3481.9707
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1279.7947
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:953.8674
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:318.5196
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:833.9175
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.9291
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3615.5933
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1136.5989
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:631.8647
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:330.7275
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:903.1680
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:154.4453
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3798.6123
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:809.7928
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:400.1483
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:335.1729
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:882.1403
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.4344
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3944.5581
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:389.4124
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:195.1659
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.6602
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:875.7463
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.2601
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4118.6904
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:324.6830
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:127.9640
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:361.6138
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:926.6870
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:162.0756
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4292.4087
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:732.5958
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:334.0197
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:377.2339
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:955.2170
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:161.4476
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4467.7427
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:376.3471
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:156.4513
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:406.3283
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:997.7705
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:168.7647
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4648.4478
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:873.3474
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:481.3580
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:465.2893
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1039.0415
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:179.2022
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4826.3262
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:298.5475
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:322.9702
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:428.0958
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:964.9313
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:174.3961
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5029.1367
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:406.6513
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:321.4732
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:432.6942
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1041.4591
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:192.1834
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5222.3379
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:205502.4062, RMS->RMS:1170.9348
step:3130/6200 train_time:466282ms step_avg:148.97ms
step:3140/6200 train_time:467802ms step_avg:148.98ms
step:3150/6200 train_time:469322ms step_avg:148.99ms
step:3160/6200 train_time:470842ms step_avg:149.00ms
step:3170/6200 train_time:472366ms step_avg:149.01ms
step:3180/6200 train_time:473891ms step_avg:149.02ms
step:3190/6200 train_time:475414ms step_avg:149.03ms
step:3200/6200 train_time:476939ms step_avg:149.04ms
step:3210/6200 train_time:478463ms step_avg:149.05ms
step:3220/6200 train_time:479996ms step_avg:149.07ms
step:3230/6200 train_time:481537ms step_avg:149.08ms
step:3240/6200 train_time:483076ms step_avg:149.10ms
step:3250/6200 train_time:484618ms step_avg:149.11ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.3125, RMS->RMS:381.6984
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9669
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9770
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9868
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9697
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9733
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9773
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9729
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9796
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9745
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9813
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9828
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9770
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9850
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9852
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:141.5137, RMS->RMS:0.9607
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7070
Logits:    0.6639
>>> Act Max Entries:
Embed:     7.8750
Block #0: 7.3438
Block #1: 6.8438
Block #2: 6.3750
Block #3: 5.9375
Block #4: 5.5312
Block #5: 5.1562
Block #6: 4.8125
Block #7: 4.4688
Block #8: 4.1562
Block #9: 3.8750
Block #10: 3.6094
Block #11: 3.3594
Logits:    20.1250
step:3250/6200 val_loss:5.6165 val_acc:0.1940 train_time:484652ms step_avg:149.12ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.4375, RMS->RMS:32.6889
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:807.6160
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:332.5605
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:372.2530
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:912.2476
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:160.8833
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3698.8943
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1064.0128
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:866.6854
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:381.3246
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:960.8143
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:160.4767
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3919.6206
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1287.8268
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1189.5443
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:449.1110
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1050.8427
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.9740
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4088.2241
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1284.9000
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:770.8170
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:387.3204
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1077.9156
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:183.5031
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4205.1768
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:803.8289
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:275.5874
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:409.9354
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1045.9357
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:189.1491
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4337.1328
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:595.7763
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:198.8192
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:431.2738
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1010.5459
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:195.9205
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4464.4663
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:558.4304
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:177.7115
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:414.8640
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1051.4473
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:192.8035
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4609.4995
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:745.2033
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:409.8507
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:457.4482
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1072.5271
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:204.3891
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4752.4561
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:509.1758
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:223.4118
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:467.6528
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1094.9138
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:209.3533
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4890.7568
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:900.3256
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:533.5706
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:460.5719
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1134.0610
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:211.3771
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5037.1533
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:659.8944
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:326.5019
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:444.7964
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1042.0756
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:222.0035
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5169.1245
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:803.2417
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:275.8065
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:477.7089
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1115.1892
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:221.2346
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5278.4033
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:159096.9375, RMS->RMS:1048.9524
step:3260/6200 train_time:486179ms step_avg:149.13ms
step:3270/6200 train_time:487714ms step_avg:149.15ms
step:3280/6200 train_time:489252ms step_avg:149.16ms
step:3290/6200 train_time:490792ms step_avg:149.18ms
step:3300/6200 train_time:492333ms step_avg:149.19ms
step:3310/6200 train_time:493904ms step_avg:149.22ms
step:3320/6200 train_time:495442ms step_avg:149.23ms
step:3330/6200 train_time:496981ms step_avg:149.24ms
step:3340/6200 train_time:498522ms step_avg:149.26ms
step:3350/6200 train_time:500064ms step_avg:149.27ms
step:3360/6200 train_time:501606ms step_avg:149.29ms
step:3370/6200 train_time:503145ms step_avg:149.30ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.8125, RMS->RMS:381.7601
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9886
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9756
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9723
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9814
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9825
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9756
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9851
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:141.9127, RMS->RMS:0.9614
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6619
>>> Act Max Entries:
Embed:     6.9062
Block #0: 6.4062
Block #1: 5.9375
Block #2: 5.5312
Block #3: 5.1562
Block #4: 4.7812
Block #5: 4.4375
Block #6: 4.1250
Block #7: 3.8281
Block #8: 3.5625
Block #9: 3.3281
Block #10: 3.1094
Block #11: 2.9062
Logits:    20.2500
step:3375/6200 val_loss:5.6234 val_acc:0.1937 train_time:503946ms step_avg:149.32ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.3125, RMS->RMS:18.2712
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1451.7869
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:581.8741
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:336.8321
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:855.0410
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.2113
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3686.6887
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1704.9089
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1491.9252
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:427.5791
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1022.7628
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.5349
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4045.0276
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1752.6960
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1635.9883
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:362.7365
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:969.4830
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.9028
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4221.7227
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1838.0695
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1084.8141
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.3272
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1037.1556
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:163.3309
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4368.0771
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:989.3387
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:465.4148
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:397.3328
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1051.2128
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:177.1766
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4512.2441
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:766.7238
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:258.9731
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:393.6241
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:996.3415
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:177.6190
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4677.3472
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:647.2615
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:208.6087
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:401.4194
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1073.5294
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.0705
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4865.0229
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:496.7366
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:225.2968
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:427.2626
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1094.5250
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:191.1244
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5076.1914
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:786.9233
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:374.5907
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:463.8760
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1112.3318
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:197.5094
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5253.7368
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1245.6288
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:676.5588
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:492.9142
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1145.7974
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:202.9363
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5394.3643
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:835.0203
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:390.2839
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:460.9291
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1054.1282
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:212.1551
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5595.6968
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:723.5544
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:218.5092
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:507.5006
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1158.9086
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:221.9663
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5768.4673
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:215063.0156, RMS->RMS:1241.7716
step:3380/6200 train_time:504704ms step_avg:149.32ms
step:3390/6200 train_time:506243ms step_avg:149.33ms
step:3400/6200 train_time:507784ms step_avg:149.35ms
step:3410/6200 train_time:509323ms step_avg:149.36ms
step:3420/6200 train_time:510861ms step_avg:149.37ms
step:3430/6200 train_time:512401ms step_avg:149.39ms
step:3440/6200 train_time:513939ms step_avg:149.40ms
step:3450/6200 train_time:515477ms step_avg:149.41ms
step:3460/6200 train_time:517014ms step_avg:149.43ms
step:3470/6200 train_time:518551ms step_avg:149.44ms
step:3480/6200 train_time:520091ms step_avg:149.45ms
step:3490/6200 train_time:521631ms step_avg:149.46ms
step:3500/6200 train_time:523168ms step_avg:149.48ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.9375, RMS->RMS:381.0564
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9884
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9878
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9740
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9728
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9802
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9764
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9829
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9777
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9848
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:142.0045, RMS->RMS:0.9615
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7070
Logits:    0.6629
>>> Act Max Entries:
Embed:     7.0312
Block #0: 6.5000
Block #1: 6.0000
Block #2: 5.5625
Block #3: 5.1562
Block #4: 4.7812
Block #5: 4.4375
Block #6: 4.1250
Block #7: 3.8281
Block #8: 3.5625
Block #9: 3.3125
Block #10: 3.0781
Block #11: 2.8750
Logits:    20.5000
step:3500/6200 val_loss:5.6227 val_acc:0.1929 train_time:523202ms step_avg:149.49ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.9375, RMS->RMS:31.4801
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:849.6696
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:304.5730
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:310.9450
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:843.5643
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.9369
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3560.4775
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1096.6445
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:682.9454
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.0019
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:905.3243
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:131.4000
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3793.9270
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:926.3522
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:604.2994
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.7646
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:978.2300
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.2202
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3957.7859
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:936.3237
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:649.6302
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:361.5167
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1014.7792
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:153.8164
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4107.5449
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:697.7328
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:300.4169
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:437.1180
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1023.0284
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.4966
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4277.0190
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:376.3999
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:160.8908
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.8400
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:972.7943
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:165.7249
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4458.9785
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:468.4746
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:167.9772
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:411.9147
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1027.3895
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:171.5097
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4642.1255
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:878.5509
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:458.9797
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:431.7783
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1045.2280
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:171.0289
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4825.5869
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:525.4778
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:215.0506
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:455.6283
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1073.3756
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:186.9846
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5005.6216
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:633.7328
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:466.5751
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:444.0466
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1125.3394
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:186.8168
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5208.3618
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:556.6794
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:346.8415
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:463.7377
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1060.9944
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:183.7880
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5399.5654
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:587.9578
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:324.5098
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:494.6348
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1140.8318
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:192.0757
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5623.8477
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:224016.8906, RMS->RMS:1250.6050
step:3510/6200 train_time:524727ms step_avg:149.50ms
step:3520/6200 train_time:526264ms step_avg:149.51ms
step:3530/6200 train_time:527802ms step_avg:149.52ms
step:3540/6200 train_time:529340ms step_avg:149.53ms
step:3550/6200 train_time:530879ms step_avg:149.54ms
step:3560/6200 train_time:532448ms step_avg:149.56ms
step:3570/6200 train_time:533985ms step_avg:149.58ms
step:3580/6200 train_time:535525ms step_avg:149.59ms
step:3590/6200 train_time:537065ms step_avg:149.60ms
step:3600/6200 train_time:538605ms step_avg:149.61ms
step:3610/6200 train_time:540144ms step_avg:149.62ms
step:3620/6200 train_time:541685ms step_avg:149.64ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.5625, RMS->RMS:381.2076
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9880
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9765
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9756
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9725
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9712
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9766
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9799
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9738
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9805
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9759
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9830
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9763
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9853
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9855
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:142.1657, RMS->RMS:0.9624
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6634
>>> Act Max Entries:
Embed:     6.4375
Block #0: 5.9688
Block #1: 5.5312
Block #2: 5.1250
Block #3: 4.7500
Block #4: 4.4062
Block #5: 4.0938
Block #6: 3.7969
Block #7: 3.5312
Block #8: 3.2969
Block #9: 3.1250
Block #10: 2.9531
Block #11: 2.7969
Logits:    20.5000
step:3625/6200 val_loss:5.6198 val_acc:0.1926 train_time:542486ms step_avg:149.65ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:11.1875, RMS->RMS:38.6235
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:988.0291
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:371.5777
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:383.1122
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:956.1672
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:164.6286
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3921.7290
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1027.2384
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:851.6593
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.2369
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:972.1287
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.5664
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4131.4712
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1384.3312
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1319.8428
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:434.1318
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1062.3583
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.4359
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4345.4478
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1525.5009
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:934.5959
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.8337
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1092.3193
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:198.0860
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4470.5737
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:917.2928
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:324.0732
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:417.6563
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1114.8105
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:202.4764
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4626.9966
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:685.8728
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:229.8368
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:419.6071
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1035.0188
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:204.7723
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4767.7285
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:683.9819
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:204.5379
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:406.4038
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1083.6449
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:203.7027
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4917.1743
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:507.0529
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:196.8521
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:455.6136
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1133.6235
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:208.7670
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5089.0615
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:517.8637
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:269.7921
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:482.4181
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1156.5885
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:214.9115
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5259.4941
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1157.5468
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:784.1889
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:466.9522
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1170.7247
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:221.1454
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5417.7666
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:724.0927
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:459.4725
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:477.0374
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1088.8489
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:225.8596
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5567.5737
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:735.2535
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:236.1120
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:529.5120
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1194.3008
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:227.0810
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5733.8174
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:232311.3594, RMS->RMS:1287.2600
step:3630/6200 train_time:543242ms step_avg:149.65ms
step:3640/6200 train_time:544776ms step_avg:149.66ms
step:3650/6200 train_time:546313ms step_avg:149.67ms
step:3660/6200 train_time:547855ms step_avg:149.69ms
step:3670/6200 train_time:549396ms step_avg:149.70ms
step:3680/6200 train_time:550935ms step_avg:149.71ms
step:3690/6200 train_time:552480ms step_avg:149.72ms
step:3700/6200 train_time:554019ms step_avg:149.73ms
step:3710/6200 train_time:555562ms step_avg:149.75ms
step:3720/6200 train_time:557107ms step_avg:149.76ms
step:3730/6200 train_time:559256ms step_avg:149.93ms
step:3740/6200 train_time:560799ms step_avg:149.95ms
step:3750/6200 train_time:562343ms step_avg:149.96ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9375, RMS->RMS:380.4935
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9896
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9893
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9876
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9689
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9748
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9892
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9876
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9747
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9756
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9766
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9800
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9808
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9775
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9829
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9770
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9853
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9863
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:141.5784, RMS->RMS:0.9630
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9258
Block #3: 0.8984
Block #4: 0.8750
Block #5: 0.8516
Block #6: 0.8281
Block #7: 0.8008
Block #8: 0.7773
Block #9: 0.7539
Block #10: 0.7305
Block #11: 0.7070
Logits:    0.6660
>>> Act Max Entries:
Embed:     7.8438
Block #0: 7.2500
Block #1: 6.6875
Block #2: 6.1875
Block #3: 5.7188
Block #4: 5.3125
Block #5: 4.9062
Block #6: 4.5312
Block #7: 4.1875
Block #8: 3.8750
Block #9: 3.6094
Block #10: 3.3750
Block #11: 3.1406
Logits:    19.1250
step:3750/6200 val_loss:5.6125 val_acc:0.1935 train_time:562377ms step_avg:149.97ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:12.0000, RMS->RMS:41.4060
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:772.4424
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:345.7717
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:240.0705
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:663.0274
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:107.5885
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2836.4924
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:427.0069
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:269.4036
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:247.9513
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:751.9860
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:111.4586
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3025.6985
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:827.9747
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:439.5893
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:273.9032
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:799.4429
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.1197
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3236.3433
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:867.9609
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:663.7741
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:276.7361
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:843.5631
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:124.7287
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3411.9490
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:566.5580
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:252.0922
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:300.0476
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:852.5577
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.4860
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3610.0857
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:482.9387
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:305.4725
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:312.9824
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:834.5897
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.1360
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3805.1558
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:530.5966
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:265.7215
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:328.9390
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:896.8096
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:118.9745
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3996.9199
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:437.8244
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:158.1889
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.4595
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:918.2737
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.7594
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4181.2598
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:434.3756
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:226.4210
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:386.2146
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:953.3777
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:137.3525
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4360.4819
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:884.1619
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:590.7410
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:392.0341
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1006.0850
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.0213
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4549.7119
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:741.2974
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:382.8324
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:386.1176
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:937.1255
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:154.1084
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4729.5352
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:504.2299
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:205.0770
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:442.7185
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1015.3154
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:151.8926
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4917.7627
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:180031.6094, RMS->RMS:1189.2262
step:3760/6200 train_time:563907ms step_avg:149.98ms
step:3770/6200 train_time:565450ms step_avg:149.99ms
step:3780/6200 train_time:566992ms step_avg:150.00ms
step:3790/6200 train_time:568533ms step_avg:150.01ms
step:3800/6200 train_time:570076ms step_avg:150.02ms
step:3810/6200 train_time:571620ms step_avg:150.03ms
step:3820/6200 train_time:573208ms step_avg:150.05ms
step:3830/6200 train_time:574749ms step_avg:150.06ms
step:3840/6200 train_time:576295ms step_avg:150.08ms
step:3850/6200 train_time:577838ms step_avg:150.09ms
step:3860/6200 train_time:579383ms step_avg:150.10ms
step:3870/6200 train_time:580928ms step_avg:150.11ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4062, RMS->RMS:381.6544
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9751
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9787
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9723
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9805
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9824
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9770
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9852
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9857
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:142.6224, RMS->RMS:0.9669
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6666
>>> Act Max Entries:
Embed:     9.2500
Block #0: 8.5625
Block #1: 7.9062
Block #2: 7.3125
Block #3: 6.7500
Block #4: 6.2500
Block #5: 5.7812
Block #6: 5.3438
Block #7: 4.9375
Block #8: 4.5625
Block #9: 4.2188
Block #10: 3.9062
Block #11: 3.6250
Logits:    20.3750
step:3875/6200 val_loss:5.6092 val_acc:0.1941 train_time:581735ms step_avg:150.13ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:10.8125, RMS->RMS:37.6778
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:933.1970
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:418.9761
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:249.9294
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:652.6934
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:106.0619
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2788.6904
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:530.7418
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:330.6005
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:287.0669
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:758.9036
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:113.5847
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2939.3306
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1131.1647
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:945.6804
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:266.9434
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:743.4590
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.1771
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3090.4919
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1169.1947
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:850.3691
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:263.7049
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:784.3564
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:135.5851
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3203.2273
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:497.8531
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:209.6160
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:282.8004
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:816.6371
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:132.2241
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3341.1855
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:626.2654
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:269.9915
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:295.0462
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:757.5302
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:134.0340
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3477.1411
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:604.6934
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:266.7509
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:295.4218
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:800.8281
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:135.6479
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3614.0454
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:596.9358
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:212.5520
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:326.5784
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:818.8967
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.1056
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3747.8435
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:515.0746
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:289.2485
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:368.6420
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:852.7516
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.8497
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3886.5356
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:808.1782
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:562.8502
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:354.6311
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:881.4177
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:157.4269
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4019.6616
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:531.2076
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:309.4827
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:349.7974
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:833.2012
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:166.2681
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4150.7979
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:578.5902
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:227.5263
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:391.2168
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:908.2862
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:166.2871
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4282.8115
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:129051.0000, RMS->RMS:957.5424
step:3880/6200 train_time:582498ms step_avg:150.13ms
step:3890/6200 train_time:584037ms step_avg:150.14ms
step:3900/6200 train_time:585579ms step_avg:150.15ms
step:3910/6200 train_time:587124ms step_avg:150.16ms
step:3920/6200 train_time:588667ms step_avg:150.17ms
step:3930/6200 train_time:590207ms step_avg:150.18ms
step:3940/6200 train_time:591752ms step_avg:150.19ms
step:3950/6200 train_time:593293ms step_avg:150.20ms
step:3960/6200 train_time:594837ms step_avg:150.21ms
step:3970/6200 train_time:596377ms step_avg:150.22ms
step:3980/6200 train_time:597925ms step_avg:150.23ms
step:3990/6200 train_time:599470ms step_avg:150.24ms
step:4000/6200 train_time:601017ms step_avg:150.25ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.6562, RMS->RMS:380.7865
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9887
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9701
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9869
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9676
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9863
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9765
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9782
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9705
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9793
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9825
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9855
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:144.0437, RMS->RMS:0.9694
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6675
>>> Act Max Entries:
Embed:     7.5000
Block #0: 6.9375
Block #1: 6.4062
Block #2: 5.9062
Block #3: 5.4688
Block #4: 5.0625
Block #5: 4.6875
Block #6: 4.3438
Block #7: 4.0312
Block #8: 3.7344
Block #9: 3.4531
Block #10: 3.2031
Block #11: 2.9844
Logits:    20.2500
step:4000/6200 val_loss:5.6057 val_acc:0.1943 train_time:601051ms step_avg:150.26ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:13.3750, RMS->RMS:45.7895
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1250.8616
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:547.1184
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:347.2812
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:874.7613
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.0000
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3647.1980
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1488.6185
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1109.8676
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.8882
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:970.3356
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.9672
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3912.2812
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1523.9888
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1186.5524
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:327.6357
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:944.6444
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:163.3141
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4059.3379
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1248.2075
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:709.6227
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:357.0437
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1004.8818
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:179.6867
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4235.1450
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:977.8663
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:417.0134
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.2255
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:999.3062
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:176.4648
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4383.1387
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:535.7638
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:227.0646
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:379.9914
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:970.6176
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:175.4583
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4566.0239
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:578.6497
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:361.7460
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:374.5535
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1044.7991
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.3522
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4753.0557
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:767.6959
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:371.1581
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:443.9672
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1090.1226
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:185.4265
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4934.1245
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:600.4355
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:275.0878
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:488.6957
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1119.0797
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:196.6866
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5117.2134
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1130.0724
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:679.2100
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:481.0498
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1152.7252
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:206.3470
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5265.9106
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:798.3573
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:546.3516
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:448.9568
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1046.6400
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:209.2637
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5425.5220
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:448.9169
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:242.4357
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:498.7594
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1152.4927
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:213.0784
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5642.6528
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:220533.6094, RMS->RMS:1280.5750
step:4010/6200 train_time:602581ms step_avg:150.27ms
step:4020/6200 train_time:604125ms step_avg:150.28ms
step:4030/6200 train_time:605667ms step_avg:150.29ms
step:4040/6200 train_time:607211ms step_avg:150.30ms
step:4050/6200 train_time:608751ms step_avg:150.31ms
step:4060/6200 train_time:610294ms step_avg:150.32ms
step:4070/6200 train_time:611873ms step_avg:150.34ms
step:4080/6200 train_time:613420ms step_avg:150.35ms
step:4090/6200 train_time:614965ms step_avg:150.36ms
step:4100/6200 train_time:616507ms step_avg:150.37ms
step:4110/6200 train_time:618051ms step_avg:150.38ms
step:4120/6200 train_time:619593ms step_avg:150.39ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.5625, RMS->RMS:379.9849
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9638
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9640
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9735
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9708
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9797
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9795
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9749
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9833
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9850
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:144.6112, RMS->RMS:0.9727
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6677
>>> Act Max Entries:
Embed:     7.2188
Block #0: 6.6875
Block #1: 6.1875
Block #2: 5.7500
Block #3: 5.3438
Block #4: 5.0000
Block #5: 4.6562
Block #6: 4.3438
Block #7: 4.0625
Block #8: 3.7969
Block #9: 3.5469
Block #10: 3.3125
Block #11: 3.0938
Logits:    20.2500
step:4125/6200 val_loss:5.5842 val_acc:0.1968 train_time:620400ms step_avg:150.40ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.0938, RMS->RMS:24.8171
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:822.3931
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:297.2317
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:233.4998
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:565.9461
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:106.1189
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2380.5967
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:624.8976
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:487.4716
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:290.5292
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:632.0778
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:107.7154
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2502.4309
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1021.1481
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:902.5098
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:238.2665
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:613.1707
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:115.7956
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2598.2764
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:790.6877
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:540.0523
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:252.7169
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:652.4435
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:118.6168
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2697.1646
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:580.4199
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:249.9545
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:244.3967
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:641.2982
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:113.3065
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2793.8042
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:550.3937
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:198.4314
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:244.1107
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:610.3040
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:116.8049
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2906.1270
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:514.7432
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:202.4446
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:249.1997
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:671.9537
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.6664
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3024.9128
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:898.6392
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:460.5007
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:330.5306
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:686.9034
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.5447
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3132.9155
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:516.0798
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:283.2895
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:321.0041
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:699.5606
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.4367
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3231.7705
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:668.0964
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:469.0989
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:298.0366
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:711.4005
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:144.2123
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3320.9700
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:530.3860
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:327.8486
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:284.7090
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:653.5772
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:146.7765
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3419.9631
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:461.3049
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:129.0190
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:313.3568
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:708.0866
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:148.8442
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3514.1836
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:110906.8359, RMS->RMS:726.2110
step:4130/6200 train_time:621159ms step_avg:150.40ms
step:4140/6200 train_time:622711ms step_avg:150.41ms
step:4150/6200 train_time:624269ms step_avg:150.43ms
step:4160/6200 train_time:625828ms step_avg:150.44ms
step:4170/6200 train_time:627384ms step_avg:150.45ms
step:4180/6200 train_time:628940ms step_avg:150.46ms
step:4190/6200 train_time:630499ms step_avg:150.48ms
step:4200/6200 train_time:632055ms step_avg:150.49ms
step:4210/6200 train_time:633612ms step_avg:150.50ms
step:4220/6200 train_time:635173ms step_avg:150.51ms
step:4230/6200 train_time:636737ms step_avg:150.53ms
step:4240/6200 train_time:638291ms step_avg:150.54ms
step:4250/6200 train_time:639850ms step_avg:150.55ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9375, RMS->RMS:379.9584
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9633
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9638
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9711
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9872
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9750
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9658
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9727
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9702
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9704
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9775
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9710
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9791
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9859
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9741
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9814
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9828
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9850
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:145.2005, RMS->RMS:0.9759
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7539
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6718
>>> Act Max Entries:
Embed:     8.1875
Block #0: 7.5938
Block #1: 7.0312
Block #2: 6.5000
Block #3: 6.0000
Block #4: 5.5625
Block #5: 5.1562
Block #6: 4.7812
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8281
Block #10: 3.5469
Block #11: 3.2812
Logits:    20.5000
step:4250/6200 val_loss:5.5820 val_acc:0.1963 train_time:639884ms step_avg:150.56ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.8125, RMS->RMS:30.5495
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1095.5454
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:361.8395
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:271.0990
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:719.7349
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.5701
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3070.6431
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1252.4955
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:950.2880
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:364.5106
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:853.8687
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:116.5799
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3327.4072
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1384.8975
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1301.4944
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:269.0896
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:790.9580
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:128.9903
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3493.9768
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1580.3055
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:866.2456
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:327.2530
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:862.8458
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:140.4013
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3626.3452
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:902.4340
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:341.7497
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:305.0541
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:881.6512
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.6617
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3761.7080
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:648.1323
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:243.1951
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:321.7523
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:827.2345
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:136.1818
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3921.7783
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:640.5050
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:275.9445
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:313.6765
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:880.1647
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:139.9085
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4095.7849
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:429.4743
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:292.8451
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:348.6465
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:911.6647
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:141.7134
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4263.9990
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:690.2723
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:278.1185
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:394.3011
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:949.9160
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:155.6963
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4409.7236
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1130.5625
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:781.2274
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:385.9520
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:975.5074
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:164.1738
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4532.8535
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:809.9911
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:511.7096
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:390.0358
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:909.0684
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:172.5272
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4684.1265
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:766.0821
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:408.5670
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:452.3821
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:995.6998
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:177.9937
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4829.4722
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:217495.1875, RMS->RMS:1146.7664
step:4260/6200 train_time:641431ms step_avg:150.57ms
step:4270/6200 train_time:642988ms step_avg:150.58ms
step:4280/6200 train_time:644545ms step_avg:150.59ms
step:4290/6200 train_time:646103ms step_avg:150.61ms
step:4300/6200 train_time:647660ms step_avg:150.62ms
step:4310/6200 train_time:649217ms step_avg:150.63ms
step:4320/6200 train_time:650823ms step_avg:150.65ms
step:4330/6200 train_time:652380ms step_avg:150.67ms
step:4340/6200 train_time:653937ms step_avg:150.68ms
step:4350/6200 train_time:655492ms step_avg:150.69ms
step:4360/6200 train_time:657051ms step_avg:150.70ms
step:4370/6200 train_time:658611ms step_avg:150.71ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9375, RMS->RMS:380.5927
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9643
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9888
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9624
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9701
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9648
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9717
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9743
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9850
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9794
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9740
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9809
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9737
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9832
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9897
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9847
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:146.7802, RMS->RMS:0.9798
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7070
Logits:    0.6746
>>> Act Max Entries:
Embed:     8.1875
Block #0: 7.5938
Block #1: 7.0312
Block #2: 6.5000
Block #3: 6.0000
Block #4: 5.5625
Block #5: 5.1562
Block #6: 4.7812
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8281
Block #10: 3.5469
Block #11: 3.2969
Logits:    20.3750
step:4375/6200 val_loss:5.5686 val_acc:0.1977 train_time:659422ms step_avg:150.73ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.7500, RMS->RMS:30.1812
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1154.1238
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:533.3235
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:357.2705
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:781.5869
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:122.3992
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3214.5654
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1374.6884
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1180.4083
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:437.3629
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:894.2278
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:132.7084
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3453.9099
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1415.0048
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1312.1362
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:306.4128
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:844.4395
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.4955
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3541.6533
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1145.7043
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:731.6149
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:337.8324
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:905.1146
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:161.8316
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3667.5173
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:998.3218
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:426.2523
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:329.9931
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:881.6743
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:161.7288
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3760.9861
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:695.3533
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:178.4024
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:354.5322
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:841.9598
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:157.7748
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3895.6470
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:611.9248
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:286.5627
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:331.5152
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:898.1009
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.1118
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4045.7808
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:909.2938
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:442.8696
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:380.2335
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:924.2294
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:162.8755
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4189.4888
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:724.6624
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:300.9689
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:407.2630
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:966.5867
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:168.8998
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4325.6255
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:999.4294
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:548.3275
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:412.8023
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:994.9504
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:179.5378
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4438.4004
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:733.3922
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:468.2597
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:375.5435
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:917.0580
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:183.7007
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4589.7944
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:441.0197
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:201.0653
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:420.1919
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1003.1081
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:184.1660
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4765.5352
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:163260.2031, RMS->RMS:942.1222
step:4380/6200 train_time:660185ms step_avg:150.73ms
step:4390/6200 train_time:661745ms step_avg:150.74ms
step:4400/6200 train_time:663304ms step_avg:150.75ms
step:4410/6200 train_time:664863ms step_avg:150.76ms
step:4420/6200 train_time:666427ms step_avg:150.78ms
step:4430/6200 train_time:667989ms step_avg:150.79ms
step:4440/6200 train_time:669546ms step_avg:150.80ms
step:4450/6200 train_time:671101ms step_avg:150.81ms
step:4460/6200 train_time:672657ms step_avg:150.82ms
step:4470/6200 train_time:674215ms step_avg:150.83ms
step:4480/6200 train_time:675772ms step_avg:150.84ms
step:4490/6200 train_time:677331ms step_avg:150.85ms
step:4500/6200 train_time:678889ms step_avg:150.86ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9375, RMS->RMS:379.9662
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9880
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9648
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9641
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9850
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9682
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9849
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9720
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9843
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9775
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9840
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9719
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9797
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9837
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9740
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9829
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9896
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9848
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9839
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:148.6174, RMS->RMS:0.9849
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6763
>>> Act Max Entries:
Embed:     7.9375
Block #0: 7.3438
Block #1: 6.8125
Block #2: 6.3125
Block #3: 5.8750
Block #4: 5.4688
Block #5: 5.0938
Block #6: 4.7500
Block #7: 4.4062
Block #8: 4.0938
Block #9: 3.8125
Block #10: 3.5625
Block #11: 3.3125
Logits:    20.7500
step:4500/6200 val_loss:5.5638 val_acc:0.1977 train_time:678922ms step_avg:150.87ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.4062, RMS->RMS:20.0059
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1483.7443
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:533.7762
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.0335
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:893.6108
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:146.6580
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3825.6279
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1702.9642
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1447.3922
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:457.5923
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1060.8379
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:152.3075
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4131.2515
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1753.4548
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1664.3970
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:365.0159
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:995.4702
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:172.1181
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4276.9009
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2126.1501
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1214.8408
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:371.6151
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1070.0509
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:188.5514
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4373.5396
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:998.3241
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:384.0359
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:382.9630
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1059.6041
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:192.7828
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4489.9380
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1048.2592
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:277.0919
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:400.7040
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:990.0288
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:184.9570
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4601.9277
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1000.2912
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:262.1615
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:388.0849
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1060.3958
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:191.0634
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4728.5776
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:653.4227
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:219.4197
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:424.5864
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1072.8040
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:199.0545
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4882.4951
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:967.7391
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:374.7764
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:455.8237
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1091.6251
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:210.3094
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5008.1812
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1284.1433
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:737.4174
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:441.7643
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1119.3186
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:217.1982
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5104.0166
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:938.2781
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:372.3221
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:440.6625
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1022.8177
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:225.7236
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5231.2671
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:878.6927
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:274.6293
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:484.8718
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1100.1705
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:229.5038
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5358.9512
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:164351.3281, RMS->RMS:1079.2527
step:4510/6200 train_time:680464ms step_avg:150.88ms
step:4520/6200 train_time:682026ms step_avg:150.89ms
step:4530/6200 train_time:683582ms step_avg:150.90ms
step:4540/6200 train_time:685143ms step_avg:150.91ms
step:4550/6200 train_time:686698ms step_avg:150.92ms
step:4560/6200 train_time:688258ms step_avg:150.93ms
step:4570/6200 train_time:689817ms step_avg:150.94ms
step:4580/6200 train_time:691417ms step_avg:150.96ms
step:4590/6200 train_time:692976ms step_avg:150.98ms
step:4600/6200 train_time:694534ms step_avg:150.99ms
step:4610/6200 train_time:696099ms step_avg:151.00ms
step:4620/6200 train_time:697658ms step_avg:151.01ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9688, RMS->RMS:380.5260
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9631
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9619
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9866
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9627
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9702
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9634
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9668
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9853
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9626
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9676
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9749
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9861
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9771
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9796
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9848
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9842
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9724
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9824
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9846
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9844
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:150.1211, RMS->RMS:0.9908
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6782
>>> Act Max Entries:
Embed:     8.4375
Block #0: 7.8125
Block #1: 7.2500
Block #2: 6.7188
Block #3: 6.2500
Block #4: 5.8125
Block #5: 5.4062
Block #6: 5.0312
Block #7: 4.6875
Block #8: 4.3750
Block #9: 4.0625
Block #10: 3.7969
Block #11: 3.5312
Logits:    19.7500
step:4625/6200 val_loss:5.5545 val_acc:0.1989 train_time:698471ms step_avg:151.02ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:10.6250, RMS->RMS:37.1853
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1246.2572
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:530.9341
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:396.3687
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:1006.9383
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:168.1680
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4244.8911
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1564.6865
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1322.8782
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:393.6920
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1115.3145
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:175.4096
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4519.1875
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1752.5583
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1626.3635
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:561.0803
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1207.6541
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:199.7061
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4648.3535
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1564.0371
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:975.3792
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:427.9582
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1209.7305
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:209.5351
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4772.5698
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1193.3970
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:580.8337
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:536.6054
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1196.4923
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:222.1644
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4885.8970
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:969.3121
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:277.1120
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:487.7137
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1107.6902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:216.0208
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5003.8467
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:810.9753
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:335.4664
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:463.0042
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1172.0259
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:223.4331
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5141.9028
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:783.8111
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:266.1092
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:489.6636
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1170.6644
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:225.7895
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5280.9551
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:929.6492
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:351.2406
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:520.6475
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1182.1101
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:231.0743
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5397.0713
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1226.8331
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:662.9436
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:475.6959
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1223.7091
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:243.0169
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5504.1475
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:920.7617
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:482.8698
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:482.0798
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1112.7706
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:243.2153
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5637.7441
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:676.0681
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:397.2480
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:530.4090
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1192.7903
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:251.0641
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5789.2979
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:136149.5781, RMS->RMS:1018.8795
step:4630/6200 train_time:699237ms step_avg:151.02ms
step:4640/6200 train_time:700790ms step_avg:151.03ms
step:4650/6200 train_time:702354ms step_avg:151.04ms
step:4660/6200 train_time:703917ms step_avg:151.06ms
step:4670/6200 train_time:705476ms step_avg:151.07ms
step:4680/6200 train_time:707034ms step_avg:151.08ms
step:4690/6200 train_time:708595ms step_avg:151.09ms
step:4700/6200 train_time:710153ms step_avg:151.10ms
step:4710/6200 train_time:711718ms step_avg:151.11ms
step:4720/6200 train_time:713279ms step_avg:151.12ms
step:4730/6200 train_time:714841ms step_avg:151.13ms
step:4740/6200 train_time:716405ms step_avg:151.14ms
step:4750/6200 train_time:717968ms step_avg:151.15ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.9688, RMS->RMS:381.3120
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9629
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9611
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9861
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9616
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9619
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9700
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9653
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9637
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9848
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9627
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9683
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9841
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9673
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9743
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9675
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9764
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9849
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9843
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9724
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9791
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9850
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9824
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9898
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9846
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9844
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:152.0678, RMS->RMS:0.9969
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6809
>>> Act Max Entries:
Embed:     10.0000
Block #0: 9.3125
Block #1: 8.6875
Block #2: 8.0625
Block #3: 7.5000
Block #4: 6.9688
Block #5: 6.5000
Block #6: 6.0625
Block #7: 5.6562
Block #8: 5.2812
Block #9: 4.9375
Block #10: 4.5938
Block #11: 4.2812
Logits:    21.1250
step:4750/6200 val_loss:5.5486 val_acc:0.1993 train_time:718002ms step_avg:151.16ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.3125, RMS->RMS:32.3922
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:2005.3499
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:692.6476
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:471.3400
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:1225.8086
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:218.1319
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5244.8042
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2525.2280
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1829.9298
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:487.8918
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1329.2021
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:231.9401
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5577.9541
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2556.4563
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:2148.4238
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:641.6918
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1384.2347
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:266.2009
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5723.0122
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2871.0327
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1610.6763
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:459.6325
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1371.1277
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:282.0996
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5772.3081
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1866.3124
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:756.3495
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:603.7673
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1333.6555
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:295.6694
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5826.8643
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1805.9147
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:646.8427
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:530.0132
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1219.9159
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:283.8356
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5852.3369
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1401.5819
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:377.5623
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:512.9706
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1295.4551
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:289.2928
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5924.6582
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1085.5558
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:348.9239
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:524.0840
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1299.9138
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:289.7208
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6019.9873
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1347.2716
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:393.6539
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:533.0392
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1299.7126
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:290.5188
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6087.7739
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1722.1686
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:872.0480
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:535.2341
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1291.5769
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:302.3987
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:6125.2842
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1287.6295
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:505.6566
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:521.3002
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:1201.3145
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:309.9737
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:6206.7427
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:1171.5605
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:364.1530
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:570.8839
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1308.1595
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:308.0955
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:6295.4893
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:186757.1562, RMS->RMS:1060.8904
step:4760/6200 train_time:719548ms step_avg:151.17ms
step:4770/6200 train_time:721107ms step_avg:151.18ms
step:4780/6200 train_time:722669ms step_avg:151.19ms
step:4790/6200 train_time:724231ms step_avg:151.20ms
step:4800/6200 train_time:725789ms step_avg:151.21ms
step:4810/6200 train_time:727358ms step_avg:151.22ms
step:4820/6200 train_time:728918ms step_avg:151.23ms
step:4830/6200 train_time:730529ms step_avg:151.25ms
step:4840/6200 train_time:732093ms step_avg:151.26ms
step:4850/6200 train_time:733653ms step_avg:151.27ms
step:4860/6200 train_time:735216ms step_avg:151.28ms
step:4870/6200 train_time:736782ms step_avg:151.29ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.1250, RMS->RMS:383.2712
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9624
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9668
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9865
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9640
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9923
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9878
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9609
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9623
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9672
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9854
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9616
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9680
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9867
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9849
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9626
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9904
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9837
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9739
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9918
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9851
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9835
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9653
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9763
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9847
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9830
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9715
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9781
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9840
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9816
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9900
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9840
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9837
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:154.3191, RMS->RMS:1.0032
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8438
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.6830
>>> Act Max Entries:
Embed:     9.5000
Block #0: 8.8125
Block #1: 8.1875
Block #2: 7.5938
Block #3: 7.0625
Block #4: 6.5625
Block #5: 6.1250
Block #6: 5.7188
Block #7: 5.3438
Block #8: 4.9688
Block #9: 4.6250
Block #10: 4.3125
Block #11: 4.0312
Logits:    20.0000
step:4875/6200 val_loss:5.5329 val_acc:0.2005 train_time:737594ms step_avg:151.30ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.3125, RMS->RMS:25.5948
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1262.6669
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:533.3010
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:359.9870
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:825.9537
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.9015
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3479.5745
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1478.7321
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1204.3411
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:430.1845
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:930.1221
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:147.7396
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3711.9387
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1613.5604
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1549.9867
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:325.1721
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:886.2601
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.7112
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3792.6548
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1341.1349
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:850.3698
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:351.4604
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:939.8640
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.0701
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3922.8481
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:947.5128
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:435.2033
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:347.1130
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:915.7220
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.3853
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4032.0325
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:769.6909
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:194.0541
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:374.8076
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:871.2512
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.4788
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4159.5781
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:720.2834
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:308.9651
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:347.4866
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:939.9350
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:174.1007
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4297.0771
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1297.7783
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:649.2136
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:426.0377
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:962.2635
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:187.4128
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4405.0186
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:856.5355
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:402.0771
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:439.8394
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:988.3182
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:196.3387
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4512.8477
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1093.9313
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:615.5103
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:393.0490
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:997.3776
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:210.8359
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4590.9683
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:758.2668
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:351.6164
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:385.9849
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:937.6620
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:207.2063
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4704.6523
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:601.8198
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:346.4079
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:430.9538
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:990.4196
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:214.7153
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4847.7812
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:137279.4062, RMS->RMS:918.6240
step:4880/6200 train_time:738363ms step_avg:151.30ms
step:4890/6200 train_time:739924ms step_avg:151.31ms
step:4900/6200 train_time:741485ms step_avg:151.32ms
step:4910/6200 train_time:743045ms step_avg:151.33ms
step:4920/6200 train_time:744603ms step_avg:151.34ms
step:4930/6200 train_time:746164ms step_avg:151.35ms
step:4940/6200 train_time:747724ms step_avg:151.36ms
step:4950/6200 train_time:749286ms step_avg:151.37ms
step:4960/6200 train_time:750852ms step_avg:151.38ms
step:4970/6200 train_time:752417ms step_avg:151.39ms
step:4980/6200 train_time:753980ms step_avg:151.40ms
step:4990/6200 train_time:755544ms step_avg:151.41ms
step:5000/6200 train_time:757102ms step_avg:151.42ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.1250, RMS->RMS:382.8174
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9620
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9605
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9873
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9600
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9619
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9923
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9623
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9920
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9633
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9842
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9607
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9841
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9734
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9851
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9835
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9671
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9746
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9847
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9833
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9692
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9774
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9902
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9837
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9706
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9815
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9847
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9837
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:155.9755, RMS->RMS:1.0098
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.6854
>>> Act Max Entries:
Embed:     8.5000
Block #0: 7.9062
Block #1: 7.3438
Block #2: 6.8438
Block #3: 6.3750
Block #4: 5.9375
Block #5: 5.5312
Block #6: 5.1875
Block #7: 4.8438
Block #8: 4.5312
Block #9: 4.2188
Block #10: 3.9375
Block #11: 3.6875
Logits:    20.5000
step:5000/6200 val_loss:5.5177 val_acc:0.2018 train_time:757136ms step_avg:151.43ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.9375, RMS->RMS:30.8956
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1410.7362
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:547.3711
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.0057
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:869.4280
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:134.7935
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3718.1355
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1878.6770
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1523.5577
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:499.5071
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1052.8329
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.5398
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3994.5869
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2005.3132
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1822.4501
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.1541
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:951.1951
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:182.7139
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4059.0061
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2342.2878
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1341.5994
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:389.1161
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1000.2375
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:200.7231
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4079.5134
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1253.2347
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:525.5676
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:360.1463
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:965.3975
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:203.4250
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4134.4072
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1356.6786
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:321.5380
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:374.8067
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:884.8099
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:194.3777
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4164.5063
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1120.7421
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:425.1410
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.2772
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:941.7253
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:197.2160
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4227.7803
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:939.4834
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:313.9447
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:374.9541
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:939.1776
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:199.4343
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4303.8257
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1192.8120
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:323.4921
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:381.5293
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:956.3024
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:196.1115
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4356.4614
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1280.9095
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:625.4922
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:395.4519
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:970.7554
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:207.0831
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4395.4365
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1170.7719
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:390.5251
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:367.7002
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:880.1117
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:211.0845
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4442.0244
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:932.9569
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:322.8713
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:395.4546
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:924.4403
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:207.0732
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4522.0728
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:115448.5156, RMS->RMS:788.1259
step:5010/6200 train_time:758683ms step_avg:151.43ms
step:5020/6200 train_time:760245ms step_avg:151.44ms
step:5030/6200 train_time:761805ms step_avg:151.45ms
step:5040/6200 train_time:763365ms step_avg:151.46ms
step:5050/6200 train_time:764927ms step_avg:151.47ms
step:5060/6200 train_time:766494ms step_avg:151.48ms
step:5070/6200 train_time:768071ms step_avg:151.49ms
step:5080/6200 train_time:769645ms step_avg:151.50ms
step:5090/6200 train_time:771256ms step_avg:151.52ms
step:5100/6200 train_time:772833ms step_avg:151.54ms
step:5110/6200 train_time:774407ms step_avg:151.55ms
step:5120/6200 train_time:775989ms step_avg:151.56ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.2500, RMS->RMS:384.2563
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9605
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9614
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9639
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9928
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9600
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9850
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9611
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9667
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9925
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9616
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9666
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9865
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9845
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9622
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9925
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9841
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9622
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9857
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9838
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9854
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9849
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9840
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9634
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9922
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9906
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9848
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9694
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9775
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9903
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9839
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9696
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9807
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9899
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9834
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:157.5284, RMS->RMS:1.0160
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6934
>>> Act Max Entries:
Embed:     8.1875
Block #0: 7.5938
Block #1: 7.0625
Block #2: 6.5625
Block #3: 6.0938
Block #4: 5.6562
Block #5: 5.2500
Block #6: 4.9062
Block #7: 4.5938
Block #8: 4.3125
Block #9: 4.0312
Block #10: 3.7656
Block #11: 3.5312
Logits:    20.6250
step:5125/6200 val_loss:5.5066 val_acc:0.2036 train_time:776806ms step_avg:151.57ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:6.7500, RMS->RMS:23.3292
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1217.9515
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:429.1377
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:309.1722
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:774.5269
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.6264
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3250.6826
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1346.4719
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1127.8846
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:331.8695
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:805.5914
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:153.2215
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3392.6091
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1423.1886
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1349.7546
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:452.6485
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:878.3984
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.6556
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3499.3684
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1716.3550
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:989.6048
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:316.3308
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:854.1653
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:176.0431
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3530.7749
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:776.6824
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:272.0055
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:382.2043
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:866.2358
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:178.6383
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3592.2690
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:967.3144
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:229.7892
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:320.7312
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:765.3441
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:176.0166
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3635.1597
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:804.0530
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:207.3123
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:320.4562
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:806.8220
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:174.7309
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3705.0066
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:595.6687
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:160.6581
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:332.5115
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:814.9513
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:183.6161
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3764.1882
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:777.1698
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:305.9459
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:348.0366
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:816.8299
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:184.5030
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3816.0720
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1063.2784
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:608.0954
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:339.0249
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:818.7368
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:192.6741
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3845.0435
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:685.4684
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:254.4802
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:339.2080
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:739.8807
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:196.0108
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3910.1975
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:581.6440
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:157.6929
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:362.6368
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:823.1705
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:196.7345
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3979.5352
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:107228.6562, RMS->RMS:694.1621
step:5130/6200 train_time:777589ms step_avg:151.58ms
step:5140/6200 train_time:779173ms step_avg:151.59ms
step:5150/6200 train_time:780746ms step_avg:151.60ms
step:5160/6200 train_time:782325ms step_avg:151.61ms
step:5170/6200 train_time:783899ms step_avg:151.62ms
step:5180/6200 train_time:785470ms step_avg:151.64ms
step:5190/6200 train_time:787050ms step_avg:151.65ms
step:5200/6200 train_time:788627ms step_avg:151.66ms
step:5210/6200 train_time:790203ms step_avg:151.67ms
step:5220/6200 train_time:791778ms step_avg:151.68ms
step:5230/6200 train_time:793353ms step_avg:151.69ms
step:5240/6200 train_time:794933ms step_avg:151.70ms
step:5250/6200 train_time:796506ms step_avg:151.72ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.2500, RMS->RMS:385.3021
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9598
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9877
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9590
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9930
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9591
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9845
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9600
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9930
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9868
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9597
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9841
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9599
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9836
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9598
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9687
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9909
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9852
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9832
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9639
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9907
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9844
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9830
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9625
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9905
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9829
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9636
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9838
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9828
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9684
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9931
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9842
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9832
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9693
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9810
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9923
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9901
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9836
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9832
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:159.2567, RMS->RMS:1.0213
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.6940
>>> Act Max Entries:
Embed:     7.9688
Block #0: 7.3750
Block #1: 6.8438
Block #2: 6.3750
Block #3: 5.9688
Block #4: 5.5938
Block #5: 5.2188
Block #6: 4.8750
Block #7: 4.5625
Block #8: 4.2812
Block #9: 4.0000
Block #10: 3.7344
Block #11: 3.4844
Logits:    20.6250
step:5250/6200 val_loss:5.5005 val_acc:0.2042 train_time:796540ms step_avg:151.72ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:3.9375, RMS->RMS:15.5806
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1834.4240
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:752.9614
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:426.3726
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:983.9365
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.0069
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4235.2588
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2373.0078
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1869.7239
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:532.3978
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1157.4089
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:184.9323
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4555.5518
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2366.0642
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:2064.4954
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:408.6318
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1060.8032
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:205.5888
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4636.4590
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2704.8430
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1541.0542
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:422.7788
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1111.4816
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:219.7221
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4681.3018
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1529.2327
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:624.5331
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:425.9253
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1071.2728
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:223.8763
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4741.9175
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1554.9276
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:405.9965
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:421.9762
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:998.1260
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:216.8931
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4784.5415
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1156.2999
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:342.2108
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:404.8394
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1077.6031
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:214.5723
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4890.1865
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1029.0775
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:337.7831
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:429.8602
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1074.7074
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:225.2609
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5000.5405
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1324.4332
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:480.5247
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:459.1899
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1092.8481
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:228.6384
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5079.1816
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1587.6210
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:857.2825
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:437.3119
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1071.2367
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:244.5884
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5110.8896
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1436.3835
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:529.9416
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:427.2125
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:977.4770
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:243.9049
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5174.2573
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:720.5005
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:194.5614
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:483.8972
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1082.8562
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:253.5826
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5304.9180
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:146809.9062, RMS->RMS:943.4512
step:5260/6200 train_time:798111ms step_avg:151.73ms
step:5270/6200 train_time:799686ms step_avg:151.74ms
step:5280/6200 train_time:801260ms step_avg:151.75ms
step:5290/6200 train_time:802842ms step_avg:151.77ms
step:5300/6200 train_time:804411ms step_avg:151.78ms
step:5310/6200 train_time:805985ms step_avg:151.79ms
step:5320/6200 train_time:807557ms step_avg:151.80ms
step:5330/6200 train_time:809130ms step_avg:151.81ms
step:5340/6200 train_time:810755ms step_avg:151.83ms
step:5350/6200 train_time:812331ms step_avg:151.84ms
step:5360/6200 train_time:813912ms step_avg:151.85ms
step:5370/6200 train_time:815489ms step_avg:151.86ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3125, RMS->RMS:386.1968
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9610
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9632
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9931
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9917
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9857
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9593
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9932
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9849
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9587
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9933
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9911
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9876
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9842
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9599
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9931
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9915
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9929
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9910
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9866
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9610
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9935
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9858
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9835
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9600
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9930
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9913
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9837
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9625
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9932
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9829
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9625
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9722
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9935
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9912
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9844
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9833
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9624
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9732
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9928
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9839
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9829
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9685
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9930
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9914
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9835
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9830
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9695
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9810
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9931
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9908
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9834
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9833
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:160.8646, RMS->RMS:1.0263
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.6987
>>> Act Max Entries:
Embed:     7.8125
Block #0: 7.2812
Block #1: 6.7812
Block #2: 6.3438
Block #3: 5.9375
Block #4: 5.5625
Block #5: 5.1875
Block #6: 4.8438
Block #7: 4.5312
Block #8: 4.2500
Block #9: 3.9844
Block #10: 3.7188
Block #11: 3.4844
Logits:    20.3750
step:5375/6200 val_loss:5.4834 val_acc:0.2053 train_time:816308ms step_avg:151.87ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4062, RMS->RMS:26.0770
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:2092.4941
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:800.7678
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:456.7390
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:1063.8961
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:164.8115
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4613.0215
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2538.6428
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1989.8177
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:566.7361
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:1238.6259
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:212.4341
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4931.7476
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2663.6570
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:2310.8796
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:418.7176
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1126.6770
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:236.5595
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4984.7866
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2827.8542
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1516.2802
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:442.0410
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:1167.8295
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:252.6005
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4994.4526
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1763.3353
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:660.3068
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:440.7586
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:1165.8837
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:247.1275
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5053.9463
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1989.2126
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:509.9615
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:435.4304
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:1040.9207
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:239.9066
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5058.0835
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1563.6759
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:447.6680
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:410.2617
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:1110.8694
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:232.4273
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5127.6450
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:1038.3480
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:286.5373
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:444.4551
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:1135.6504
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:246.0291
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5230.3076
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1583.6826
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:546.4285
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:500.2500
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:1139.2358
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:252.1597
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5279.5752
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1941.7081
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:1091.7671
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:445.1085
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:1108.9303
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:265.7185
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:5272.8691
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1849.5529
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:698.5969
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:434.4266
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:995.5115
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:264.7858
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5286.5508
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:935.9646
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:273.1144
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:487.1977
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:1107.3027
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:270.7311
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:5390.1523
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:178915.6250, RMS->RMS:954.4412
step:5380/6200 train_time:817080ms step_avg:151.87ms
step:5390/6200 train_time:818649ms step_avg:151.88ms
step:5400/6200 train_time:820225ms step_avg:151.89ms
step:5410/6200 train_time:821802ms step_avg:151.90ms
step:5420/6200 train_time:823372ms step_avg:151.91ms
step:5430/6200 train_time:824947ms step_avg:151.92ms
step:5440/6200 train_time:826523ms step_avg:151.93ms
step:5450/6200 train_time:828091ms step_avg:151.94ms
step:5460/6200 train_time:829669ms step_avg:151.95ms
step:5470/6200 train_time:831243ms step_avg:151.96ms
step:5480/6200 train_time:832819ms step_avg:151.97ms
step:5490/6200 train_time:834396ms step_avg:151.98ms
step:5500/6200 train_time:835973ms step_avg:152.00ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3125, RMS->RMS:387.2585
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9617
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9636
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9925
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9595
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9630
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9941
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9929
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9850
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9585
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9938
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9608
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9941
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9923
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9841
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9938
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9869
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9838
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9597
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9659
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9923
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9859
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9604
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9941
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9925
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9858
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9834
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9937
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9916
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9825
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9940
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9847
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9830
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9619
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9735
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9937
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9921
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9840
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9828
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9686
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9938
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9837
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9826
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9699
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9802
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9919
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9835
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9831
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:161.2050, RMS->RMS:1.0294
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8438
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.7031
Logits:    0.7011
>>> Act Max Entries:
Embed:     7.8750
Block #0: 7.3438
Block #1: 6.8125
Block #2: 6.3438
Block #3: 5.9062
Block #4: 5.5000
Block #5: 5.1250
Block #6: 4.7500
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8438
Block #10: 3.5781
Block #11: 3.3281
Logits:    20.8750
step:5500/6200 val_loss:5.4701 val_acc:0.2065 train_time:836007ms step_avg:152.00ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:9.3125, RMS->RMS:32.2999
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1286.1895
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:462.0529
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:317.8598
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:725.8705
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:120.8893
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3149.5051
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1579.0085
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1268.9489
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:424.4166
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:850.7183
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.7939
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3336.2166
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1876.3705
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1630.1505
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:267.0423
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:734.5146
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:172.6269
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3333.4170
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1733.5878
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:892.4955
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:290.0731
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:752.2650
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:182.3114
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3321.1167
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1307.4967
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:498.9475
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:283.7141
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:730.8240
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:181.3156
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3308.9075
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1608.1194
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:393.6382
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:279.9010
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:658.9386
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:170.4648
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3251.6294
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1301.4944
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:335.4804
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:273.8602
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:696.0556
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:168.4783
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3231.9871
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:899.2515
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:218.1783
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:274.2911
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:695.6245
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:165.3734
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3252.2471
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1213.4744
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:339.4934
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:294.1159
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:704.3300
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.1004
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3241.1243
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1144.1797
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:548.6316
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:280.2227
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:687.3471
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:169.1726
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3235.6575
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:882.5317
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:300.0119
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:282.6653
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:632.7557
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:175.2503
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3263.6321
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:890.6195
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:269.7237
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:289.8700
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:654.3243
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:174.9276
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3290.6199
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:73142.1016, RMS->RMS:575.5564
step:5510/6200 train_time:837563ms step_avg:152.01ms
step:5520/6200 train_time:839136ms step_avg:152.02ms
step:5530/6200 train_time:840715ms step_avg:152.03ms
step:5540/6200 train_time:842298ms step_avg:152.04ms
step:5550/6200 train_time:843877ms step_avg:152.05ms
step:5560/6200 train_time:845464ms step_avg:152.06ms
step:5570/6200 train_time:847042ms step_avg:152.07ms
step:5580/6200 train_time:848616ms step_avg:152.08ms
step:5590/6200 train_time:850239ms step_avg:152.10ms
step:5600/6200 train_time:851815ms step_avg:152.11ms
step:5610/6200 train_time:853395ms step_avg:152.12ms
step:5620/6200 train_time:854977ms step_avg:152.13ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:8.3750, RMS->RMS:387.3647
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9598
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9638
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9934
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9886
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9582
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9629
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9942
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9935
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9883
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9851
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9569
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9630
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9940
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9935
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9594
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9929
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9874
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9843
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9594
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9649
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9941
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9839
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9582
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9655
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9940
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9930
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9838
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9583
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9645
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9929
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9862
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9833
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9615
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9942
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9928
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9854
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9834
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9611
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9944
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9835
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9618
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9927
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9849
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9635
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9751
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9944
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9926
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9843
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9840
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9795
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9944
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9924
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9840
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9834
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:162.6131, RMS->RMS:1.0304
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9453
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8672
Block #5: 0.8438
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7227
Block #11: 0.6992
Logits:    0.7011
>>> Act Max Entries:
Embed:     7.8438
Block #0: 7.2812
Block #1: 6.7812
Block #2: 6.3125
Block #3: 5.8750
Block #4: 5.4688
Block #5: 5.0938
Block #6: 4.7500
Block #7: 4.4062
Block #8: 4.0938
Block #9: 3.8125
Block #10: 3.5469
Block #11: 3.2969
Logits:    21.2500
step:5625/6200 val_loss:5.4593 val_acc:0.2074 train_time:855804ms step_avg:152.14ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:5.1875, RMS->RMS:18.2987
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1772.1564
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:586.8825
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.9581
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:893.4009
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.0438
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3810.4268
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2086.4910
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1617.3054
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:358.0786
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:935.8622
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:184.7227
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4000.7612
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2258.8254
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1959.1047
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:525.9233
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:1007.8508
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:208.3791
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4072.3914
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2759.4331
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1360.3469
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:354.4259
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:970.8392
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:215.8883
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4018.3428
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1302.9169
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:431.2900
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:421.8880
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:959.1844
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:216.1208
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4049.1816
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1812.9065
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:462.0990
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:355.9173
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:822.5743
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:211.7356
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4015.2061
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1503.2772
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:390.9155
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:347.6082
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:859.8636
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:206.4720
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4018.9885
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:794.6275
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:173.8125
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:358.9383
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:870.3414
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:212.9367
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4063.0730
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1261.5034
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:357.9391
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:371.4875
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:863.7632
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:213.6726
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4065.4514
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1836.5957
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:961.4224
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:385.3695
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:856.7070
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:218.7385
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:4012.1892
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1401.9265
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:479.3544
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:334.7318
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:750.5330
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:221.2418
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3994.3726
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:868.3094
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:252.8412
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:372.7621
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:810.9538
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:219.2289
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:4019.5588
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:121599.5078, RMS->RMS:654.5103
step:5630/6200 train_time:856577ms step_avg:152.15ms
step:5640/6200 train_time:858154ms step_avg:152.15ms
step:5650/6200 train_time:859730ms step_avg:152.16ms
step:5660/6200 train_time:861307ms step_avg:152.17ms
step:5670/6200 train_time:862884ms step_avg:152.18ms
step:5680/6200 train_time:864467ms step_avg:152.19ms
step:5690/6200 train_time:866041ms step_avg:152.20ms
step:5700/6200 train_time:867623ms step_avg:152.21ms
step:5710/6200 train_time:869204ms step_avg:152.22ms
step:5720/6200 train_time:870779ms step_avg:152.23ms
step:5730/6200 train_time:872357ms step_avg:152.24ms
step:5740/6200 train_time:873929ms step_avg:152.25ms
step:5750/6200 train_time:875505ms step_avg:152.26ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4375, RMS->RMS:387.2288
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9603
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9638
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9945
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9896
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9858
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9588
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9631
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9944
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9941
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9568
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9629
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9948
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9942
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9590
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9644
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9946
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9934
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9884
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9585
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9643
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9879
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9846
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9576
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9945
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9871
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9840
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9570
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9953
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9870
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9847
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9608
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9939
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9864
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9612
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9707
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9860
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9617
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9714
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9942
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9854
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9845
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9762
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9951
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9943
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9856
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9844
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9656
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9813
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9938
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9853
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9845
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:163.6448, RMS->RMS:1.0299
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8203
Block #7: 0.7969
Block #8: 0.7695
Block #9: 0.7461
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.7020
>>> Act Max Entries:
Embed:     7.9062
Block #0: 7.3438
Block #1: 6.8438
Block #2: 6.3750
Block #3: 5.9375
Block #4: 5.5312
Block #5: 5.1562
Block #6: 4.7812
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8438
Block #10: 3.5781
Block #11: 3.3281
Logits:    21.8750
step:5750/6200 val_loss:5.4448 val_acc:0.2092 train_time:875539ms step_avg:152.27ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:6.0938, RMS->RMS:21.1172
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1568.2894
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:693.7200
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:390.7734
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:772.0645
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:119.9476
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3243.8643
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2145.4570
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1642.4877
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:467.6456
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:905.7474
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:162.7897
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3447.7249
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2124.9490
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1732.8390
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:295.7667
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:786.0829
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:183.5621
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3399.5876
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2336.7283
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1263.9105
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:373.9213
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:822.3837
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:191.4521
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3339.6301
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1475.9342
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:564.9392
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:297.2461
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:749.1484
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:192.0922
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3294.3662
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:2160.2090
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:558.0093
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:306.8560
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:655.8523
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:174.8609
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3169.5823
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1496.4872
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:402.9855
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:275.3680
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:690.1451
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.7294
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3125.3845
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:962.5056
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:287.4102
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:275.6732
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:693.2225
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:166.8321
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3131.7173
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1250.1753
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:316.3690
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:291.8585
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:687.2390
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:167.2653
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3107.9443
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1381.3698
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:645.9072
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:281.4532
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:668.8401
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:168.3507
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3060.2024
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1704.4485
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:650.6498
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:248.3943
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:592.2537
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:162.2612
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:2976.7542
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:692.4243
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:273.8800
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:265.4212
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:631.2183
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:162.3804
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:3011.9834
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:66081.2969, RMS->RMS:501.3266
step:5760/6200 train_time:877098ms step_avg:152.27ms
step:5770/6200 train_time:878674ms step_avg:152.28ms
step:5780/6200 train_time:880258ms step_avg:152.29ms
step:5790/6200 train_time:881835ms step_avg:152.30ms
step:5800/6200 train_time:883414ms step_avg:152.31ms
step:5810/6200 train_time:884990ms step_avg:152.32ms
step:5820/6200 train_time:886566ms step_avg:152.33ms
step:5830/6200 train_time:888139ms step_avg:152.34ms
step:5840/6200 train_time:889720ms step_avg:152.35ms
step:5850/6200 train_time:891353ms step_avg:152.37ms
step:5860/6200 train_time:892927ms step_avg:152.38ms
step:5870/6200 train_time:894506ms step_avg:152.39ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4688, RMS->RMS:387.4011
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9607
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9639
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9910
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9875
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9591
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9636
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9904
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9867
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9569
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9638
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9951
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9904
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9862
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9594
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9642
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9948
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9945
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9895
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9596
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9950
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9945
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9899
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9859
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9581
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9650
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9948
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9890
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9578
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9662
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9852
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9605
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9947
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9881
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9860
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9619
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9698
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9953
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9875
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9856
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9627
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9724
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9863
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9855
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9643
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9781
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9872
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9864
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9653
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9834
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9859
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9862
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:164.0748, RMS->RMS:1.0304
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.7969
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.7032
>>> Act Max Entries:
Embed:     7.9375
Block #0: 7.3750
Block #1: 6.8438
Block #2: 6.3438
Block #3: 5.9062
Block #4: 5.5000
Block #5: 5.0938
Block #6: 4.7188
Block #7: 4.3750
Block #8: 4.0625
Block #9: 3.7812
Block #10: 3.5156
Block #11: 3.2812
Logits:    22.0000
step:5875/6200 val_loss:5.4320 val_acc:0.2103 train_time:895328ms step_avg:152.40ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:4.3750, RMS->RMS:15.8735
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1203.4799
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:508.6728
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:303.8534
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:630.0392
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:104.7702
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2718.3391
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1782.8584
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1391.1222
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:391.6146
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:746.7512
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:138.5178
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2882.8064
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1870.8416
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1509.3662
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:243.5936
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:625.0876
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.0889
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2846.5425
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1936.1815
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:1052.9861
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:321.8683
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:666.4534
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:160.6354
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2810.7749
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1241.5873
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:444.6207
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:243.4559
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:609.3143
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.4072
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2784.1907
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1699.9745
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:437.3469
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:240.8949
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:545.9161
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:148.7269
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2715.8237
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1217.3423
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:314.5276
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:223.4518
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:580.7987
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:141.1708
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2700.8499
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:804.3107
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:229.0603
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:226.5326
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:580.4809
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:141.4425
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2716.1531
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:845.8427
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:200.6938
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:248.8123
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:586.0781
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:143.2950
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2728.4829
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1363.2455
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:658.8051
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:243.3660
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:564.6481
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:151.6802
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2677.9731
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1432.9380
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:547.8714
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:222.3112
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:506.4776
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:144.6829
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:2623.8442
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:730.8599
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:273.0670
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:240.7876
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:535.4120
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:145.7153
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:2647.2292
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:88313.3750, RMS->RMS:479.4136
step:5880/6200 train_time:896105ms step_avg:152.40ms
step:5890/6200 train_time:897683ms step_avg:152.41ms
step:5900/6200 train_time:899262ms step_avg:152.42ms
step:5910/6200 train_time:900844ms step_avg:152.43ms
step:5920/6200 train_time:902423ms step_avg:152.44ms
step:5930/6200 train_time:904009ms step_avg:152.45ms
step:5940/6200 train_time:905595ms step_avg:152.46ms
step:5950/6200 train_time:907175ms step_avg:152.47ms
step:5960/6200 train_time:908761ms step_avg:152.48ms
step:5970/6200 train_time:910341ms step_avg:152.49ms
step:5980/6200 train_time:911932ms step_avg:152.50ms
step:5990/6200 train_time:913526ms step_avg:152.51ms
step:6000/6200 train_time:915132ms step_avg:152.52ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4688, RMS->RMS:386.9939
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9594
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9651
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9962
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9969
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9927
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9883
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9579
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9652
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9953
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9919
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9882
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9570
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9631
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9953
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9920
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9881
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9581
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9646
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9908
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9870
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9591
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9660
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9957
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9910
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9578
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9670
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9899
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9873
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9576
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9681
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9960
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9960
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9899
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9871
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9657
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9697
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9902
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9874
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9627
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9957
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9894
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9884
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9643
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9730
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9960
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9967
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9891
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9885
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9788
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9963
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9960
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9885
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9877
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9677
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9853
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9961
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9882
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9876
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:165.2584, RMS->RMS:1.0423
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7266
Block #11: 0.7031
Logits:    0.7037
>>> Act Max Entries:
Embed:     7.8750
Block #0: 7.3125
Block #1: 6.7812
Block #2: 6.2812
Block #3: 5.8438
Block #4: 5.4375
Block #5: 5.0625
Block #6: 4.7188
Block #7: 4.4062
Block #8: 4.0938
Block #9: 3.8125
Block #10: 3.5469
Block #11: 3.3125
Logits:    23.2500
step:6000/6200 val_loss:5.4209 val_acc:0.2117 train_time:915166ms step_avg:152.53ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:6.5938, RMS->RMS:23.0806
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1172.8271
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:509.0632
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:295.4491
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:608.3024
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:101.1606
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2494.2900
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:1693.4221
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1249.1573
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:381.1944
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:703.7203
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:129.5208
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2632.1296
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:1680.6394
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1363.4712
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:231.1230
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:580.1048
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:142.2758
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2559.7698
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:1912.2195
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:958.9172
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:317.5288
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:618.8634
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:148.8505
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2483.3315
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1444.1169
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:485.6819
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:230.1020
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:554.5282
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:147.9427
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2397.3247
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:1842.4338
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:507.5411
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:224.4716
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:458.1144
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:134.0267
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2262.6663
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1196.0789
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:326.9348
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:197.8135
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:489.2144
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:130.0768
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2205.9316
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:814.3010
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:260.9845
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:188.5595
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:477.3536
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.0460
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2181.1685
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:881.8193
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:207.7925
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:198.9314
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:474.1181
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:125.9016
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2148.6199
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1087.4712
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:480.3582
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:215.0998
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:447.9406
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:126.5039
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2084.0859
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1608.6257
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:647.7009
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:179.5499
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:399.1155
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:120.5783
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:1972.7992
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:595.4885
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:186.6072
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:188.9630
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:407.6687
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:117.9389
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:1969.3748
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:41209.1641, RMS->RMS:315.9840
step:6010/6200 train_time:916736ms step_avg:152.54ms
step:6020/6200 train_time:918335ms step_avg:152.55ms
step:6030/6200 train_time:919921ms step_avg:152.56ms
step:6040/6200 train_time:921503ms step_avg:152.57ms
step:6050/6200 train_time:923088ms step_avg:152.58ms
step:6060/6200 train_time:924682ms step_avg:152.59ms
step:6070/6200 train_time:926277ms step_avg:152.60ms
step:6080/6200 train_time:927873ms step_avg:152.61ms
step:6090/6200 train_time:929455ms step_avg:152.62ms
step:6100/6200 train_time:931098ms step_avg:152.64ms
step:6110/6200 train_time:932691ms step_avg:152.65ms
step:6120/6200 train_time:934275ms step_avg:152.66ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4688, RMS->RMS:386.3983
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9628
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9968
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9937
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9899
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9620
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9674
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9961
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9939
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9894
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9605
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9663
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9937
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9893
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9623
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9697
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9933
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9894
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9626
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9701
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9931
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9891
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9609
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9713
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9925
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9895
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9654
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9716
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9971
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9927
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9896
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9703
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9745
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9957
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9962
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9919
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9902
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9688
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9753
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9969
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9918
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9906
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9691
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9959
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9966
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9919
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9901
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9718
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9819
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9970
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9922
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9907
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9752
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9879
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9967
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9922
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9914
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:166.2597, RMS->RMS:1.0577
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9766
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8945
Block #4: 0.8711
Block #5: 0.8477
Block #6: 0.8242
Block #7: 0.8008
Block #8: 0.7734
Block #9: 0.7500
Block #10: 0.7305
Block #11: 0.7070
Logits:    0.7030
>>> Act Max Entries:
Embed:     7.8750
Block #0: 7.3125
Block #1: 6.8125
Block #2: 6.3438
Block #3: 5.9062
Block #4: 5.5000
Block #5: 5.1250
Block #6: 4.7500
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8438
Block #10: 3.5781
Block #11: 3.3438
Logits:    23.6250
step:6125/6200 val_loss:5.4081 val_acc:0.2132 train_time:935101ms step_avg:152.67ms
>>> Grads:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:4.2500, RMS->RMS:15.5079
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:1261.3002
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:473.2408
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:267.8036
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:697.4714
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:124.4656
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3011.6772
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:2000.2227
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:1441.1562
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:286.8111
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:731.9015
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:156.2311
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3152.4983
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:2223.0063
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:1544.4396
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:422.4624
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:798.7352
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:173.8597
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3115.1938
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:2127.3640
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:982.7106
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:271.1649
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:724.3253
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:178.9072
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:3016.0137
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:1533.0347
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:471.9492
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:350.2758
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:709.6100
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:177.6049
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2964.2871
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:2212.7268
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:575.4689
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:271.1416
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:587.2290
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:165.9053
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2821.4556
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:1488.5614
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:403.3061
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:252.1326
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:602.8399
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.0270
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2756.0901
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:795.1757
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:243.7247
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:251.4566
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:591.1918
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:159.2916
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2743.6853
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:1066.4830
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:257.0685
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:261.7369
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:586.2908
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:158.1104
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2704.5591
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:1410.9702
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:598.3116
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:244.3228
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:573.4082
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:154.6683
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:2645.4500
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:1727.2703
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:591.8458
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:212.9873
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:494.4948
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:147.8838
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:2544.2156
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:674.7365
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:218.6208
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:235.9205
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:509.6728
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:150.6450
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:2533.8938
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:53254.6953, RMS->RMS:362.7429
step:6130/6200 train_time:935880ms step_avg:152.67ms
step:6140/6200 train_time:937472ms step_avg:152.68ms
step:6150/6200 train_time:939070ms step_avg:152.69ms
step:6160/6200 train_time:940658ms step_avg:152.70ms
step:6170/6200 train_time:942251ms step_avg:152.71ms
step:6180/6200 train_time:943834ms step_avg:152.72ms
step:6190/6200 train_time:945416ms step_avg:152.73ms
step:6200/6200 train_time:947011ms step_avg:152.74ms
>>> Weights:
_orig_mod.embed.weight                   (50257, 768) :  l1->RMS:7.4688, RMS->RMS:385.9845
_orig_mod.blocks.0.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9661
_orig_mod.blocks.0.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9709
_orig_mod.blocks.0.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9961
_orig_mod.blocks.0.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9968
_orig_mod.blocks.0.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9943
_orig_mod.blocks.0.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9905
_orig_mod.blocks.1.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9664
_orig_mod.blocks.1.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9724
_orig_mod.blocks.1.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.1.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.1.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9943
_orig_mod.blocks.1.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9903
_orig_mod.blocks.2.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9621
_orig_mod.blocks.2.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9702
_orig_mod.blocks.2.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.2.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.2.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9943
_orig_mod.blocks.2.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9904
_orig_mod.blocks.3.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9679
_orig_mod.blocks.3.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9751
_orig_mod.blocks.3.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9950
_orig_mod.blocks.3.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9949
_orig_mod.blocks.3.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9937
_orig_mod.blocks.3.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9904
_orig_mod.blocks.4.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9665
_orig_mod.blocks.4.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9744
_orig_mod.blocks.4.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.4.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.4.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9938
_orig_mod.blocks.4.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9903
_orig_mod.blocks.5.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9640
_orig_mod.blocks.5.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9768
_orig_mod.blocks.5.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9952
_orig_mod.blocks.5.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9965
_orig_mod.blocks.5.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9931
_orig_mod.blocks.5.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9908
_orig_mod.blocks.6.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9678
_orig_mod.blocks.6.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9782
_orig_mod.blocks.6.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9955
_orig_mod.blocks.6.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9968
_orig_mod.blocks.6.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9931
_orig_mod.blocks.6.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9903
_orig_mod.blocks.7.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9755
_orig_mod.blocks.7.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9782
_orig_mod.blocks.7.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9954
_orig_mod.blocks.7.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9963
_orig_mod.blocks.7.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9929
_orig_mod.blocks.7.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9910
_orig_mod.blocks.8.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9726
_orig_mod.blocks.8.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9791
_orig_mod.blocks.8.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.8.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9965
_orig_mod.blocks.8.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9928
_orig_mod.blocks.8.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9914
_orig_mod.blocks.9.attn.attn_q.weight    (768, 768)   : RMS->RMS:0.9736
_orig_mod.blocks.9.attn.attn_k.weight    (768, 768)   : RMS->RMS:0.9803
_orig_mod.blocks.9.attn.attn_v.weight    (768, 768)   : RMS->RMS:0.9958
_orig_mod.blocks.9.attn.c_proj.weight    (768, 768)   : RMS->RMS:0.9961
_orig_mod.blocks.9.mlp.c_fc.weight       (3072, 768)  : RMS->RMS:0.9928
_orig_mod.blocks.9.mlp.c_proj.weight     (768, 3072)  : RMS->RMS:0.9913
_orig_mod.blocks.10.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9742
_orig_mod.blocks.10.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9851
_orig_mod.blocks.10.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9956
_orig_mod.blocks.10.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9974
_orig_mod.blocks.10.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9931
_orig_mod.blocks.10.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9928
_orig_mod.blocks.11.attn.attn_q.weight   (768, 768)   : RMS->RMS:0.9767
_orig_mod.blocks.11.attn.attn_k.weight   (768, 768)   : RMS->RMS:0.9890
_orig_mod.blocks.11.attn.attn_v.weight   (768, 768)   : RMS->RMS:0.9957
_orig_mod.blocks.11.attn.c_proj.weight   (768, 768)   : RMS->RMS:0.9967
_orig_mod.blocks.11.mlp.c_fc.weight      (3072, 768)  : RMS->RMS:0.9936
_orig_mod.blocks.11.mlp.c_proj.weight    (768, 3072)  : RMS->RMS:0.9929
_orig_mod.lm_head.weight                 (50304, 768) : RMS->INF:166.5384, RMS->RMS:1.0649
>>> Act RMS Norms:
Embed:     1.0000
Block #0: 0.9727
Block #1: 0.9492
Block #2: 0.9219
Block #3: 0.8984
Block #4: 0.8750
Block #5: 0.8516
Block #6: 0.8281
Block #7: 0.8008
Block #8: 0.7773
Block #9: 0.7539
Block #10: 0.7344
Block #11: 0.7109
Logits:    0.7060
>>> Act Max Entries:
Embed:     7.8750
Block #0: 7.3125
Block #1: 6.8125
Block #2: 6.3438
Block #3: 5.9062
Block #4: 5.5000
Block #5: 5.1250
Block #6: 4.7812
Block #7: 4.4375
Block #8: 4.1250
Block #9: 3.8438
Block #10: 3.5781
Block #11: 3.3438
Logits:    23.1250
step:6200/6200 val_loss:5.3999 val_acc:0.2140 train_time:947045ms step_avg:152.75ms
peak memory allocated: 29585 MiB reserved: 47290 MiB