from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, d_embed, num_blocks, dtype=None, project_dtype=None, zero_init=False, project=None, **kwargs):
    project_kwargs = {"dtype": dtype, "project_dtype": project_dtype, "project": project}
    m = Linear(output_dim, d_embed, **project_kwargs, tracker="mlp_in") @ ReLU()
    for i in range(num_blocks-2):
        m = m @ Linear(d_embed, d_embed, **project_kwargs, tracker=f"mlp_{i}") @ ReLU()
    return m @ Linear(d_embed, input_dim, **project_kwargs, zero_init=zero_init, tracker="mlp_out")

def Attention(num_heads, d_embed, d_query, d_value, dtype=None, project_dtype=None, zero_init=False, project=None, layer_idx=0):
    """Multi-head attention"""

    project_kwargs = {"dtype": dtype, "project_dtype": project_dtype, "project": project}
    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, **project_kwargs, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, **project_kwargs, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed, **project_kwargs, tracker=f"v{layer_idx}")
    W = Linear(d_embed, num_heads * d_value, **project_kwargs, zero_init=zero_init, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax() @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, dtype=None, project_dtype=None, softmax_scale=None, final_scale=None, residual_scale=None, scales_learnable=False, zero_init=False, project=None, max_embed_inflation_factor=1, **kwargs):
    project_kwargs = {"dtype": dtype, "project_dtype": project_dtype, "project": project}

    embed = Embed(d_embed, vocab_size, dtype=dtype, max_inflation_factor=max_embed_inflation_factor, tracker="embed")
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = Attention(num_heads, d_embed, d_embed // num_heads, d_embed // num_heads, **project_kwargs, zero_init=zero_init, layer_idx=i)
        linear_out = Linear(d_embed, 4*d_embed, **project_kwargs, zero_init=zero_init, tracker=f"mlp_out{i}")
        linear_in = Linear(4*d_embed, d_embed, **project_kwargs, tracker=f"mlp_in{i}")
        mlp = linear_out @ GeLU() @ linear_in
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block

    blocks.tare(absolute=blocks_mass)

    out = Linear(vocab_size, d_embed, **project_kwargs, tracker="mlp_final")
    return out @ blocks @ embed
