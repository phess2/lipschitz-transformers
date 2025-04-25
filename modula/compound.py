from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, d_embed, num_blocks, dtype=None, project_dtype=None, zero_init=False, project=None, **kwargs):
    m = Linear(output_dim, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker="mlp_in") @ ReLU()
    for i in range(num_blocks-2):
        m = m @ Linear(d_embed, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"mlp_{i}") @ ReLU()
    return m @ Linear(d_embed, input_dim, dtype=dtype, project_dtype=project_dtype, zero_init=zero_init, project=project, tracker="mlp_out")

def Attention(num_heads, d_embed, d_query, d_value, dtype=None, project_dtype=None, zero_init=False, project=None, layer_idx=0):
    """Multi-head attention"""

    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"v{layer_idx}")
    W = Linear(d_embed, num_heads * d_value, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax() @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, dtype=None, project_dtype=None, softmax_scale=None, final_scale=None, residual_scale=None, scales_learnable=False, zero_init=False, project=None, **kwargs):
    embed = Embed(d_embed, vocab_size, dtype=dtype)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = Attention(num_heads, d_embed, d_embed // num_heads, d_embed // num_heads, dtype=dtype, project_dtype=project_dtype, zero_init=zero_init, project=project, layer_idx=i)
        mlp = Linear(d_embed, 4*d_embed, dtype=dtype, project_dtype=project_dtype, zero_init=zero_init, project=project, tracker=f"mlp_out{i}") @ GeLU() @ Linear(4*d_embed, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker=f"mlp_in{i}")
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block

    blocks.tare(absolute=blocks_mass)

    out = Linear(vocab_size, d_embed, dtype=dtype, project_dtype=project_dtype, project=project, tracker="mlp_final")
    return out @ blocks @ embed
