from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, d_embed, blocks, zero_init=False, project=None, **kwargs):
    m = Linear(output_dim, d_embed, project=project, tracker="mlp_in") @ ReLU()
    for i in range(blocks-2):
        m = m @ Linear(d_embed, d_embed, project=project, tracker=f"mlp_{i}") @ ReLU()
    return m @ Linear(d_embed, input_dim, project=project, zero_init=zero_init, tracker="mlp_out")

def ManifoldMLP(output_dim, input_dim, d_embed, blocks, **kwargs):
    m = ManifoldLinear(output_dim, d_embed, tracker="mlp_in") @ ReLU()
    for i in range(blocks-2):
        m = m @ ManifoldLinear(d_embed, d_embed, tracker=f"mlp_{i}") @ ReLU()
    return m @ ManifoldLinear(d_embed, input_dim, tracker="mlp_out")

def Attention(num_heads, d_embed, d_query, d_value, zero_init=False, project=None, layer_idx=0):
    """Multi-head attention"""

    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, project=project, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, project=project, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed, project=project, tracker=f"v{layer_idx}")
    W = Linear(d_embed, num_heads * d_value, project=project, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax() @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, softmax_scale=None, final_scale=None, residual_scale=None, scales_learnable=False, zero_init=False, project=None, **kwargs):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = Attention(num_heads, d_embed, d_embed // num_heads, d_embed // num_heads, zero_init=zero_init, project=project, layer_idx=i)
        mlp = Linear(d_embed, 4*d_embed, zero_init=zero_init, project=project, tracker=f"mlp_out{i}") @ GeLU() @ Linear(4*d_embed, d_embed, project=project, tracker=f"mlp_in{i}")
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block

    blocks.tare(absolute=blocks_mass)

    out = Linear(vocab_size, d_embed, project=project, tracker="mlp_final")
    return out @ blocks @ embed

def OrthogonalAttention(num_heads, d_embed, softmax_scale, layer_idx=0):
    """
    Orthogonal attention uses 3-tensors for Q, K, V to make the input and output dimensions explicitly equal.
    """
    Q = TransposeHeads() @ ManifoldHeadedLinear(num_heads, d_embed, d_embed, tracker=f"q{layer_idx}")
    K = TransposeHeads() @ ManifoldHeadedLinear(num_heads, d_embed, d_embed, tracker=f"k{layer_idx}")
    V = TransposeHeads() @ ManifoldHeadedLinear(num_heads, d_embed, d_embed, tracker=f"v{layer_idx}")
    W = ManifoldHeadedLinearOut(num_heads, d_embed, d_embed, tracker=f"w{layer_idx}") @ TransposeHeads()

    AttentionScores = Softmax() @ SquareScalar(scale=softmax_scale, tracker=f"softmax{layer_idx}") @ CausalMask() @ AttentionQK() @ Rope(d_embed) @ (Q, K)
    return ReduceHeads() @ ((1/3) * W) @ ApplyAttentionScores() @ (V, AttentionScores)

def OrthogonalGPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, softmax_scale=1.0, final_scale=1.0, residual_scale=1.0, scales_learnable=False, zero_init=False, project=None, **kwargs):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = OrthogonalAttention(num_heads, d_embed, softmax_scale, layer_idx=i)
        mlp = ManifoldLinear(d_embed, d_embed, tracker=f"mlp_out{i}") @ GeLU() @ ManifoldLinear(d_embed, d_embed, tracker=f"mlp_in{i}")
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block
    
    blocks.tare(absolute=blocks_mass)

    out = SquareScalar(scale=final_scale, tracker="final_scale") @ Linear(vocab_size, d_embed, tracker="mlp_final")

    return out @ blocks @ embed
