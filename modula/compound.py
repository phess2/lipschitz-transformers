from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width, tracker="mlp_in") @ ReLU()
    for i in range(depth-2):
        m = m @ Linear(width, width, tracker=f"mlp_{i}") @ ReLU()
    return m @ Linear(width, input_dim, tracker="mlp_out")

def ManifoldMLP(output_dim, input_dim, width, depth):
    m = ManifoldLinear(output_dim, width, tracker="mlp_in") @ ReLU()
    for i in range(depth-2):
        m = m @ ManifoldLinear(width, width, tracker=f"mlp_{i}") @ ReLU()
    return m @ ManifoldLinear(width, input_dim, tracker="mlp_out")

def LakerMLP(output_dim, input_dim, width, depth):
    m = LakerLinear(output_dim, width, tracker="mlp_in") @ ReLU()
    for i in range(depth-2):
        m = m @ LakerLinear(width, width, tracker=f"mlp_{i}") @ ReLU()
    return m @ LakerLinear(width, input_dim, tracker="mlp_out")

def Attention(num_heads, d_embed, d_query, d_value, zero_init=False, layer_idx=0):
    """Multi-head attention"""

    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed, tracker=f"v{layer_idx}")
    W = Linear(d_embed, num_heads * d_value, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax() @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, softmax_scale=None, final_scale=None, residual_scale=None, zero_init=False, wd=None):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = Attention(num_heads, d_embed, d_query, d_value, zero_init=zero_init, layer_idx=i)
        mlp = Linear(d_embed, 4*d_embed, tracker=f"mlp_out{i}") @ GeLU() @ Linear(4*d_embed, d_embed, tracker=f"mlp_in{i}")
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block

    blocks.tare(absolute=blocks_mass)

    out = Linear(vocab_size, d_embed, tracker="mlp_final")
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

def OrthogonalGPT(vocab_size, num_heads, d_embed, num_blocks, blocks_mass=5, softmax_scale=1.0, final_scale=1.0, residual_scale=1.0, wd=None):
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

def LakerAttention(num_heads, d_embed, d_query, d_value, softmax_scale=1, zero_init=True, layer_idx=0):
    """
    Attention except all the singular values are at most 1.
    """

    Q = SplitIntoHeads(num_heads) @ LakerLinear(num_heads * d_query, d_embed, tracker=f"q{layer_idx}")
    K = SplitIntoHeads(num_heads) @ LakerLinear(num_heads * d_query, d_embed, tracker=f"k{layer_idx}")
    V = SplitIntoHeads(num_heads) @ LakerLinear(num_heads * d_value, d_embed, tracker=f"v{layer_idx}")
    W = LakerLinear(d_embed, num_heads * d_value, zero_init=zero_init, tracker=f"w{layer_idx}") @ MergeHeads()

    AttentionScores = Softmax() @ Scalar(scale=softmax_scale) @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def LakerGPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, softmax_scale=1.0, final_scale=1.0, residual_scale=1.0, zero_init=True):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    blocks = Identity()
    for i in range(num_blocks):
        att = LakerAttention(num_heads, d_embed, d_query, d_value, softmax_scale, zero_init=zero_init, layer_idx=i)
        mlp = LakerLinear(d_embed, d_embed, tracker=f"mlp_out{i}") @ GeLU() @ LakerLinear(d_embed, d_embed, tracker=f"mlp_in{i}")
        att_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * att
        mlp_block = (1-residual_scale/(2*num_blocks)) * Identity() + residual_scale/(2*num_blocks) * mlp
        blocks @= mlp_block @ att_block
    
    blocks.tare(absolute=blocks_mass)

    out = Scalar(scale=final_scale, tracker="final_scale") @ LakerLinear(vocab_size, d_embed, tracker="mlp_final")

    return out @ blocks @ embed
