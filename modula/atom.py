import jax
import jax.numpy as jnp

from modula.abstract import Atom

def _orthogonalize(M):
    """Orthogonalize a single matrix."""
    a, b, c = 3.0, -3.2, 1.2
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for _ in range(10):
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

def orthogonalize(M):
    """Batch orthogonalize tensors of shape [..., fanout, fanin]."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape((-1,) + matrix_shape)
    M_orthogonalized = jax.vmap(_orthogonalize)(M_flattened)
    return M_orthogonalized.reshape(M.shape) / len(M_flattened)


class Linear(Atom):
    def __init__(self, fanout, fanin, tracker=None):
        super().__init__(tracker)
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return x @ weights.transpose()  # shape is [..., fanout]

    def initialize(self, key):
        if self.tracker is not None:
            self.log_info = {}
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
        return [d_weight]
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []

        self.log_info["weight_norm"].append(jnp.linalg.norm(w[0], ord=2))
        return {self.tracker: self.log_info}


class HeadedLinear(Linear):
    """Rank-3 tensor version of Linear so that dualize batches over the head dimension."""
    def __init__(self, num_heads, fanout, fanin, tracker=None):
        super().__init__(fanout, fanin, tracker)
        self.fanout = fanout
        self.fanin = fanin
        self.num_heads = num_heads
    
    def forward(self, x, w):
        # x is shape [...fanin]
        # w[0] is shape [heads, fanout, fanin]
        # output is shape [...heads, fanout]
        return jnp.einsum("...i, h o i -> ...h o", x, w[0])

    def initialize(self, key):
        if self.tracker is not None:
            self.log_info = {}
        weight = jax.random.normal(key, shape=(self.num_heads, self.fanout, self.fanin))
        return self.project([weight])
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []

        max_norm = max([jnp.linalg.norm(w[0][i], ord=2) for i in range(self.num_heads)])
        self.log_info["weight_norm"].append(max_norm * self.num_heads)
        return {self.tracker: self.log_info}

class HeadedLinearOut(HeadedLinear):
    def __init__(self, num_heads, fanout, fanin, tracker=None):
        super().__init__(num_heads, fanout, fanin, tracker)

    def forward(self, x, w):
        # x is shape [...heads, fanin]
        # w is shape [heads, fanout, fanin]
        # output is shape [...heads, fanout]
        return jnp.einsum("...h i, h o i -> ...h o", x, w[0])
        


def sr_sinkhorn(g, steps=5):
    """
    Implementation of the Square-Root Sinkhorn algorithm,
    Algorithm 3 in https://arxiv.org/abs/2502.06742v1
    """
    X = g
    m, n = X.shape

    for _ in range(steps):
        # Row-wise normalization
        row_norms = jnp.linalg.norm(X, axis=1, keepdims=True)  # [m, 1]
        row_norms = jnp.clip(row_norms, a_min=1e-8)
        X = n**0.5 * X / row_norms

        # Column-wise normalization
        col_norms = jnp.linalg.norm(X, axis=0, keepdims=True)  # [1, n]
        col_norms = jnp.clip(col_norms, a_min=1e-8)
        X = m**0.5 * X / col_norms

    return X

class SinkhornLinear(Linear):
    def __init__(self, fanout, fanin, tracker=None):
        super().__init__(fanout, fanin, tracker)
        self.smooth = False

    def dualize(self, grad_w, target_norm=1.0):
        weight = sr_sinkhorn(grad_w[0]) * target_norm
        return [weight]

class Embed(Atom):
    def __init__(self, d_embed, num_embed, tracker=None):
        super().__init__(tracker)
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [d_embed, num_embed]
        embed = weights[:, x]  # shape [d_embed, x_shape]
        return jnp.moveaxis(embed, 0, -1) # shape [x_shape, d_embed]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.d_embed, self.num_embed))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = weight / jnp.linalg.norm(weight, axis=0, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        d_weight = self.project(grad_w)[0] * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]
    
    def log(self, w, grad_w):
        return {}

class Scalar(Atom):
    def __init__(self, tracker=None):
        super().__init__(tracker)
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
        
    def forward(self, x, w):
        return x * w[0]
    
    def initialize(self, key):
        return [jnp.ones(1)]
    
    def project(self, w):
        return [w[0]]
    
    def dualize(self, grad_w, target_norm=1.0):
        return [grad_w[0] * target_norm]
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "scalar" not in self.log_info:
            self.log_info["scalar"] = []

        self.log_info["scalar"].append(w[0])
        return {self.tracker: self.log_info}

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # sample a random d0xd1 matrix
    d0, d1 = 50, 100
    M = jax.random.normal(key, shape=(d0, d1))
    O = orthogonalize(M)

    # compute SVD of M and O
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    s = jnp.linalg.svd(O, compute_uv=False)

    # print singular values
    print(f"min singular value of O: {jnp.min(s)}")
    print(f"max singular value of O: {jnp.max(s)}")

    # check that M is close to its SVD
    error_M = jnp.linalg.norm(M - U @ jnp.diag(S) @ Vh) / jnp.linalg.norm(M)
    error_O = jnp.linalg.norm(O - U @ Vh) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in M's SVD: {error_M}")
    print(f"relative error in O: {error_O}")

    # Test batched orthogonalization
    batch, heads = 3, 4
    batched_M = jax.random.normal(key, shape=(batch, heads, d0, d1))
    batched_O = orthogonalize(batched_M)
    
    # Check shape preservation
    assert batched_O.shape == batched_M.shape
    print(f"Batched shape preserved: {batched_O.shape}")
    
    # Check orthogonality for each matrix in the batch
    for b in range(batch):
        for h in range(heads):
            matrix = batched_O[b, h]
            if matrix.shape[0] <= matrix.shape[1]:  # If tall matrix
                ortho_check = matrix @ matrix.T
                target = jnp.eye(matrix.shape[0])
            else:  # If wide matrix
                ortho_check = matrix.T @ matrix
                target = jnp.eye(matrix.shape[1])
            
            error = jnp.linalg.norm(ortho_check - target) / jnp.linalg.norm(target)
            if error > 1e-2:
                # the typical error is 2e-3
                print(f"Orthogonality error for batch {b}, head {h}: {error}")
