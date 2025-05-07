import jax
import jax.numpy as jnp
#jax.config.update("jax_enable_x64", True)

from modula.abstract import Atom

def batch_project(M, project_fn):
    """Batch project tensors of shape [..., fanout, fanin]."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape((-1,) + matrix_shape)
    M_projected = jax.vmap(project_fn)(M_flattened)
    return M_projected.reshape(M.shape) / len(M_flattened)

def _orthogonalize(M):
    """Orthogonalize a single matrix, always bfloat16. Credit for coefficients to @YouJiacheng and @leloykun."""
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    original_dtype = M.dtype
    M = M.astype(jnp.bfloat16)
    M = M / (jnp.linalg.norm(M) + 1e-12)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0], dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M.astype(original_dtype)

def _hard_cap(M):
    """Apply min(1, x) approximately to the singular values of a single matrix. Credit: Franz Cecista."""
    coeffs = [
        (0.805032, 0.206361, -0.019763),
        (0.649867, 0.162935, -0.011150),
        (1.810259, -0.200265, 0.008251),
        (1.004384, -0.183490, 0.014413),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    for a, b, c in coeffs:
        A = M.T @ M
        I = jnp.eye(A.shape[0], dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

def _soft_cap(M, alpha):
    """Apply min(1, x) approximately to the singular values of a single matrix."""
    coeffs = [
        (1, -alpha),
        (1, alpha),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    for a, b in coeffs:
        A = M.T @ M
        I = jnp.eye(A.shape[0], dtype=M.dtype)
        M = M @ (a * I + b * A)
    if transpose:
        M = M.T
    return M

def _pure_svd(M):
    """Apply min(1, x) exactly to the singular values of a single matrix."""
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    S = jnp.clip(S, a_max=1)
    return U @ jnp.diag(S) @ Vh

def soft_cap_coupling(w_max, wd, max_update_norm):
    """Calculates the strength for soft cap that bounds singular values at w_max."""
    k = w_max * (1 - wd) + max_update_norm
    coeffs = jnp.array([-9 * k**9, 3 * k**7, -3 * k**5, 0, k - w_max])
    roots = jnp.roots(coeffs, strip_zeros=False)
    is_real = jnp.abs(roots.imag) < 1e-6
    is_nonnegative = roots.real >= 0
    padded_reals = jnp.where(is_real & is_nonnegative, roots.real, jnp.ones_like(roots.real))
    return jnp.min(padded_reals)

# Define batch versions of the project functions (as functions so they can be imported)
def orthogonalize(M, **kwargs):
    return batch_project(M, _orthogonalize)
def hard_cap(M, **kwargs):
    return batch_project(M, _hard_cap)
def soft_cap(M, alpha):
    return batch_project(M, lambda x: _soft_cap(x, alpha=alpha))
def soft_cap1(M, **kwargs):
    return batch_project(M, lambda x: _soft_cap(x, alpha=0.002))
def soft_cap2(M, **kwargs):
    return batch_project(M, lambda x: _soft_cap(x, alpha=0.05))
def soft_cap3(M, **kwargs):
    return batch_project(M, lambda x: _soft_cap(x, alpha=0.1))
def pure_svd(M, **kwargs):
    return batch_project(M, _pure_svd)

# Embed
def _embed_project(M, axis, max_inflation_factor):
    """RMS normalize the rows of M, then clip at max_inflation_factor. M is [d_embed, num_embed]."""
    rmsnorm_of_rows = jnp.linalg.norm(M, axis=axis, keepdims=True) / jnp.sqrt(M.shape[axis])
    M = M / jnp.maximum(1 / max_inflation_factor, rmsnorm_of_rows)
    return M

def embed_project(M, max_inflation_factor, **kwargs):
    return batch_project(M, lambda x: _embed_project(x, -2, max_inflation_factor))
def unembed_project(M, max_inflation_factor, **kwargs):
    return batch_project(M, lambda x: _embed_project(x, -1, max_inflation_factor))

class Linear(Atom):
    def __init__(self, fanout, fanin, dtype=jnp.float32, project_dtype=None, zero_init=False, project=None, tracker=None):
        super().__init__(tracker)
        self.fanin  = fanin
        self.fanout = fanout
        self.dtype = dtype
        self.project_dtype = project_dtype
        self.zero_init = zero_init
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

        self._project = lambda x, **kwargs: x
        if project is not None:
            if tracker in project:  # project is a dictionary mapping tracker names to projection functions
                self._project = project[tracker]
            elif "default" in project:
                self._project = project["default"]

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return x @ weights.transpose()  # shape is [..., fanout]

    def orthogonalize(self, w):
        weight = w[0]
        return [orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)]

    def initialize(self, key):
        if self.tracker is not None:
            self.log_info = {}
        if self.zero_init:
            return [jnp.zeros((self.fanout, self.fanin), dtype=self.dtype)]
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin), dtype=self.dtype)
        return self.orthogonalize([weight])
    
    def project(self, w, w_max=1.0, wd=0.0, max_update_norm=1.0):
        weight = w[0]
        casted = weight.astype(self.project_dtype)
        scale = jnp.sqrt(self.fanout / self.fanin)
        # max_update_norm is correct in the RMS->RMS induced norm,
        # but we divide by scale to account for the effect it will have on casted / scale
        alpha = soft_cap_coupling(w_max, wd, max_update_norm / scale)  # only some proj functions use this
        projected = scale * self._project(casted / scale, alpha=alpha)
        return [projected.astype(self.dtype)]

    def dualize(self, grad_w, w=None, target_norm=1.0):
        d_weight = self.orthogonalize(grad_w)[0]
        return [d_weight * target_norm]
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []
        fan_out, fan_in = w[0].shape
        self.log_info["weight_norm"].append((fan_in/fan_out)**0.5 * jnp.linalg.norm(w[0].astype(jnp.float32), ord=2))

        if "raw_grad_norm" not in self.log_info:
            self.log_info["raw_grad_norm"] = []
        self.log_info["raw_grad_norm"].append(jnp.linalg.norm(grad_w[0].astype(jnp.float32), ord=2))

        
        if "cos_angle_w_with_d_w" not in self.log_info:
            self.log_info["cos_angle_w_with_d_w"] = []
        self.log_info["cos_angle_w_with_d_w"].append(
            jnp.sum(w[0].flatten() * grad_w[0].flatten()) / (jnp.linalg.norm(w[0]) * jnp.linalg.norm(grad_w[0]))
        )
        return {self.tracker: self.log_info}


class Embed(Atom):
    def __init__(self, d_embed, num_embed, dtype=jnp.float32, max_inflation_factor=1, tracker=None):
        super().__init__(tracker)
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.max_inflation_factor = max_inflation_factor
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [d_embed, num_embed]
        embed = weights[:, x]  # shape [d_embed, x_shape]
        return jnp.moveaxis(embed, 0, -1) # shape [x_shape, d_embed]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.d_embed, self.num_embed), dtype=self.dtype)
        #weight = embed_project(weight, max_inflation_factor=1e9)  # always send to norm 1
        return [weight]
    
    def project(self, w, **kwargs):
        weight = w[0]
        #weight = embed_project(weight, max_inflation_factor=1)  # allow decaying to zero
        return [weight]

    def dualize(self, grad_w, w=None, target_norm=1.0):
        d_weight = grad_w[0]
        #d_weight = embed_project(d_weight, max_inflation_factor=self.max_inflation_factor)
        return [d_weight * target_norm]
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []
        self.log_info["weight_norm"].append(jnp.max(jnp.linalg.norm(w[0], axis=0, keepdims=True)) / jnp.sqrt(self.d_embed))
            
        return {self.tracker: self.log_info}


class Unembed(Atom):
    def __init__(self, d_embed, num_embed, dtype=jnp.float32, max_inflation_factor=1, zero_init=False, tracker=None):
        super().__init__(tracker)
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.max_inflation_factor = max_inflation_factor
        self.zero_init = zero_init
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1
        self.dtype = dtype

    def forward(self, x, w):
        weights = w[0]  # shape [d_embed, num_embed]
        return x @ weights.transpose()  # shape [..., num_embed]

    def initialize(self, key):
        if self.zero_init:
            return [jnp.zeros((self.num_embed, self.d_embed), dtype=self.dtype)]
        weight = jax.random.normal(key, shape=(self.num_embed, self.d_embed), dtype=self.dtype)
        weight = embed_project(weight, max_inflation_factor=1e9)
        return [weight]

    def project(self, w, **kwargs):
        weight = w[0]
        weight = embed_project(weight, max_inflation_factor=1)  # allow decaying to zero
        return [weight]

    def dualize(self, grad_w, w=None, target_norm=1.0):
        d_weight = grad_w[0]
        d_weight = embed_project(d_weight, max_inflation_factor=self.max_inflation_factor)
        return [d_weight * target_norm]
    
    def log(self, w, grad_w):
        if self.tracker is None:
            return {}
        
        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []
        self.log_info["weight_norm"].append(jnp.max(jnp.linalg.norm(w[0], axis=1, keepdims=True)) / jnp.sqrt(self.d_embed))
        
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
