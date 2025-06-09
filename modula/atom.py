import jax
import jax.numpy as jnp

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
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / (jnp.linalg.norm(M) + 1e-12)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0], dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


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


def _pure_svd(M, w_max=1):
    """Apply min(w_max, x) exactly to the singular values of a single matrix."""
    U, S, Vh = jnp.linalg.svd(M, full_matrices=False)
    S = jnp.clip(S, a_max=w_max)
    return U @ jnp.diag(S) @ Vh


def _power_iterate(A, key, num_iters=16):
    """Power iterate to find the principal singular value and vectors of M."""
    m, n = A.shape
    # Fold in dimensions to diversify the key per matrix shape
    init_key = jax.random.fold_in(key, m)
    init_key = jax.random.fold_in(init_key, n)

    if m < n:
        # iterate on AA^T to find u (shape m)
        u = jax.random.normal(init_key, (m,))
        u = u / jnp.linalg.norm(u)

        def body_u(u, _):
            w = A @ (A.T @ u)
            return w / jnp.linalg.norm(w), None

        u, _ = jax.lax.scan(body_u, u, None, length=num_iters)
        # compute sigma and v
        ATu = A.T @ u
        sigma = jnp.linalg.norm(ATu)
        v = ATu / sigma
    else:
        # iterate on A^T A to find v (shape n)
        v = jax.random.normal(init_key, (n,))
        v = v / jnp.linalg.norm(v)

        def body_v(v, _):
            w = A.T @ (A @ v)
            return w / jnp.linalg.norm(w), None

        v, _ = jax.lax.scan(body_v, v, None, length=num_iters)
        # compute sigma and u
        Av = A @ v
        sigma = jnp.linalg.norm(Av)
        u = Av / sigma

    return u, sigma, v


def _spectral_hammer(M, key, w_max=1):
    """Set the largest singular value of M to w_max."""
    u, sigma_max, v = _power_iterate(M, key)
    outer = jnp.outer(u, v)
    change = w_max - sigma_max
    return M + change * outer


def _spectral_weight_decay(M, key, spectral_wd=0.1):
    """Decay the largest singular value of M by 1 - wd."""
    u, sigma_max, v = _power_iterate(M, key)
    outer = jnp.outer(u, v)
    change = spectral_wd * sigma_max
    return M - change * outer


def _spectral_normalize(M, key):
    """Normalize the singular values of M to 1."""
    u, sigma_max, v = _power_iterate(M, key)
    return M / jnp.maximum(1, sigma_max)


def soft_cap_coupling(w_max, wd, max_update_norm):
    """Calculates the strength for soft cap that bounds singular values at w_max."""
    k = w_max * (1 - wd) + max_update_norm
    coeffs = jnp.array([-(k**9), 3 * k**7, -3 * k**5, 0, k - w_max])
    roots = jnp.roots(coeffs, strip_zeros=False)
    is_real = jnp.abs(roots.imag) < 1e-6
    is_nonnegative = roots.real >= 0
    padded_reals = jnp.where(
        is_real & is_nonnegative, roots.real, jnp.ones_like(roots.real)
    )
    return jnp.min(padded_reals)


# Define batch versions of the project functions (as functions so they can be imported)
def orthogonalize(M, **kwargs):
    return batch_project(M, _orthogonalize)


def hard_cap(M, **kwargs):
    return batch_project(M, _hard_cap)


def soft_cap(M, alpha, **kwargs):
    return batch_project(M, lambda x: _soft_cap(x, alpha=alpha))


def pure_svd(M, w_max=1, **kwargs):
    return batch_project(M, lambda x: _pure_svd(x, w_max))


def spectral_hammer(M, key, w_max=1, **kwargs):
    return batch_project(M, lambda x: _spectral_hammer(x, key, w_max))


def spectral_weight_decay(M, key, spectral_wd=0.1, **kwargs):
    return batch_project(M, lambda x: _spectral_weight_decay(x, key, spectral_wd))


def spectral_normalize(M, key, **kwargs):
    return batch_project(M, lambda x: _spectral_normalize(x, key))


# Embed
def _embed_project(M, axis, max_inflation_factor):
    """RMS normalize the rows of M, then clip at max_inflation_factor. M is [d_embed, num_embed]."""
    rmsnorm_of_rows = jnp.linalg.norm(M, axis=axis, keepdims=True) / jnp.sqrt(
        M.shape[axis]
    )
    M = M / jnp.maximum(1 / max_inflation_factor, rmsnorm_of_rows)
    return M


def embed_project(M, max_inflation_factor, **kwargs):
    return batch_project(M, lambda x: _embed_project(x, -2, max_inflation_factor))


def unembed_project(M, max_inflation_factor, **kwargs):
    return batch_project(M, lambda x: _embed_project(x, -1, max_inflation_factor))


class Linear(Atom):
    def __init__(
        self,
        fanout,
        fanin,
        dtype=jnp.float32,
        project_dtype=None,
        zero_init=False,
        project=None,
        sensitive_to_wmax=None,
        tracker=None,
    ):
        super().__init__(tracker)
        self.fanin = fanin
        self.fanout = fanout
        self.dtype = dtype
        self.project_dtype = project_dtype
        self.zero_init = zero_init
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

        self._project = lambda x, **kwargs: x
        self.sensitive_to_wmax = (
            sensitive_to_wmax["default"] if sensitive_to_wmax is not None else False
        )
        if project is not None:
            if tracker in project:
                self._project = project[tracker]
            elif "default" in project:
                self._project = project["default"]

    def forward(self, x, w):
        weights = w[0]
        return x @ weights.transpose()

    def orthogonalize(self, w):
        weight = w[0]
        return [orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)]

    def initialize(self, key):
        if self.tracker is not None:
            self.log_info = {}
        if self.zero_init:
            return [jnp.zeros((self.fanout, self.fanin), dtype=self.dtype)]
        weight = jax.random.normal(
            key, shape=(self.fanout, self.fanin), dtype=self.dtype
        )
        return self.orthogonalize([weight])

    def project(
        self, w, w_max=1.0, wd=0.0, spectral_wd=0.0, max_update_norm=1.0, key=None
    ):
        weight = w[0]
        casted = weight.astype(self.project_dtype)
        scale = jnp.sqrt(self.fanout / self.fanin)
        # max_update_norm is correct in the RMS->RMS induced norm,
        # but we divide by scale to account for the effect it will have on casted / scale
        alpha = soft_cap_coupling(
            w_max, wd, max_update_norm / scale
        )  # only some proj functions use this
        if self.sensitive_to_wmax:
            scale *= w_max
        projected = scale * self._project(
            casted / scale, spectral_wd=spectral_wd, alpha=alpha, w_max=w_max, key=key
        )
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
        self.log_info["weight_norm"].append(
            (fan_in / fan_out) ** 0.5 * jnp.linalg.norm(w[0].astype(jnp.float32), ord=2)
        )

        if "raw_grad_norm" not in self.log_info:
            self.log_info["raw_grad_norm"] = []
        self.log_info["raw_grad_norm"].append(
            jnp.linalg.norm(grad_w[0].astype(jnp.float32), ord=2)
        )

        if "spectral_norm" not in self.log_info:
            self.log_info["spectral_norm"] = []
        self.log_info["spectral_norm"].append(jnp.linalg.svd(w[0], compute_uv=False)[0])
        return {self.tracker: self.log_info}


class Embed(Atom):
    def __init__(
        self,
        d_embed,
        num_embed,
        dtype=jnp.float32,
        max_inflation_factor=1,
        tracker=None,
    ):
        super().__init__(tracker)
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.max_inflation_factor = max_inflation_factor
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]
        embed = weights[:, x]
        return jnp.moveaxis(embed, 0, -1)

    def initialize(self, key):
        weight = jax.random.normal(
            key, shape=(self.d_embed, self.num_embed), dtype=self.dtype
        )
        weight = embed_project(
            weight, max_inflation_factor=1e9
        )  # always send to norm 1
        return [weight]

    def project(self, w, **kwargs):
        weight = w[0]
        weight = embed_project(weight, max_inflation_factor=1)  # allow decaying to zero
        return [weight]

    def dualize(self, grad_w, w=None, target_norm=1.0):
        d_weight = grad_w[0]
        d_weight = embed_project(
            d_weight, max_inflation_factor=self.max_inflation_factor
        )
        return [d_weight * target_norm]

    def log(self, w, grad_w):
        if self.tracker is None:
            return {}

        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []
        self.log_info["weight_norm"].append(
            jnp.max(jnp.linalg.norm(w[0], axis=0, keepdims=True))
            / jnp.sqrt(self.d_embed)
        )

        return {self.tracker: self.log_info}


class Unembed(Atom):
    def __init__(
        self,
        d_embed,
        num_embed,
        dtype=jnp.float32,
        max_inflation_factor=1,
        zero_init=False,
        tracker=None,
    ):
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
        weights = w[0]
        return x @ weights.transpose()

    def initialize(self, key):
        if self.zero_init:
            return [jnp.zeros((self.num_embed, self.d_embed), dtype=self.dtype)]
        weight = jax.random.normal(
            key, shape=(self.num_embed, self.d_embed), dtype=self.dtype
        )
        weight = unembed_project(weight, max_inflation_factor=1e9)
        return [weight]

    def project(self, w, **kwargs):
        weight = w[0]
        weight = unembed_project(
            weight, max_inflation_factor=1
        )  # allow decaying to zero
        return [weight]

    def dualize(self, grad_w, w=None, target_norm=1.0):
        d_weight = grad_w[0]
        d_weight = unembed_project(
            d_weight, max_inflation_factor=self.max_inflation_factor
        )
        return [d_weight * target_norm]

    def log(self, w, grad_w):
        if self.tracker is None:
            return {}

        if "weight_norm" not in self.log_info:
            self.log_info["weight_norm"] = []
        self.log_info["weight_norm"].append(
            jnp.max(jnp.linalg.norm(w[0], axis=1, keepdims=True))
            / jnp.sqrt(self.d_embed)
        )

        return {self.tracker: self.log_info}
