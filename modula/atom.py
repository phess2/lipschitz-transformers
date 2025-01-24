import jax
import jax.numpy as jnp

from modula.abstract import Atom

def orthogonalize(M):
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

class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.scale = jnp.sqrt(self.fanout / self.fanin)
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]
        return self.scale * weights @ x, [x]

    def backward(self, w, acts, grad_output):
        weights = w[0]
        input = acts[0]
        grad_input = self.scale * weights.T @ grad_output                         # oops: self.scale appears here
        grad_weight = self.scale * grad_output @ input.T                          # oops: self.scale appears here
        return [grad_weight], grad_input

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        return self.project([weight])

    def project(self, w):
        weight = w[0]
        weight = orthogonalize(weight)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad_weight = grad_w[0]
        d_weight = orthogonalize(grad_weight)
        return [d_weight * target_norm]

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
