import jax
import jax.numpy as jnp

class Optimizer:
    """Base class for optimizers with standardized interface."""
    def __init__(self, config):
        self.config = config
        self.step_count = 0
    
    def init_state(self, params):
        """Initialize optimizer state."""
        raise NotImplementedError
    
    def update(self, params, grads, state):
        """Update parameters using gradients."""
        raise NotImplementedError

class Muon(Optimizer):
    def init_state(self, params):
        return jax.tree.map(jnp.zeros_like, params)
    
    def update(self, params, grads, state):
        buf = jax.tree.map(lambda m, g: self.config.beta1 * m + (1-self.config.beta1) * g, 
                          state, grads)
        # Calculate parameter updates (momentum)
        d_params = buf  # In Muon, the buffer itself represents the parameter updates
        return params, buf, d_params

class Adam(Optimizer):
    def init_state(self, params):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return (m, v)
    
    def update(self, params, grads, state):
        m, v = state
        m_new = jax.tree.map(lambda m, g: self.config.beta1 * m + (1-self.config.beta1) * g, 
                            m, grads)
        v_new = jax.tree.map(lambda v, g: self.config.beta2 * v + (1-self.config.beta2) * g**2, 
                            v, grads)
        d_params = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + 1e-12), m_new, v_new)
        return params, (m_new, v_new), d_params

def get_optimizer(config):
    """Factory function to create an optimizer instance."""
    if config.optimizer == "muon":
        return Muon(config)
    elif config.optimizer == "adam":
        return Adam(config)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
