import jax
import copy

class Module:
    def __init__(self, tracker=None):
        self.children = []
        self.log_info = {}
        self.tracker = tracker      # optional tracker label

        self.atoms = None           # number of atoms: int
        self.bonds = None           # number of bonds: int
        self.smooth = None          # is this module smooth?: bool
        self.sensitivity = None     # input Lipschitz estimate: float > 0
        self.mass = None            # proportional contribution of module toward feature learning of any supermodule: float >= 0

    def __str__(self):
        string = self.__class__.__name__
        string += f"\n...consists of {self.atoms} atoms and {self.bonds} bonds"
        string += f"\n...{'smooth' if self.smooth else 'non-smooth'}"
        string += f"\n...input sensitivity is {self.sensitivity}"
        string += f"\n...contributes proportion {self.mass} to feature learning of any supermodule"
        return string

    def tare(self, absolute=1.0, relative=None):
        if relative is None:
            self.tare(relative = absolute / self.mass)
        else:
            self.mass *= relative
            for m in self.children:
                m.tare(relative = relative)

    def jit(self):
        self.forward = jax.jit(self.forward)
        self.project = jax.jit(self.project)
        self.dualize = jax.jit(self.dualize)

    def forward(self, x, w):
        # Input and weight list --> output
        raise NotImplementedError

    def initialize(self, key):
        # Return a weight list.
        raise NotImplementedError

    def project(self, w, lr=1.0):
        # Return a weight list.
        raise NotImplementedError

    def dualize(self, grad_w, w=None, target_norm=1.0):
        # Weight gradient list, weight list, and number --> dualized weight gradient list
        raise NotImplementedError
    
    def step(self, w, d_w, lr):
        # Weight list, update list, learning rate --> updated weight list
        raise NotImplementedError
    
    def log(self, w, grad_w):
        # Measure and return observables -- mutates self.log_info
        raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, tuple):
            other = TupleModule(other)
        return CompositeModule(self, other)

    def __add__(self, other):
        return Add() @ TupleModule((self, other))

    def __mul__(self, other):
        assert other != 0, "cannot multiply a module by zero"
        return self @ Mul(other)

    def __rmul__(self, scalar):
        return Mul(scalar) @ self

    def __pow__(self, n):
        assert n >= 0 and n % 1 == 0, "nonnegative integer powers only"
        return copy.deepcopy(self) @ (self ** (n-1)) if n > 0 else Identity()

    def __call__(self, x, w):
        return self.forward(x, w)

class Atom(Module):
    def __init__(self, tracker=None):
        super().__init__(tracker)
        self.atoms = 1
        self.bonds = 0
    
    def step(self, w, d_w, lr):
        return [w[0] - lr * d_w[0]]

class Bond(Module):
    def __init__(self):
        super().__init__()
        self.atoms = 0
        self.bonds = 1
        self.mass = 0

    def initialize(self, key):
        return []

    def project(self, w, lr=1.0):
        return []

    def dualize(self, grad_w, w=None, target_norm=1.0):
        return []
    
    def step(self, w, d_w, lr):
        return []
    
    def log(self, w, grad_w):
        return {}

class CompositeModule(Module):
    def __init__(self, m1, m0):
        super().__init__()
        self.children = (m0, m1)

        self.atoms       = m0.atoms + m1.atoms
        self.bonds       = m0.bonds + m1.bonds
        self.smooth      = m0.smooth and m1.smooth
        self.mass        = m0.mass + m1.mass
        self.sensitivity = m0.sensitivity * m1.sensitivity

    def forward(self, x, w):
        m0, m1 = self.children
        w0 = w[:m0.atoms]
        w1 = w[m0.atoms:]
        x0 = m0.forward(x, w0)
        x1 = m1.forward(x0, w1)
        return x1

    def initialize(self, key):
        m0, m1 = self.children
        key, subkey = jax.random.split(key)
        return m0.initialize(key) + m1.initialize(subkey)

    def project(self, w, lr=1.0):
        m0, m1 = self.children
        w0 = w[:m0.atoms]
        w1 = w[m0.atoms:]
        return m0.project(w0, lr) + m1.project(w1, lr)

    def dualize(self, grad_w, w=None, target_norm=1.0):
        if self.mass > 0:
            m0, m1 = self.children
            grad_w0, grad_w1 = grad_w[:m0.atoms], grad_w[m0.atoms:]
            w0 = w[:m0.atoms] if w is not None else None
            w1 = w[m0.atoms:] if w is not None else None
            d_w0 = m0.dualize(grad_w0, w0, target_norm = target_norm * m0.mass / self.mass / m1.sensitivity)
            d_w1 = m1.dualize(grad_w1, w1, target_norm = target_norm * m1.mass / self.mass)
            d_w = d_w0 + d_w1
        else:
            d_w = [0 * grad_weight for grad_weight in grad_w]
        return d_w
    
    def step(self, w, d_w, lr):
        if self.mass == 0:
            return w
        m0, m1 = self.children
        w0, w1 = w[:m0.atoms], w[m0.atoms:]
        d_w0, d_w1 = d_w[:m0.atoms], d_w[m0.atoms:]
        step0 = m0.step(w0, d_w0, lr)
        step1 = m1.step(w1, d_w1, lr)
        return step0 + step1

    def log(self, w, grad_w):
        m0, m1 = self.children
        w0 = w[:m0.atoms]
        w1 = w[m0.atoms:]
        grad_w0 = grad_w[:m0.atoms]
        grad_w1 = grad_w[m0.atoms:]
        return m0.log(w0, grad_w0) | m1.log(w1, grad_w1)

class TupleModule(Module):
    def __init__(self, python_tuple_of_modules):
        super().__init__()
        self.children = python_tuple_of_modules
        self.atoms       = sum(m.atoms       for m in self.children)
        self.bonds       = sum(m.bonds       for m in self.children)
        self.smooth      = all(m.smooth      for m in self.children)
        self.mass        = sum(m.mass        for m in self.children)
        self.sensitivity = sum(m.sensitivity for m in self.children)

    def forward(self, x, w):
        output_list = []
        for m in self.children:
            output = m.forward(x, w[:m.atoms])
            output_list.append(output)
            w = w[m.atoms:]
        return output_list

    def initialize(self, key):
        w = []
        for m in self.children:
            key, subkey = jax.random.split(key)
            w += m.initialize(subkey)
        return w

    def project(self, w, lr=1.0):
        projected_w = []
        for m in self.children:
            projected_w_m = m.project(w[:m.atoms], lr)
            projected_w += projected_w_m
            w = w[m.atoms:]
        return projected_w

    def dualize(self, grad_w, w=None, target_norm=1.0):
        if self.mass > 0:
            d_w = []
            for m in self.children:
                grad_w_m = grad_w[:m.atoms]
                w_m = w[:m.atoms] if w is not None else None
                d_w_m = m.dualize(grad_w_m, w_m, target_norm = target_norm * m.mass / self.mass)
                d_w += d_w_m
                grad_w = grad_w[m.atoms:]
                w = w[m.atoms:] if w is not None else None
        else:
            d_w = [0 * grad_weight for grad_weight in grad_w]
        return d_w
    
    def step(self, w, d_w, lr):
        if self.mass == 0:
            return w
        steps = []
        for m in self.children:
            w_m = w[:m.atoms]
            d_w_m = d_w[:m.atoms]
            step_m = m.step(w_m, d_w_m, lr)
            steps += step_m
            w = w[m.atoms:]
            d_w = d_w[m.atoms:]
        return steps
    
    def log(self, w, grad_w):
        log_info = {}
        for m in self.children:
            log_info |= m.log(w, grad_w)
            w = w[m.atoms:]
            grad_w = grad_w[m.atoms:]
        return log_info

class Identity(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return x

class Add(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, x, w):
        return sum(x)

class Mul(Bond):
    def __init__(self, scalar):
        super().__init__()
        self.smooth = True
        self.sensitivity = scalar

    def forward(self, x, w):
        return x * self.sensitivity
