"""Lorentz (hyperboloid) model of hyperbolic space with curvature -1, and Riemannian SGD on it.

Ported from HyperLLM (https://github.com/Qin-lab-code/HyperLLM: manifolds/lmath.py, manifolds/lorentz.py,
manifolds/utils.py, optim/rsgd.py), which builds on geoopt.  Only the operations HyperLLM uses are kept, with the
same arithmetic, and the geoopt dependency is gone.

A point is a row x = (x_0, x_1, ..., x_{d-1}) with <x, x>_L = -x_0^2 + x_1^2 + ... + x_{d-1}^2 = -1: the first
coordinate is the time-like one, so a d-dimensional embedding lives on a (d-1)-dimensional hyperboloid.
"""
import torch as t
from torch import nn

EXP_MAX_NORM = 1.5  # an exponential map moves a point by at most this distance (lmath.EXP_MAX_NORM)
EPS = 1e-8


class _LeakyClamp(t.autograd.Function):
    """clamp whose gradient outside [min, max] is EPS instead of 0."""

    @staticmethod
    def forward(ctx, x, min, max):
        ctx.save_for_backward(x.ge(min) & x.le(max))
        return t.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * EPS, None, None


def _sqrt(x):
    return t.sqrt(_LeakyClamp.apply(x, 1e-9, float('inf')))


class _Acosh(t.autograd.Function):
    """arccosh(x) = log(x + sqrt(max(x^2 - 1, 1e-9))): never NaN, bounded gradient."""

    @staticmethod
    def forward(ctx, x):
        x = t.clamp(x, min=1 + EPS)
        z = t.sqrt(t.clamp(x * x - 1., min=1e-9))
        ctx.save_for_backward(z)
        return t.log(x + z)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        return grad_output / z


def acosh(x):
    return _Acosh.apply(x)


def inner(u, v, keepdim=False):
    """Minkowski inner product of matching rows."""
    uv = u * v
    spatial = uv.narrow(-1, 1, u.size(-1) - 1)
    if keepdim:
        return -uv.narrow(-1, 0, 1) + spatial.sum(dim=-1, keepdim=True)
    return -uv.narrow(-1, 0, 1).squeeze(-1) + spatial.sum(dim=-1)


def inner_multi(u, v):
    """Minkowski inner product of every row of u with every row of v: [n, m]."""
    return u @ v.transpose(0, 1) - 2 * u.narrow(-1, 0, 1) @ v.narrow(-1, 0, 1).transpose(0, 1)


def norm(u, keepdim=False):
    return _sqrt(inner(u, u, keepdim=keepdim))


def project(x):
    """Onto the hyperboloid: keep the space-like coordinates, recompute the time-like one."""
    spatial = x.narrow(-1, 1, x.size(-1) - 1)
    return t.cat((t.sqrt(1 + (spatial * spatial).sum(dim=-1, keepdim=True)), spatial), dim=-1)


def expmap0(u):
    """Exponential map at the origin (1, 0, ..., 0), followed by the projection."""
    length = norm(u, keepdim=True)
    u = u / length
    length = length.clamp_max(EXP_MAX_NORM)
    right = t.sinh(length) * u
    return project(t.cat((t.cosh(length) + right.narrow(-1, 0, 1), right.narrow(-1, 1, u.size(-1) - 1)), dim=-1))


def logmap0(y):
    """Logarithmic map at the origin: a tangent vector whose time-like coordinate is 0."""
    alpha = y.narrow(-1, 0, 1)
    origin = t.zeros(y.shape[-1], device=y.device)
    origin[0] = 1
    return acosh(alpha) / t.sqrt(alpha * alpha - 1) * (y - alpha * origin)


def sqdist(x, y):
    """Squared geodesic distance of matching rows."""
    return acosh(-inner(x, y)) ** 2


def sqdist_multi(x, y):
    """Squared geodesic distance of every row of x to every row of y: [n, m]."""
    return acosh(-inner_multi(x, y)) ** 2


# ---- operations of the optimizer ----

def egrad2rgrad(x, grad):
    """Euclidean gradient -> Riemannian gradient at x (flips the time-like coordinate of `grad` in place)."""
    grad.narrow(-1, 0, 1).mul_(-1)
    return grad.addcmul(inner(x, grad, keepdim=True), x)


def proju(x, v):
    """Onto the tangent space at x."""
    return v.addcmul(inner(x, v, keepdim=True), x)


def expmap(x, u):
    """Exponential map at x of the tangent projection of u, followed by the projection."""
    u = proju(x, u)
    length = norm(u, keepdim=True)
    u = u / length
    length = length.clamp_max(EXP_MAX_NORM)
    return project(t.cosh(length) * x + t.sinh(length) * u)


def transp(x, y, v):
    """Parallel transport of the tangent vector v from x to y."""
    return v.addcmul(inner(y, v, keepdim=True) / t.clamp_min(1 - inner(x, y, keepdim=True), 1e-7), x + y)


class ManifoldParameter(nn.Parameter):
    """A parameter whose rows are points of the hyperboloid: RiemannianSGD moves them along geodesics."""


class RiemannianSGD(t.optim.Optimizer):
    """SGD with momentum and weight decay (geoopt's RiemannianSGD, dampening 0, no Nesterov).  A ManifoldParameter
    gets the Riemannian gradient, steps through the exponential map and has its momentum parallel-transported to the
    new point; every other parameter gets the plain update."""

    def __init__(self, params, lr, momentum=0, weight_decay=0):
        if lr < 0.0 or momentum < 0.0 or weight_decay < 0.0:
            raise ValueError("lr, momentum and weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @t.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with t.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            for point in group['params']:
                grad = point.grad
                if grad is None:
                    continue
                on_manifold = isinstance(point, ManifoldParameter)
                state = self.state[point]
                if len(state) == 0 and momentum > 0:
                    state['momentum_buffer'] = grad.clone()  # the first raw gradient, as in geoopt
                grad.add_(point, alpha=weight_decay)
                if on_manifold:
                    grad = egrad2rgrad(point, grad)
                if momentum > 0:
                    buffer = state['momentum_buffer']
                    buffer.mul_(momentum).add_(grad)
                    grad = buffer
                if on_manifold:
                    new_point = expmap(point, -lr * grad)
                    if momentum > 0:
                        buffer.copy_(transp(point, new_point, buffer))
                    point.copy_(new_point)
                else:
                    point.copy_(point + (-lr * grad))
        return loss
