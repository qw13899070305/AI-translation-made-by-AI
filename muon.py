# muon.py —— Muon 优化器 (参考 Keller Jordan 实现)
import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.float32)
    X = X / (X.norm() + eps)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ A
        X = a * X + b * (X @ A) + c * (X @ B)
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adam_wd=0.01, muon_wd=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adam_wd=adam_wd, muon_wd=muon_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, nesterov = group['lr'], group['momentum'], group['nesterov']
            ns_steps, adam_wd, muon_wd = group['ns_steps'], group['adam_wd'], group['muon_wd']

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)

                state['step'] += 1
                buf = state['momentum_buffer']

                if p.ndim >= 2 and muon_wd > 0:
                    p.mul_(1 - lr * muon_wd)

                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                if p.ndim >= 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update *= 0.2 * max(p.size(0), p.size(1)) ** 0.5
                    p.add_(update, alpha=-lr)
                else:
                    if adam_wd > 0:
                        p.mul_(1 - lr * adam_wd)
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    exp_avg_sq = state['exp_avg_sq']
                    beta2 = 0.95
                    exp_avg_sq.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    p.addcdiv_(update, denom, value=-lr)
        return loss