import torch
from torch.optim.optimizer import Optimizer

class DualAdam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, switch_rate=0.01, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, switch_rate=switch_rate,
                        weight_decay=weight_decay)
        super(DualAdam, self).__init__(params, defaults)

        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        self._step += 1

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                lr, epsilon, switch_rate, weight_decay = group['lr'], group['epsilon'], group['switch_rate'], group['weight_decay']
                inverse_adam_rate = max(0.0, 1.0 - self._step * switch_rate)

                # Adam's moment estimates
                m.mul_(beta1).add_((1.0 - beta1) * grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected moment estimates
                m_hat = m.div(1.0 - beta1**self._step)
                v_hat = v.div(1.0 - beta2**self._step)

                ada_norm = v_hat.sqrt().add(epsilon)

                if inverse_adam_rate >= switch_rate:
                    # Inverse Adam part
                    inverse_update = m_hat * ada_norm

                # Adam part
                adam_update = m_hat / ada_norm

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr, p.data)

                if inverse_adam_rate >= switch_rate:
                    # Combined update
                    p.data.add_(-lr * ((1.0 - inverse_adam_rate) * adam_update + inverse_adam_rate * inverse_update))
                else:
                    p.data.add_(-lr * adam_update)

        return loss
