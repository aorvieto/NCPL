import math
import torch
from torch.optim.optimizer import Optimizer, required

#params is a list containing all parameters (each is a parameter object)

#what are param_groups? is there anything other than params? This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.

#with some magic, the lr gets saved into group['lr']. In general we have one group which contains: all parameters group['params'] (a list, each element gets updated separately) and all the hyperparameters as group['lr'], group['momentum']...

class GD(Optimizer):
    def __init__(self, params, lr): 
        defaults = dict(lr=lr)
        super(GD, self).__init__(params, defaults)
                
    def step(self, closure):
        loss = closure()
        for group in self.param_groups: #the list of all parameters, groups are layers
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha = -group['lr'])
        return loss
    
    
class HB(Optimizer):
    def __init__(self, params, lr, momentum): 
        defaults = dict(lr=lr, momentum=momentum)
        super(HB, self).__init__(params, defaults)

    def step(self, closure):
        loss = closure()
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha = 1)
                    d_p = buf
                p.data.add_(d_p, alpha = -group['lr'])
        return loss
    
class Cata(Optimizer):
    def __init__(self, params, lr, beta, P): 
        defaults = dict(lr=lr, beta=beta, P=P)
        super(Cata, self).__init__(params, defaults)
        #self.param_groups['params_z'] = copy.deepcopy(self.param_groups['params'])

    def step(self, closure):
        loss = closure()
        for group in self.param_groups:
            beta = group['beta']
            P = group['P']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'z' not in param_state:
                    param_state['z'] = torch.clone(p.data).detach()
                else:
                    zz = param_state['z']
                    zz.add_(p.data-zz, alpha = group['beta'])
                    d_p.add_(p.data-zz, alpha = P)
                p.data.add_(d_p, alpha = -group['lr'])
        return loss
    
    
class Adam(Optimizer):
    def __init__(self, params, lr, betas, eps):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure):
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    