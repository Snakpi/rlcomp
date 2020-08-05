"""
Sharing optimizer's paramters in the multiprocessors
"""

import torch
from torch.optim import Adam

class SharedAdam(Adam):
    def __init__(self, 
        params, lr=1e-3, 
        betas=(.9,.99), 
        eps=1e-6, weight_decay=0):

        super().__init__(params=params, 
                        lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()