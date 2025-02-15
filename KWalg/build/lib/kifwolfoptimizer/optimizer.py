import torch
from torch.optim import Optimizer

class KieferWolfowitzOptimizer(Optimizer):
    def __init__(self, params, lr=None, perturbation=0.05):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if perturbation < 0.0:
            raise ValueError("Invalid perturbation: {}".format(perturbation))
        
        if lr is None:
            raise ValueError("Learning rate is required")
        
        defaults = dict(lr=lr, perturbation=perturbation)
        super(KieferWolfowitzOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                lr = group['lr']
                perturbation = group['perturbation']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['iteration'] = 0
                    state['gradient_estimate'] = torch.zeros_like(p.data)

                iteration = state['iteration']
                gradient_estimate = state['gradient_estimate']

                c_n = perturbation / (iteration + 1) ** 0.5  # Decaying perturbation
                a_n = lr / (iteration + 1) ** 0.5  # Adjusted decay rate for step size

                original_data = p.data.clone()

                f_plus_list = []
                f_minus_list = []

                # Batch processing for f_plus and f_minus
                for i in range(p.data.numel()):
                    perturbation_vector = torch.zeros_like(p.data).view(-1)
                    perturbation_vector[i] = c_n
                    perturbation_vector = perturbation_vector.view_as(p.data)

                    p.data.copy_(original_data + perturbation_vector)
                    f_plus = closure()
                    f_plus_list.append(f_plus.item())

                    p.data.copy_(original_data - perturbation_vector)
                    f_minus = closure()
                    f_minus_list.append(f_minus.item())

                f_plus = torch.tensor(f_plus_list, device=p.device)
                f_minus = torch.tensor(f_minus_list, device=p.device)

                gradient_estimate = (f_plus - f_minus) / (2 * c_n)
                gradient_estimate = gradient_estimate.view_as(p.data)

                p.data.copy_(original_data - a_n * gradient_estimate)

                state['iteration'] += 1

        return loss