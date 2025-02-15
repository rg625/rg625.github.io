import torch
import unittest
import matplotlib.pyplot as plt
from kifwolfoptimizer.optimizer_simple import KieferWolfowitzOptimizer

class TestKieferWolfowitzOptimizer(unittest.TestCase):
    def test_optimizer_step(self):
        model = torch.nn.Linear(10, 1)
        optimizer = KieferWolfowitzOptimizer(model.parameters(), lr=0.01, perturbation=0.05)

        def closure():
            optimizer.zero_grad()
            output = model(torch.randn(10))
            loss = (output - torch.randn(1)).pow(2).sum()
            loss.backward()
            return loss

        initial_params = [p.clone() for p in model.parameters()]
        
        # Collect losses for plotting
        losses = []
        for _ in range(100):  # Number of optimization steps
            loss = optimizer.step(closure)
            losses.append(loss.item())

        updated_params = [p for p in model.parameters()]

        for initial, updated in zip(initial_params, updated_params):
            self.assertFalse(torch.equal(initial, updated))

        # Plot the losses
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs Iteration for Kiefer-Wolfowitz Optimizer')
        plt.show()

if __name__ == '__main__':
    unittest.main()