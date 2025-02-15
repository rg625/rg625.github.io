import unittest
import torch
from kiefer_wolfowitz_optimizer import KieferWolfowitzOptimizer

class TestKieferWolfowitzOptimizer(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        torch.manual_seed(42)
        self.n_samples, self.n_features = 1000, 3
        self.X = torch.randn((self.n_samples, self.n_features))
        self.true_w = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float32)
        self.y = self.X @ self.true_w + torch.randn(self.n_samples) * 0.1

        # Define a simple linear regression model
        self.model = torch.nn.Linear(self.n_features, 1, bias=False)
        self.criterion = torch.nn.MSELoss()

        # Move model and data to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.X, self.y = self.X.to(self.device), self.y.to(self.device).view(-1, 1)

        # Initialize the optimizer
        self.optimizer = KieferWolfowitzOptimizer(self.model.parameters(), lr=0.1, perturbation=0.05)

    def test_optimizer_step(self):
        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.y)
            loss.backward()
            return loss

        initial_loss = closure().item()
        for _ in range(10):
            self.optimizer.step(closure)
        
        final_loss = closure().item()
        self.assertLess(final_loss, initial_loss, "Optimizer did not reduce the loss")

    def test_optimizer_convergence(self):
        num_epochs = 100
        batch_size = 100
        for epoch in range(num_epochs):
            permutation = torch.randperm(self.n_samples)
            for i in range(0, self.n_samples, batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = self.X[indices], self.y[indices]

                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

        estimated_w = self.model.weight.data.cpu().numpy().flatten()
        self.assertTrue(torch.allclose(torch.tensor(estimated_w), self.true_w, atol=0.1), f"Estimated weights {estimated_w} do not match true weights {self.true_w}")

if __name__ == "__main__":
    unittest.main()