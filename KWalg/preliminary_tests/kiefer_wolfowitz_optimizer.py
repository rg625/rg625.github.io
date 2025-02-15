import torch
from torch.optim.optimizer import Optimizer, required
import time
import matplotlib.pyplot as plt

class KieferWolfowitzOptimizer(Optimizer):
    def __init__(self, params, lr=required, perturbation=0.05):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if perturbation < 0.0:
            raise ValueError("Invalid perturbation: {}".format(perturbation))
        
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
                perturb = torch.zeros_like(p.data)

                # Batch processing for f_plus and f_minus
                perturbations = torch.eye(p.data.numel(), device=p.device) * c_n
                perturbations = perturbations.view(-1, *p.data.shape)

                f_plus_list = []
                f_minus_list = []

                for perturbation in perturbations:
                    p.data.copy_(original_data + perturbation)
                    f_plus = closure()
                    f_plus_list.append(f_plus.item())
                
                for perturbation in perturbations:
                    p.data.copy_(original_data - perturbation)
                    f_minus = closure()
                    f_minus_list.append(f_minus.item())

                f_plus = torch.tensor(f_plus_list, device=p.device)
                f_minus = torch.tensor(f_minus_list, device=p.device)

                gradient_estimate = (f_plus - f_minus) / (2 * c_n)
                gradient_estimate = gradient_estimate.view_as(p.data)

                p.data.copy_(original_data - a_n * gradient_estimate)

                state['iteration'] += 1

        return loss

if __name__ == "__main__":
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples, n_features = 1000, 5  # Updated to 5 features for scalability demonstration
    X = torch.randn((n_samples, n_features))  # Deterministic coordinates
    true_w = torch.tensor([2.0, -3.0, 1.0, 0.5, -1.5], dtype=torch.float32)  # Updated to match n_features
    y = X @ true_w + torch.randn(n_samples) * 0.1  # Adding noise

    # Define a simple linear regression model
    model = torch.nn.Linear(n_features, 1, bias=False)
    criterion = torch.nn.MSELoss()

    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X, y = X.to(device), y.to(device).view(-1, 1)

    # Initialize the optimizer
    optimizer = KieferWolfowitzOptimizer(model.parameters(), lr=0.01, perturbation=0.05)

    # Training loop with batching
    num_epochs = 1000
    batch_size = 100
    num_batches = n_samples // batch_size

    # Performance measurement
    start_time = time.time()
    
    # Track loss and parameter updates for plotting
    losses = []
    parameter_updates = {i: [] for i in range(n_features)}

    for epoch in range(num_epochs):
        permutation = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X[indices], y[indices]

            def closure():
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                return loss

            loss = optimizer.step(closure)

        losses.append(loss.item())
        for i, param in enumerate(model.weight.data.cpu().numpy().flatten()):
            parameter_updates[i].append(param)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Print the estimated weights
    estimated_w = model.weight.data.cpu().numpy()
    print(f"Estimated weights: {estimated_w}")

    # Plot the loss over epochs
    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig('loss_over_epochs.png')
    plt.show()
    plt.close()

    fig, axs = plt.subplots(n_features, 1, figsize=(10, 5 * n_features))

    for i in range(n_features):
        axs[i].plot(range(num_epochs), parameter_updates[i], label=f'Parameter {i}')
        axs[i].axhline(y=true_w[i], color='r', linestyle='--', label=f'True Parameter {i}')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Parameter Value')
        axs[i].set_title(f'Parameter {i} Updates over Epochs')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('parameter_updates_over_epochs.png')
    plt.show()
    plt.close()