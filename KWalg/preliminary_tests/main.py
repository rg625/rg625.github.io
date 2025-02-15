import torch
import time
import psutil
import plotly.graph_objects as go
from scipy.stats import wasserstein_distance
import json

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a 1D Gaussian Mixture Model (GMM)
def log_prob(x):
    """Compute log probability of a Gaussian mixture model."""
    x = x.to(device).requires_grad_(True)
    p1 = torch.exp(-0.5 * ((x - 2) / 0.8) ** 2) / (0.8 * (2 * torch.pi) ** 0.5)
    p2 = torch.exp(-0.5 * ((x + 2) / 0.8) ** 2) / (0.8 * (2 * torch.pi) ** 0.5)
    return torch.log(0.5 * p1 + 0.5 * p2 + 1e-9)  # Small constant for numerical stability

# Compute the true score function using autograd
def true_score(x, step, step_size):
    x = x.to(device)
    log_p = log_prob(x)
    grad = step_size * torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
    return grad.detach(), step_size

# Estimate the score function using finite differences
def estimate_score(x, step, step_size, delta=5e-2):
    x = x.to(device)
    step_size = (1 + step_size) ** 0.6
    delta = step_size * torch.tensor(delta / (step + 1) ** 0.5, device=device)  # Adaptive step size
    return (log_prob(x + delta) - log_prob(x - delta)) / (2 * delta), step_size

# SPSA Gradient Estimation
def spsa_gradient(x, step, step_size, delta=5e-2):
    x = x.to(device)
    step_size = (1 + step_size) ** 0.6
    perturbation = torch.empty_like(x).uniform_(-1, 1).sign()
    delta = step_size * torch.tensor(delta / (step + 1) ** 0.5, device=device)  # Adaptive step size
    x_plus = x + delta * perturbation
    x_minus = x - delta * perturbation
    gradient_estimate = (log_prob(x_plus) - log_prob(x_minus)) / (2 * delta * perturbation)
    return gradient_estimate, step_size

# Measure memory usage on CUDA
def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        return torch.cuda.max_memory_allocated() / (1024**2)  # Peak memory in MB
    else:
        return psutil.Process().memory_info().rss / (1024**2)  # CPU memory usage

# Langevin Dynamics sampler with better exploration
def langevin_dynamics(x0_list, score_function, steps=100, eta=0.5, noise_scale=0.7):
    """Perform Langevin Dynamics using a specified score function, with multiple initial points."""
    x = x0_list.to(device)
    samples = []

    for i in range(steps):
        score, step_size = score_function(x, i, eta)  # Compute score function
        noise = torch.randn_like(x, device=device) * noise_scale  # Reduce noise over time
        x = x + score + torch.sqrt(torch.tensor(2 * step_size, device=device)) * noise
        samples.append(x.clone().detach())

    return torch.cat(samples)

# Hamiltonian Monte Carlo sampler without Metropolis step
def hmc_sampler(x0_list, score_function, steps=100, leapfrog_steps=10, step_size=0.1):
    """Perform Hamiltonian Monte Carlo sampling with multiple initial points using a specified score function."""
    x = x0_list.to(device)
    samples = []

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        p = torch.randn_like(x, device=device)

        for _ in range(leapfrog_steps):
            score, _ = score_function(x, _, step_size)
            p = p - 0.5 * step_size * score
            x = x + step_size * p
            score, _ = score_function(x, _, step_size)
            p = p - 0.5 * step_size * score
        
        samples.append(x.clone().detach())

    return torch.cat(samples)

# Experiment configurations
num_samples_list = [1, 10, 100, 1000, 10000]
steps_list = [1, 10, 100, 1000, 10000]
results = []

# Run experiments
for num_samples in num_samples_list:
    for steps in steps_list:
        x0_list = torch.randn(num_samples, device=device)

        # Benchmarking True Gradients (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_true = langevin_dynamics(x0_list, true_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_true = time.perf_counter() - start_time
        mem_true = mem_after - mem_before

        # Benchmarking Finite Differences (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_estimated = langevin_dynamics(x0_list, estimate_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_estimated = time.perf_counter() - start_time
        mem_estimated = mem_after - mem_before

        # Benchmarking SPSA (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_spsa = langevin_dynamics(x0_list, spsa_gradient, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_spsa = time.perf_counter() - start_time
        mem_spsa = mem_after - mem_before

        # Benchmarking True Gradients (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_true = hmc_sampler(x0_list, true_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_true = time.perf_counter() - start_time
        mem_hmc_true = mem_after - mem_before

        # Benchmarking Finite Differences (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_estimated = hmc_sampler(x0_list, estimate_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_estimated = time.perf_counter() - start_time
        mem_hmc_estimated = mem_after - mem_before

        # Benchmarking SPSA (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_spsa = hmc_sampler(x0_list, spsa_gradient, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_spsa = time.perf_counter() - start_time
        mem_hmc_spsa = mem_after - mem_before

        # Compute Wasserstein distance
        samples_true_np = samples_true.detach().cpu().numpy()
        samples_estimated_np = samples_estimated.detach().cpu().numpy()
        samples_spsa_np = samples_spsa.detach().cpu().numpy()
        samples_hmc_true_np = samples_hmc_true.detach().cpu().numpy()
        samples_hmc_estimated_np = samples_hmc_estimated.detach().cpu().numpy()
        samples_hmc_spsa_np = samples_hmc_spsa.detach().cpu().numpy()
        x_vals = torch.linspace(-5, 5, 1000).to(device)
        true_density = torch.exp(log_prob(x_vals)).detach().cpu().numpy()

        wasserstein_true = wasserstein_distance(samples_true_np.flatten(), true_density)
        wasserstein_estimated = wasserstein_distance(samples_estimated_np.flatten(), true_density)
        wasserstein_spsa = wasserstein_distance(samples_spsa_np.flatten(), true_density)
        wasserstein_hmc_true = wasserstein_distance(samples_hmc_true_np.flatten(), true_density)
        wasserstein_hmc_estimated = wasserstein_distance(samples_hmc_estimated_np.flatten(), true_density)
        wasserstein_hmc_spsa = wasserstein_distance(samples_hmc_spsa_np.flatten(), true_density)

        # Collect results
        results.append({
            "num_samples": num_samples,
            "steps": steps,
            "time_true": time_true,
            "mem_true": mem_true,
            "time_estimated": time_estimated,
            "mem_estimated": mem_estimated,
            "time_spsa": time_spsa,
            "mem_spsa": mem_spsa,
            "time_hmc_true": time_hmc_true,
            "mem_hmc_true": mem_hmc_true,
            "time_hmc_estimated": time_hmc_estimated,
            "mem_hmc_estimated": mem_hmc_estimated,
            "time_hmc_spsa": time_hmc_spsa,
            "mem_hmc_spsa": mem_hmc_spsa,
            "wasserstein_true": wasserstein_true,
            "wasserstein_estimated": wasserstein_estimated,
            "wasserstein_spsa": wasserstein_spsa,
            "wasserstein_hmc_true": wasserstein_hmc_true,
            "wasserstein_hmc_estimated": wasserstein_hmc_estimated,
            "wasserstein_hmc_spsa": wasserstein_hmc_spsa
        })
        print(f"Completed: num_samples = {num_samples}, steps = {steps}")

# Save results to file
with open('results.json', 'w') as f:
    json.dump(results, f)
print("Results saved to results.json")

# Prepare data for plotting
def prepare_plot_data(results, variable):
    num_samples = sorted(list(set(r["num_samples"] for r in results)))
    steps = sorted(list(set(r["steps"] for r in results)))
    data_true = [[r[f"{variable}_true"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    data_estimated = [[r[f"{variable}_estimated"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    data_spsa = [[r[f"{variable}_spsa"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    data_hmc_true = [[r[f"{variable}_hmc_true"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    data_hmc_estimated = [[r[f"{variable}_hmc_estimated"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    data_hmc_spsa = [[r[f"{variable}_hmc_spsa"] for r in results if r["steps"] == s and r["num_samples"] == num_samples] for s in steps]
    return num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa

def plot_3d(num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa, ylabel, title, filename):
    fig = go.Figure()
    fig.add_trace(go.Surface(z=data_true, x=num_samples, y=steps, colorscale='Viridis', name='True Gradients'))
    fig.add_trace(go.Surface(z=data_estimated, x=num_samples, y=steps, colorscale='Cividis', name='Finite Differences', showscale=False))
    fig.add_trace(go.Surface(z=data_spsa, x=num_samples, y=steps, colorscale='Blues', name='SPSA', showscale=False))
    fig.add_trace(go.Surface(z=data_hmc_true, x=num_samples, y=steps, colorscale='Inferno', name='HMC True', showscale=False))
    fig.add_trace(go.Surface(z=data_hmc_estimated, x=num_samples, y=steps, colorscale='Magma', name='HMC Estimated', showscale=False))
    fig.add_trace(go.Surface(z=data_hmc_spsa, x=num_samples, y=steps, colorscale='Plasma', name='HMC SPSA', showscale=False))
    fig.update_layout(
        scene=dict(
            xaxis_title='Number of Samples',
            yaxis_title='Number of Steps',
            zaxis_title=ylabel,
            xaxis_type='log',
            yaxis_type='log'
        ),
        title=title
    )
    fig.write_html(filename)
    print(f"Plot saved to {filename}")

# Plot results
num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa = prepare_plot_data(results, "time")
plot_3d(num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa, "Runtime (seconds)", "Runtime Comparison: True Gradients vs Finite Differences vs SPSA vs HMC", "runtime_comparison.html")

num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa = prepare_plot_data(results, "mem")
plot_3d(num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa, "Memory Usage (MB)", "Memory Usage Comparison: True Gradients vs Finite Differences vs SPSA vs HMC", "memory_comparison.html")

num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa = prepare_plot_data(results, "wasserstein")
plot_3d(num_samples, steps, data_true, data_estimated, data_spsa, data_hmc_true, data_hmc_estimated, data_hmc_spsa, "Wasserstein Distance", "Wasserstein Distance Comparison: True Gradients vs Finite Differences vs SPSA vs HMC", "wasserstein_comparison.html")