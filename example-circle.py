# Step-by-step Diffusion Implementation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Step 1: Start with the simplest data
class ToyDataset(Dataset):
    """Simple 2D point dataset"""
    def __init__(self, num_samples=1000):
        self.data = self.generate_toy_data(num_samples)
    
    def generate_toy_data(self, num_samples):
        """Generate simple 2D points - circular shape"""
        # Generate points on a circle
        angles = np.random.uniform(0, 2*np.pi, num_samples)
        radius = 1.0 + np.random.normal(0, 0.1, num_samples)  # slight noise
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        # [[x1,y1], [x2,y2], [x3,y3], ...]
        data = np.stack([x, y], axis=1).astype(np.float32) 
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# Step 2: Simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, time_dim=16):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim), # x = nn.Linear(1, 16)
            nn.SiLU(), # x.shape = (256, 16) 
            nn.Linear(time_dim, time_dim) # 
        )
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        # t is normalized timestep in [0, 1] range
        # t.unsqueeze(-1).shape = (256, 1)
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Combine x and time embedding
        combined = torch.cat([x, t_embed], dim=-1)
        
        # Predict noise
        return self.net(combined)

# Step 3: Simple scheduler implemented manually
class SimpleScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        
        # Linear schedule
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x0, noise, t):
        """Forward process: add noise to x0"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # Adjust shape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1)
        
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
    
    def step(self, model_output, t, sample):
        """Reverse process: one step denoising"""
        # Check if t is scalar or tensor
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                t = t.item()  # Convert to scalar
            else:
                # For batch processing, handle each separately
                return self._step_batch(model_output, t, sample)
        
        # DDPM sampling formula (for scalar t)
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        # Predicted x0
        pred_x0 = (sample - torch.sqrt(1.0 - alpha_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        
        # Direction to xt-1
        pred_dir = torch.sqrt(1.0 - alpha_cumprod_prev) * model_output
        
        # Previous sample
        prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir
        
        return prev_sample
    
    def _step_batch(self, model_output, t_batch, sample):
        """Internal function for batch processing"""
        batch_size = t_batch.shape[0]
        prev_samples = []
        
        for i in range(batch_size):
            t = t_batch[i].item()
            single_output = model_output[i:i+1]
            single_sample = sample[i:i+1]
            
            # Process single sample
            prev_sample = self.step(single_output, t, single_sample)
            prev_samples.append(prev_sample)
        
        return torch.cat(prev_samples, dim=0)

# Step 4: Training function
def train_simple_diffusion():
    print("=== Training Simple Diffusion on 2D Point Data ===")
    
    # Prepare data
    dataset = ToyDataset(num_samples=5000)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Model and scheduler
    model = SimpleMLP()
    scheduler = SimpleScheduler(num_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Data visualization
    plt.figure(figsize=(6, 6))
    data_sample = dataset.data[:1000]
    plt.scatter(data_sample[:, 0], data_sample[:, 1], alpha=0.5, s=1)
    plt.title('Original Data Distribution')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Training
    model.train()
    losses = []
    
    for epoch in range(500):  # Sufficient training
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Random timestep
            t = torch.randint(0, scheduler.num_timesteps, (batch.shape[0],))
            
            # Add noise
            noise = torch.randn_like(batch)
            noisy_data = scheduler.add_noise(batch, noise, t)
            
            # Normalized timestep (0~1 range)
            t_normalized = t.float() / scheduler.num_timesteps
            
            # Predict noise
            predicted_noise = model(noisy_data, t_normalized)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses[-100:])  # Last 100 epochs
    plt.title('Recent Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, scheduler

# Step 5: Denoising process visualization (fixed)
def visualize_denoising_process(model, scheduler, num_samples=1000):
    print("=== Visualizing Denoising Process ===")
    
    model.eval()
    
    # Select timesteps to show
    timesteps_to_show = [999, 800, 600, 400, 200, 100, 50, 0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        # Start with pure noise
        samples = torch.randn(num_samples, 2)
        
        # Save state at each timestep
        saved_states = {999: samples.clone()}
        
        # Reverse process - process one at a time
        for t in range(999, -1, -1):
            if len(samples) > 0:
                t_normalized = torch.full((samples.shape[0],), t / 1000.0)
                
                # Predict noise
                predicted_noise = model(samples, t_normalized)
                
                # One step denoising - use scalar t
                samples = scheduler.step(predicted_noise, t, samples)
                
                # Save state at specific timesteps
                if t in timesteps_to_show:
                    saved_states[t] = samples.clone()
    
    # Visualization
    for i, t in enumerate(timesteps_to_show):
        if t in saved_states:
            data = saved_states[t].numpy()
            
            axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, s=1)
            axes[i].set_title(f'Timestep: {t}')
            axes[i].set_xlim(-3, 3)
            axes[i].set_ylim(-3, 3)
            axes[i].grid(True, alpha=0.3)
            axes[i].axis('equal')
            
            # Show progress
            progress = (999 - t) / 999 * 100
            axes[i].set_xlabel(f'Progress: {progress:.0f}%')
    
    plt.suptitle('Denoising Process: Noise → Clean Data', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return saved_states[0]  # Final result

# Step 6: Forward process visualization (fixed)
def visualize_forward_process(original_data, scheduler):
    print("=== Visualizing Forward Process ===")
    
    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original data sample
    data_sample = torch.tensor(original_data[:1000])
    
    for i, t in enumerate(timesteps_to_show):
        if t == 0:
            noisy_data = data_sample
        else:
            noise = torch.randn_like(data_sample)
            # Process in small batches to handle one at a time
            batch_size = 100
            noisy_batches = []
            
            for start_idx in range(0, len(data_sample), batch_size):
                end_idx = min(start_idx + batch_size, len(data_sample))
                batch_data = data_sample[start_idx:end_idx]
                batch_noise = noise[start_idx:end_idx]
                
                # Use scalar t
                batch_noisy = scheduler.add_noise(batch_data, batch_noise, 
                                                torch.full((len(batch_data),), t))
                noisy_batches.append(batch_noisy)
            
            noisy_data = torch.cat(noisy_batches, dim=0)
        
        data_np = noisy_data.numpy()
        
        axes[i].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=1)
        axes[i].set_title(f'Timestep: {t}')
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')
        
        # Show noise level
        if t > 0:
            noise_level = (1.0 - scheduler.alphas_cumprod[t]).item()
            axes[i].set_xlabel(f'Noise Level: {noise_level:.3f}')
        else:
            axes[i].set_xlabel('Original Data')
    
    plt.suptitle('Forward Process: Clean Data → Noise', fontsize=16)
    plt.tight_layout()
    plt.show()

# Step 7: Main execution function
def main_step_by_step():
    print("Step-by-step Diffusion Implementation")
    print("=" * 50)
    
    # 1. Check data
    print("1. Generate and check original data...")
    dataset = ToyDataset(num_samples=2000)
    scheduler = SimpleScheduler()
    
    # 2. Check forward process
    print("2. Check Forward Process (noise addition)...")
    visualize_forward_process(dataset.data, scheduler)
    
    # 3. Train model
    print("3. Train model...")
    model, scheduler = train_simple_diffusion()
    
    # 4. Check denoising process
    print("4. Check Reverse Process (Denoising)...")
    final_samples = visualize_denoising_process(model, scheduler)
    
    # 5. Final comparison
    print("5. Compare original vs generated data...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    original = dataset.data[:1000]
    axes[0].scatter(original[:, 0], original[:, 1], alpha=0.6, s=1, color='blue')
    axes[0].set_title('Original Data')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Generated data
    generated = final_samples.numpy()
    axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.6, s=1, color='red')
    axes[1].set_title('Generated Data')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # Overlay comparison
    axes[2].scatter(original[:, 0], original[:, 1], alpha=0.4, s=1, color='blue', label='Original')
    axes[2].scatter(generated[:, 0], generated[:, 1], alpha=0.4, s=1, color='red', label='Generated')
    axes[2].set_title('Overlay Comparison')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Complete! Now you can see that denoising actually works.")
    
    return model, scheduler, final_samples

# Step 8: Step-by-step debugging function (fixed)
def debug_model_predictions(model, scheduler):
    """Check if the model actually predicts the correct noise"""
    print("=== Checking Model Prediction Accuracy ===")
    
    model.eval()
    
    # Test data
    test_data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    
    timesteps_to_test = [100, 300, 500, 800]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, t in enumerate(timesteps_to_test):
            # Add actual noise
            actual_noise = torch.randn_like(test_data)
            t_tensor = torch.full((test_data.shape[0],), t)
            noisy_data = scheduler.add_noise(test_data, actual_noise, t_tensor)
            
            # Model's noise prediction
            t_normalized = torch.full((test_data.shape[0],), t / 1000.0)
            predicted_noise = model(noisy_data, t_normalized)
            
            # Visualization
            axes[i].scatter(actual_noise[:, 0], actual_noise[:, 1], 
                          color='blue', label='Actual Noise', s=100, alpha=0.7)
            axes[i].scatter(predicted_noise[:, 0], predicted_noise[:, 1], 
                          color='red', label='Predicted Noise', s=100, alpha=0.7, marker='x')
            
            # Connect with arrows
            for j in range(len(test_data)):
                axes[i].annotate('', xy=predicted_noise[j], xytext=actual_noise[j],
                               arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
            
            axes[i].set_title(f'Timestep {t}\nNoise Prediction Accuracy')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axis('equal')
            
            # Calculate MSE
            mse = F.mse_loss(predicted_noise, actual_noise).item()
            axes[i].set_xlabel(f'MSE: {mse:.4f}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Required libraries: pip install torch matplotlib numpy")
    print()
    
    # Step-by-step execution
    model, scheduler, final_samples = main_step_by_step()
    
    # Debugging (optional)
    debug_choice = input("\nWould you like to check model prediction accuracy? (y/n): ").lower()
    if debug_choice == 'y':
        debug_model_predictions(model, scheduler)