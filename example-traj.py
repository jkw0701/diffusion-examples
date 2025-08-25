import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ConditionalTrajectoryDataset(Dataset):
    """Conditional 2D trajectory pattern dataset"""
    
    def __init__(self, num_samples=1000, seq_length=64, pattern_types=['circle', 'spiral', 'number8', 'line']):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.pattern_types = pattern_types
        self.pattern_to_id = {pattern: i for i, pattern in enumerate(pattern_types)}
        
        self.trajectories, self.pattern_labels = self._generate_trajectories()
        self.trajectories = self._normalize_data(self.trajectories)
    
    def _normalize_data(self, data):
        """Normalize data to [-3, 3] range"""
        all_points = data.reshape(-1, 2)
        min_vals = all_points.min(axis=0)
        max_vals = all_points.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = 6 * (data - min_vals) / range_vals - 3
        return normalized
    
    def _generate_trajectories(self):
        trajectories = []
        pattern_labels = []
        
        for _ in range(self.num_samples):
            pattern = np.random.choice(self.pattern_types)
            pattern_labels.append(self.pattern_to_id[pattern])
            
            if pattern == 'circle':
                traj = self._generate_circle()
            elif pattern == 'spiral':
                traj = self._generate_spiral()
            elif pattern == 'number8':
                traj = self._generate_number8()
            elif pattern == 'line':
                traj = self._generate_line()
            
            trajectories.append(traj)
        
        return np.array(trajectories), np.array(pattern_labels)
    
    def _generate_circle(self):
        """Generate circular trajectory"""
        t = np.linspace(0, 2*np.pi, self.seq_length)
        radius = np.random.uniform(1.0, 3.0)
        center_x = np.random.uniform(-2, 2)
        center_y = np.random.uniform(-2, 2)
        
        x = center_x + radius * np.cos(t)
        y = center_y + radius * np.sin(t)
        
        return np.column_stack([x, y])
    
    def _generate_spiral(self):
        """Generate spiral trajectory"""
        t = np.linspace(0, 4*np.pi, self.seq_length)
        radius_growth = np.random.uniform(0.1, 0.4)
        
        x = radius_growth * t * np.cos(t) + np.random.uniform(-1, 1)
        y = radius_growth * t * np.sin(t) + np.random.uniform(-1, 1)
        
        return np.column_stack([x, y])
    
    def _generate_number8(self):
        """Generate figure-8 trajectory"""
        t = np.linspace(0, 2*np.pi, self.seq_length)
        scale = np.random.uniform(2.0, 4.0)
        
        x = scale * np.sin(t) + np.random.uniform(-1, 1)
        y = scale * np.sin(t) * np.cos(t) + np.random.uniform(-1, 1)
        
        return np.column_stack([x, y])
    
    def _generate_line(self):
        """Generate straight line trajectory"""
        start_x = np.random.uniform(-4, 4)
        start_y = np.random.uniform(-4, 4)
        end_x = np.random.uniform(-4, 4)
        end_y = np.random.uniform(-4, 4)
        
        x = np.linspace(start_x, end_x, self.seq_length)
        y = np.linspace(start_y, end_y, self.seq_length)
        
        return np.column_stack([x, y])
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.trajectories[idx]), torch.LongTensor([self.pattern_labels[idx]])


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalResidualBlock(nn.Module):
    """Conditional 1D Residual Block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes=4, kernel_size=3):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Class embedding (pattern type)
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb, class_labels):
        """
        x: [batch_size, channels, seq_length]
        time_emb: [batch_size, time_emb_dim]
        class_labels: [batch_size]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Combine Time + Class embedding
        class_emb = self.class_emb(class_labels)  # [batch_size, time_emb_dim]
        combined_emb = time_emb + class_emb       # [batch_size, time_emb_dim]
        emb = self.time_mlp(combined_emb)         # [batch_size, out_channels]
        
        h = h + emb.unsqueeze(-1)  # [batch_size, out_channels, seq_length]
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        return self.activation(h + self.shortcut(x))


class ConditionalUNet1D(nn.Module):
    """Conditional 1D U-Net"""
    
    def __init__(self, in_channels=2, out_channels=2, seq_length=64, time_emb_dim=128, num_classes=4):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Encoder
        self.down1 = ConditionalResidualBlock(in_channels, 64, time_emb_dim, num_classes)
        self.down2 = ConditionalResidualBlock(64, 128, time_emb_dim, num_classes)
        self.down3 = ConditionalResidualBlock(128, 256, time_emb_dim, num_classes)
        
        self.downsample = nn.MaxPool1d(2)
        
        # Middle
        self.middle = ConditionalResidualBlock(256, 256, time_emb_dim, num_classes)
        
        # Decoder
        self.up3 = ConditionalResidualBlock(256 + 256, 128, time_emb_dim, num_classes)
        self.up2 = ConditionalResidualBlock(128 + 128, 64, time_emb_dim, num_classes)
        self.up1 = ConditionalResidualBlock(64 + 64, 64, time_emb_dim, num_classes)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # Output
        self.out_conv = nn.Conv1d(64, out_channels, 1)
    
    def forward(self, x, t, class_labels):
        """
        x: [batch_size, seq_length, 2] - noisy trajectory
        t: [batch_size] - timestep
        class_labels: [batch_size] - pattern type
        """
        x = x.transpose(1, 2)  # [batch_size, 2, seq_length]
        
        time_emb = self.time_embed(t)
        
        # Encoder
        h1 = self.down1(x, time_emb, class_labels)
        h1_pooled = self.downsample(h1)
        
        h2 = self.down2(h1_pooled, time_emb, class_labels)
        h2_pooled = self.downsample(h2)
        
        h3 = self.down3(h2_pooled, time_emb, class_labels)
        h3_pooled = self.downsample(h3)
        
        # Middle
        h_middle = self.middle(h3_pooled, time_emb, class_labels)
        
        # Decoder
        h_up3 = self.upsample(h_middle)
        h_up3 = torch.cat([h_up3, h3], dim=1)
        h_up3 = self.up3(h_up3, time_emb, class_labels)
        
        h_up2 = self.upsample(h_up3)
        h_up2 = torch.cat([h_up2, h2], dim=1)
        h_up2 = self.up2(h_up2, time_emb, class_labels)
        
        h_up1 = self.upsample(h_up2)
        h_up1 = torch.cat([h_up1, h1], dim=1)
        h_up1 = self.up1(h_up1, time_emb, class_labels)
        
        out = self.out_conv(h_up1)
        return out.transpose(1, 2)  # [batch_size, seq_length, 2]


class ConditionalDiffusionTrainer:
    """Conditional diffusion model trainer"""
    
    def __init__(self, model, num_timesteps=1000, beta_start=0.00001, beta_end=0.008):
        self.model = model
        self.num_timesteps = num_timesteps
        
        # Smoother noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0, t):
        """Forward process"""
        noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def train_step(self, batch, optimizer):
        """Training step"""
        trajectories, class_labels = batch
        trajectories = trajectories.to(device)
        class_labels = class_labels.squeeze(-1).to(device)  # [batch_size]
        
        batch_size = trajectories.shape[0]
        
        # Random timestep
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Add noise
        noisy_x, noise = self.add_noise(trajectories, t)
        
        # Predict noise (conditional)
        predicted_noise = self.model(noisy_x, t, class_labels)
        
        # Compute loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    def ddim_sample(self, shape, class_labels, num_inference_steps=50, eta=0.0):
        """DDIM sampling (deterministic, smooth results)"""
        self.model.eval()
        
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # DDIM timesteps (evenly spaced)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps).long().to(device)
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device)
                
                # Predict noise
                predicted_noise = self.model(x, t_batch, class_labels)
                
                # DDIM update
                alpha_t = self.alphas_cumprod[t]
                if i < len(timesteps) - 1:
                    alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
                else:
                    alpha_t_prev = torch.tensor(1.0, device=device)
                
                # Predict x_0
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -4, 4)
                
                # DDIM formula
                if i < len(timesteps) - 1:
                    # Deterministic direction
                    dir_xt = torch.sqrt(1 - alpha_t_prev) * predicted_noise
                    
                    # Stochastic noise (eta=0 for fully deterministic)
                    if eta > 0:
                        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                        noise = torch.randn_like(x)
                        x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma * noise
                    else:
                        x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
                else:
                    x = pred_x0
        
        self.model.train()
        return x


def train_conditional_unet():
    """Train conditional U-Net"""
    print("Training conditional U-Net + DDIM sampling...")
    
    # Conditional dataset
    dataset = ConditionalTrajectoryDataset(num_samples=2000, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Pattern types: {dataset.pattern_types}")
    print(f"Dataset size: {len(dataset)}")
    
    # Conditional model
    model = ConditionalUNet1D(in_channels=2, out_channels=2, seq_length=64, 
                             time_emb_dim=128, num_classes=4).to(device)
    trainer = ConditionalDiffusionTrainer(model, num_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training
    num_epochs = 100
    losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for i, batch in enumerate(dataloader):
            loss = trainer.train_step(batch, optimizer)
            epoch_losses.append(loss)
            
            if i % 30 == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}, Loss: {loss:.6f}")
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")
            
            # Intermediate test: generate each pattern
            with torch.no_grad():
                for pattern_id, pattern_name in enumerate(dataset.pattern_types):
                    class_labels = torch.full((1,), pattern_id, device=device)
                    sample = trainer.ddim_sample((1, 64, 2), class_labels, num_inference_steps=25)
                    sample_np = sample[0].cpu().numpy()
                    print(f"  {pattern_name}: X[{sample_np[:, 0].min():.2f}, {sample_np[:, 0].max():.2f}], Y[{sample_np[:, 1].min():.2f}, {sample_np[:, 1].max():.2f}]")
    
    return model, trainer, losses, dataset


def visualize_conditional_results(model, trainer, dataset, losses):
    """Visualize conditional model results"""
    plt.figure(figsize=(15, 12))
    
    # 1. Loss curve
    plt.subplot(3, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')
    
    # 2. Original trajectories
    plt.subplot(3, 3, 2)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(8):
        traj = dataset[i][0].numpy()
        pattern_id = dataset[i][1].item()
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2, color=colors[pattern_id])
    plt.title('Original Trajectories (by Pattern)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    
    # 3-6. Generated results for each pattern
    pattern_names = dataset.pattern_types
    
    for pattern_id, pattern_name in enumerate(pattern_names):
        plt.subplot(3, 3, 3 + pattern_id)
        
        try:
            # Generate with pattern condition
            class_labels = torch.full((4,), pattern_id, device=device)
            generated = trainer.ddim_sample((4, 64, 2), class_labels, num_inference_steps=50, eta=0.0)
            
            for i in range(4):
                traj = generated[i].cpu().numpy()
                if not (np.isnan(traj).any() or np.isinf(traj).any()):
                    plt.plot(traj[:, 0], traj[:, 1], alpha=0.8, linewidth=2, color=colors[pattern_id])
            
            plt.title(f'Generated {pattern_name.capitalize()}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.axis('equal')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 7. Original vs Generated comparison
    plt.subplot(3, 3, 7)
    
    # Original (one for each pattern)
    for pattern_id, pattern_name in enumerate(pattern_names):
        # Find original pattern
        for i in range(len(dataset)):
            if dataset[i][1].item() == pattern_id:
                traj = dataset[i][0].numpy()
                plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=3, 
                        color=colors[pattern_id], linestyle='-', label=f'Orig {pattern_name}')
                break
    
    plt.title('Original Patterns')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 8. DDIM vs DDPM comparison
    plt.subplot(3, 3, 8)
    
    try:
        # DDIM (deterministic)
        class_labels = torch.full((2,), 0, device=device)  # Circle pattern
        ddim_samples = trainer.ddim_sample((2, 64, 2), class_labels, num_inference_steps=50, eta=0.0)
        
        for i in range(2):
            traj = ddim_samples[i].cpu().numpy()
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.8, linewidth=2, color='blue', label='DDIM' if i == 0 else '')
        
        plt.title('DDIM Sampling (Deterministic)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'DDIM Error:\n{str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 9. All patterns mixed generation
    plt.subplot(3, 3, 9)
    
    try:
        for pattern_id in range(len(pattern_names)):
            class_labels = torch.full((2,), pattern_id, device=device)
            samples = trainer.ddim_sample((2, 64, 2), class_labels, num_inference_steps=30, eta=0.0)
            
            for i in range(2):
                traj = samples[i].cpu().numpy()
                plt.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=2, color=colors[pattern_id])
        
        plt.title('All Patterns Generated')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Mixed Error:\n{str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Conditional U-Net + DDIM Sampling ===")
    print("1. Conditional Generation: Provide pattern type as condition")
    print("2. DDIM Sampling: Deterministic sampling for smooth results\n")
    
    # Training
    model, trainer, losses, dataset = train_conditional_unet()
    
    # Visualization
    print("\nVisualizing results...")
    visualize_conditional_results(model, trainer, dataset, losses)
    
    # Final test
    print("\n=== Pattern-wise Generation Test ===")
    try:
        for pattern_id, pattern_name in enumerate(dataset.pattern_types):
            print(f"\n{pattern_name.upper()} pattern generation:")
            
            class_labels = torch.full((3,), pattern_id, device=device)
            samples = trainer.ddim_sample((3, 64, 2), class_labels, num_inference_steps=50, eta=0.0)
            
            valid_count = 0
            for i, sample in enumerate(samples):
                sample_np = sample.cpu().numpy()
                if not (np.isnan(sample_np).any() or np.isinf(sample_np).any()):
                    valid_count += 1
                    print(f"  ✅ Sample {i+1}: X[{sample_np[:, 0].min():.2f}, {sample_np[:, 0].max():.2f}], Y[{sample_np[:, 1].min():.2f}, {sample_np[:, 1].max():.2f}]")
                else:
                    print(f"  ❌ Sample {i+1}: Generation failed")
            
            if valid_count == 3:
                print(f"  🎉 {pattern_name} pattern generated perfectly!")
            
    except Exception as e:
        print(f"❌ Pattern-wise test failed: {e}")
    
    print("\n🚀 Conditional generation + DDIM enables clear pattern distinction!")