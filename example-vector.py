import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
import time

class DirectionPredictionModel(nn.Module):
    """Model for predicting 2D direction vectors"""
    def __init__(self):
        super().__init__()
        
        # Separate processing for each input
        self.current_pos_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.goal_pos_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.noisy_direction_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Time embedding
        self.time_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, 64),  # current + goal + noisy_direction + time
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: 2D direction vector
        )
    
    def forward(self, noisy_direction, current_pos, goal_pos, timestep):
        # Process each input separately
        current_emb = self.current_pos_net(current_pos)
        goal_emb = self.goal_pos_net(goal_pos)
        direction_emb = self.noisy_direction_net(noisy_direction)
        time_emb = self.time_net(timestep.float().unsqueeze(-1))
        
        # Combine and predict noise
        combined = torch.cat([current_emb, goal_emb, direction_emb, time_emb], dim=-1)
        return self.combined_net(combined)

class DirectionDiffusionSystem:
    """Step 2: Learn direction vectors using diffusion"""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🎮 Using device: {self.device}")
        
        # Model
        self.model = DirectionPredictionModel().to(self.device)
        
        # Diffusers scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Training history
        self.losses = []
        
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 Task: Predict direction = goal - current")
    
    def generate_data(self, num_samples=5000):
        """Generate training data: direction = goal - current"""
        print(f"📦 Generating {num_samples} 2D direction samples...")
        
        # Random current positions
        current_pos = torch.rand(num_samples, 2) * 10  # [0, 10] x [0, 10]
        
        # Random goal positions  
        goal_pos = torch.rand(num_samples, 2) * 10
        
        # Perfect direction vectors
        direction = goal_pos - current_pos
        
        # Normalize directions for stable training
        self.direction_mean = direction.mean(dim=0)
        self.direction_std = direction.std(dim=0)
        direction_normalized = (direction - self.direction_mean) / self.direction_std
        
        print(f"   Current pos range: [{current_pos.min():.2f}, {current_pos.max():.2f}]")
        print(f"   Goal pos range: [{goal_pos.min():.2f}, {goal_pos.max():.2f}]")
        print(f"   Direction range: [{direction.min():.2f}, {direction.max():.2f}]")
        print(f"   Direction normalized range: [{direction_normalized.min():.2f}, {direction_normalized.max():.2f}]")
        print(f"   Normalization: mean={self.direction_mean}, std={self.direction_std}")
        
        return current_pos, goal_pos, direction_normalized
    
    def train_step(self, current_batch, goal_batch, direction_batch):
        """Training step for direction prediction"""
        current_batch = current_batch.to(self.device)
        goal_batch = goal_batch.to(self.device)
        direction_batch = direction_batch.to(self.device)
        
        batch_size = current_batch.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # Add noise to direction vectors
        noise = torch.randn_like(direction_batch)
        noisy_direction = self.scheduler.add_noise(direction_batch, noise, timesteps)
        
        # Predict the noise
        noise_pred = self.model(noisy_direction, current_batch, goal_batch, timesteps)
        
        # Compute loss
        loss = nn.MSELoss()(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, epochs=50, batch_size=64):
        """Train the direction prediction model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Generate training data
        current_train, goal_train, direction_train = self.generate_data(num_samples=8000)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(current_train, goal_train, direction_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for current_batch, goal_batch, direction_batch in dataloader:
                loss = self.train_step(current_batch, goal_batch, direction_batch)
                epoch_losses.append(loss)
                self.losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📈 Final loss: {self.losses[-1]:.6f}")
    
    def generate(self, current_pos, goal_pos, num_inference_steps=50):
        """Generate direction vectors using diffusion"""
        self.model.eval()
        current_pos = current_pos.to(self.device)
        goal_pos = goal_pos.to(self.device)
        
        with torch.no_grad():
            # Start with pure noise
            direction_noisy = torch.randn(current_pos.shape[0], 2).to(self.device)
            
            # Set scheduler for inference  
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in self.scheduler.timesteps:
                # Create timestep tensor
                timestep_tensor = torch.full((current_pos.shape[0],), t, device=self.device)
                
                # Predict noise
                noise_pred = self.model(direction_noisy, current_pos, goal_pos, timestep_tensor)
                
                # Denoise
                direction_noisy = self.scheduler.step(noise_pred, t, direction_noisy).prev_sample
        
        self.model.train()
        return direction_noisy
    
    def test_examples(self):
        """Test with specific examples"""
        print(f"\n🧪 Testing direction prediction...")
        
        # Test cases: (current, goal) -> expected_direction
        test_cases = [
            ([1.0, 1.0], [3.0, 1.0], [2.0, 0.0]),   # East
            ([1.0, 1.0], [1.0, 3.0], [0.0, 2.0]),   # North  
            ([3.0, 3.0], [1.0, 1.0], [-2.0, -2.0]), # Southwest
            ([0.0, 0.0], [5.0, 5.0], [5.0, 5.0]),   # Northeast
            ([2.0, 3.0], [7.0, 1.0], [5.0, -2.0])   # Southeast
        ]
        
        print("Current -> Goal      | Expected    | Predicted   | Error")
        print("-" * 60)
        
        all_good = True
        for current, goal, expected in test_cases:
            current_tensor = torch.tensor([current], dtype=torch.float32)
            goal_tensor = torch.tensor([goal], dtype=torch.float32)
            expected_tensor = torch.tensor(expected, dtype=torch.float32)
            
            # Generate prediction (normalized)
            pred_normalized = self.generate(current_tensor, goal_tensor, num_inference_steps=50)
            
            # Ensure all tensors are on CPU for calculation
            pred_normalized_cpu = pred_normalized[0].cpu()
            direction_mean_cpu = self.direction_mean.cpu() if torch.is_tensor(self.direction_mean) else torch.tensor(self.direction_mean)
            direction_std_cpu = self.direction_std.cpu() if torch.is_tensor(self.direction_std) else torch.tensor(self.direction_std)
            
            # Denormalize
            pred_raw = pred_normalized_cpu * direction_std_cpu + direction_mean_cpu
            
            # Calculate error
            error = torch.norm(pred_raw - expected_tensor).item()
            
            # Format output
            current_str = f"({current[0]:3.1f},{current[1]:3.1f})"
            goal_str = f"({goal[0]:3.1f},{goal[1]:3.1f})"
            expected_str = f"({expected[0]:4.1f},{expected[1]:4.1f})"
            pred_str = f"({pred_raw[0]:4.1f},{pred_raw[1]:4.1f})"
            
            status = "✅" if error < 0.5 else "❌"
            if error >= 0.5:
                all_good = False
                
            print(f"{current_str} -> {goal_str} | {expected_str} | {pred_str} | {error:.3f} {status}")
        
        return all_good
    
    def visualize_predictions(self):
        """Visualize direction predictions"""
        print(f"\n🎨 Visualizing direction predictions...")
        
        # Create a grid of test points
        x = np.linspace(1, 9, 5)
        y = np.linspace(1, 9, 5)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Step 2: Direction Vector Predictions', fontsize=16)
        
        # Ensure normalization parameters are on CPU
        direction_mean_cpu = self.direction_mean.cpu() if torch.is_tensor(self.direction_mean) else torch.tensor(self.direction_mean)
        direction_std_cpu = self.direction_std.cpu() if torch.is_tensor(self.direction_std) else torch.tensor(self.direction_std)
        
        # Test case 1: All pointing to center
        ax = axes[0, 0]
        goal_center = torch.tensor([[5.0, 5.0]], dtype=torch.float32)  # Explicit float32
        
        for i in x:
            for j in y:
                if abs(i - 5.0) < 0.1 and abs(j - 5.0) < 0.1:
                    continue  # Skip center point
                    
                current = torch.tensor([[float(i), float(j)]], dtype=torch.float32)  # Explicit float32
                expected_dir = goal_center - current
                pred_dir_norm = self.generate(current, goal_center, num_inference_steps=50)
                
                # Move to CPU for denormalization
                pred_dir_norm_cpu = pred_dir_norm[0].cpu()
                pred_dir = pred_dir_norm_cpu * direction_std_cpu + direction_mean_cpu
                
                # Plot expected (blue) and predicted (red)
                ax.arrow(i, j, expected_dir[0, 0]*0.3, expected_dir[0, 1]*0.3, 
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
                ax.arrow(i, j, pred_dir[0]*0.3, pred_dir[1]*0.3, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        ax.plot(5.0, 5.0, 'go', markersize=10, label='Goal')
        ax.set_title('All pointing to center (5,5)')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.legend(['Goal', 'Expected', 'Predicted'])
        
        # Test case 2: Random goals
        ax = axes[0, 1]
        test_points = [(2, 2), (2, 8), (8, 2), (8, 8)]
        goals = [(7, 3), (6, 2), (3, 7), (2, 3)]
        
        for (cx, cy), (gx, gy) in zip(test_points, goals):
            current = torch.tensor([[float(cx), float(cy)]], dtype=torch.float32)  # Explicit float32
            goal = torch.tensor([[float(gx), float(gy)]], dtype=torch.float32)  # Explicit float32
            
            expected_dir = goal - current
            pred_dir_norm = self.generate(current, goal, num_inference_steps=50)
            
            # Move to CPU for denormalization
            pred_dir_norm_cpu = pred_dir_norm[0].cpu()
            pred_dir = pred_dir_norm_cpu * direction_std_cpu + direction_mean_cpu
            
            ax.arrow(cx, cy, expected_dir[0, 0]*0.3, expected_dir[0, 1]*0.3,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
            ax.arrow(cx, cy, pred_dir[0]*0.3, pred_dir[1]*0.3,
                    head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
            
            ax.plot(gx, gy, 'go', markersize=8)
            ax.plot(cx, cy, 'ko', markersize=6)
        
        ax.set_title('Random goal points')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        
        # Test case 3: Training loss
        ax = axes[1, 0]
        ax.plot(self.losses)
        ax.set_title('Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Test case 4: Error analysis
        ax = axes[1, 1]
        test_current = torch.tensor([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]], dtype=torch.float32)  # Explicit float32
        test_goal = torch.tensor([[8.0, 8.0], [2.0, 8.0], [2.0, 2.0]], dtype=torch.float32)  # Explicit float32
        
        errors = []
        for i in range(len(test_current)):
            current = test_current[i:i+1]
            goal = test_goal[i:i+1]
            expected = goal - current
            
            pred_norm = self.generate(current, goal, num_inference_steps=50)
            
            # Move to CPU for denormalization
            pred_norm_cpu = pred_norm[0].cpu()
            pred = pred_norm_cpu * direction_std_cpu + direction_mean_cpu
            
            error = torch.norm(pred - expected[0]).item()
            errors.append(error)
        
        ax.bar(range(len(errors)), errors)
        ax.set_title('Prediction Errors')
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Error Magnitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    print("🎯 Step 2: 2D Direction Vector Prediction")
    print("=" * 50)
    print("Goal: Learn direction = goal - current using diffusion")
    print()
    
    # Initialize system
    system = DirectionDiffusionSystem()
    
    # Train
    system.train(epochs=100, batch_size=64)
    
    # Test specific examples
    success = system.test_examples()
    
    # Visualize results
    system.visualize_predictions()
    
    if success:
        print("\n🎉 SUCCESS! Step 2 completed successfully!")
        print("🚀 Ready for Step 3: Adding simple image conditioning!")
        print("Next: Learn direction with obstacle awareness")
    else:
        print("\n⚠️  Step 2 needs refinement before moving to Step 3.")

if __name__ == "__main__":
    main()