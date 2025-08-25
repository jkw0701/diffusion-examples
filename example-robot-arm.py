import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import math
import argparse
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time
import threading
import queue
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib
matplotlib.use('Qt5Agg')  # Explicitly set GUI backend

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RobotArmDataset(Dataset):
    def __init__(self, num_samples=10000, seq_length=50):
        self.seq_length = seq_length
        self.data = []
        self.generate_data(num_samples)
    
    def forward_kinematics(self, theta1, theta2, L1=1.0, L2=1.0):
        """Forward kinematics: Calculate end-effector position from joint angles"""
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        return x, y
    
    def inverse_kinematics(self, x, y, L1=1.0, L2=1.0):
        """Inverse kinematics: Calculate joint angles from target position"""
        # Check reachability
        distance = np.sqrt(x**2 + y**2)
        if distance > (L1 + L2) or distance < abs(L1 - L2):
            return None, None
        
        # Calculate theta2 using cosine rule
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)  # Numerical stability
        theta2 = np.arccos(cos_theta2)
        
        # Calculate theta1
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        return theta1, theta2
    
    def generate_smooth_trajectory(self, start_joints, goal_joints, seq_length):
        """Generate smooth trajectory between start and goal points"""
        trajectory = np.zeros((seq_length, 2))
        
        # Linear interpolation
        for i in range(seq_length):
            t = i / (seq_length - 1)
            # Smooth interpolation with ease-in-out function
            t_smooth = 3 * t**2 - 2 * t**3
            trajectory[i] = start_joints + t_smooth * (goal_joints - start_joints)
        
        # Add slight noise for natural movement
        noise = np.random.normal(0, 0.05, trajectory.shape)
        trajectory += noise
        
        # Apply joint limits (-π, π)
        trajectory = np.clip(trajectory, -np.pi, np.pi)
        
        return trajectory
    
    def sample_reachable_position(self, L1=1.0, L2=1.0):
        """Generate random reachable position"""
        while True:
            # Sample within reachable circle
            r = np.random.uniform(0.1, L1 + L2 - 0.1)  # Avoid singularities
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Validate with inverse kinematics
            theta1, theta2 = self.inverse_kinematics(x, y, L1, L2)
            if theta1 is not None and theta2 is not None:
                return np.array([x, y]), np.array([theta1, theta2])
    
    def generate_data(self, num_samples):
        """Generate training data"""
        print("Generating training data...")
        for _ in tqdm(range(num_samples)):
            # Random start joint angles
            start_joints = np.random.uniform(-np.pi, np.pi, 2)
            
            # Reachable goal position and corresponding joint angles
            goal_pos, goal_joints = self.sample_reachable_position()
            
            # Generate smooth trajectory
            trajectory = self.generate_smooth_trajectory(
                start_joints, goal_joints, self.seq_length
            )
            
            # Condition: start joint angles + goal position
            condition = np.concatenate([start_joints, goal_pos])
            
            self.data.append({
                'trajectory': torch.FloatTensor(trajectory),
                'condition': torch.FloatTensor(condition),
                'start_joints': torch.FloatTensor(start_joints),
                'goal_pos': torch.FloatTensor(goal_pos)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SinusoidalPositionEmbedding(nn.Module):
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

class ResidualBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        self.cond_mlp = nn.Linear(cond_dim, dim)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 3, padding=1)
        )
    
    def forward(self, x, time_emb, cond_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None]
        
        # Add condition embedding
        cond_emb = self.cond_mlp(cond_emb)
        h = h + cond_emb[..., None]
        
        h = self.block2(h)
        return x + h

class UNet1D(nn.Module):
    def __init__(self, input_dim=2, cond_dim=4, dim=64, time_steps=1000):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        
        # Time embedding
        self.time_emb = SinusoidalPositionEmbedding(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, dim, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList([
            ResidualBlock(dim, dim * 4, cond_dim),
            ResidualBlock(dim, dim * 4, cond_dim),
        ])
        
        # Middle
        self.middle = ResidualBlock(dim, dim * 4, cond_dim)
        
        # Decoder
        self.decoder = nn.ModuleList([
            ResidualBlock(dim, dim * 4, cond_dim),
            ResidualBlock(dim, dim * 4, cond_dim),
        ])
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, input_dim, 3, padding=1)
        )
    
    def forward(self, x, timestep, condition):
        # x: (batch, input_dim, seq_len)
        # timestep: (batch,)
        # condition: (batch, cond_dim)
        
        # Time embedding
        time_emb = self.time_emb(timestep)
        time_emb = self.time_mlp(time_emb)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder
        for block in self.encoder:
            x = block(x, time_emb, condition)
        
        # Middle
        x = self.middle(x, time_emb, condition)
        
        # Decoder
        for block in self.decoder:
            x = block(x, time_emb, condition)
        
        # Output
        x = self.output(x)
        return x

class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        
        # Beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t, t, condition):
        """Reverse diffusion step"""
        # Predict noise
        noise_pred = self.model(x_t, t, condition)
        
        # Calculate x_{t-1}
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, shape, condition, device, fast_sampling=False, num_steps=50):
        """Complete sampling process with optional fast sampling"""
        x = torch.randn(shape, device=device)
        
        if fast_sampling:
            # Fast sampling with fewer steps
            step_size = self.timesteps // num_steps
            timesteps = list(range(0, self.timesteps, step_size))[:num_steps]
            timesteps = timesteps[::-1]  # Reverse order
            
            for i in timesteps:
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = self.p_sample(x, t, condition)
        else:
            # Original full sampling
            for i in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = self.p_sample(x, t, condition)
        
        return x
    
    def ddim_sample(self, shape, condition, device, num_steps=50, eta=0.0):
        """DDIM sampling for faster inference"""
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule
        skip = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, skip))[:num_steps]
        timesteps = timesteps[::-1]  # Reverse order
        
        timesteps_prev = [0] + timesteps[:-1]
        
        for i, (t_cur, t_prev) in enumerate(zip(timesteps, timesteps_prev)):
            t_cur_tensor = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_cur_tensor, condition)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t_cur]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
            
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # Predicted x0
            pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Direction to xt
            dir_xt = sqrt_one_minus_alpha_t_prev * noise_pred
            
            # Add noise for stochastic sampling (eta > 0)
            if eta > 0 and i < len(timesteps) - 1:
                sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                noise = torch.randn_like(x)
                dir_xt += sigma_t * noise
            
            x = sqrt_alpha_t_prev * pred_x0 + dir_xt
        
        return x

class InteractiveDemo:
    def __init__(self, model_path, fast_mode=False):
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet1D(input_dim=2, cond_dim=4, dim=64).to(self.device)
        self.ddpm = DDPM(self.model, timesteps=1000)
        self.fast_mode = fast_mode
        
        # Move DDPM parameters to device
        for name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                     'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self.ddpm, name, getattr(self.ddpm, name).to(self.device))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Dataset (for utility functions)
        self.dataset = RobotArmDataset(num_samples=1, seq_length=50)
        
        # UI state
        self.start_pos = None
        self.goal_pos = None
        self.current_trajectory = None
        self.animation_running = False
        
        # Animation related
        self.anim = None
        self.stop_animation = False
        
        print(f"Demo initialized with fast_mode={'ON' if fast_mode else 'OFF'}")
        self.setup_ui()

    def setup_ui(self):
        """Setup interactive UI - improved version"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: workspace (click to set goal)
        self.ax1.set_xlim(-2.5, 2.5)
        self.ax1.set_ylim(-2.5, 2.5)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        mode_text = "FAST" if self.fast_mode else "FULL"
        self.ax1.set_title(f'Workspace ({mode_text} mode) - Click to set goal', fontsize=14)
        
        # Display workspace
        circle = plt.Circle((0, 0), 2, fill=False, linestyle='--', color='gray', alpha=0.5)
        self.ax1.add_patch(circle)
        
        # Right: joint space
        self.ax2.set_xlim(-np.pi, np.pi)
        self.ax2.set_ylim(-np.pi, np.pi)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlabel('θ₁ (rad)')
        self.ax2.set_ylabel('θ₂ (rad)')
        self.ax2.set_title('Joint Space', fontsize=14)
        
        # Connect mouse click events (for goal setting)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Command guide
        self.fig.suptitle('Commands: [r] Random Goal | [c] Clear | [t] Toggle Speed | [s] Stop Anim | [q] Quit', 
                         fontsize=12, y=0.02)
        
        # Set initial start position
        self.set_random_start()
        
        # Set focus on plot window
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'q':
            print("Quitting...")
            plt.close('all')
        elif event.key == 'r':
            self.random_goal()
        elif event.key == 'c':
            self.clear_all()
        elif event.key == 't':
            self.toggle_speed()
        elif event.key == 's':
            self.stop_current_animation()
        elif event.key == 'h':
            self.show_help()
    
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes == self.ax1 and not self.animation_running:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(f"Clicked goal at: ({x:.2f}, {y:.2f})")
                distance = np.sqrt(x**2 + y**2)
                if distance <= 2.0 and distance >= 0.1:
                    self.goal_pos = np.array([x, y])
                    self.generate_trajectory()
                else:
                    print(f"Goal position ({x:.2f}, {y:.2f}) is not reachable!")
    
    def stop_current_animation(self):
        """Stop current animation"""
        if self.animation_running and self.anim is not None:
            self.stop_animation = True
            self.animation_running = False
            print("Animation stopped!")
    
    def random_goal(self):
        """Generate random goal"""
        if self.animation_running:
            print("Animation running, please wait or press 's' to stop...")
            return
            
        try:
            goal_pos, _ = self.dataset.sample_reachable_position()
            self.goal_pos = goal_pos
            print(f"Random goal set at: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
            self.generate_trajectory()
        except Exception as e:
            print(f"Error generating random goal: {e}")
    
    def clear_all(self):
        """Reset everything"""
        if self.animation_running:
            self.stop_current_animation()
            
        try:
            print("Clearing and resetting...")
            self.goal_pos = None
            self.current_trajectory = None
            
            # Reset animation state
            self.stop_animation = False
            
            self.set_random_start()
            print("Reset completed!")
        except Exception as e:
            print(f"Error clearing: {e}")
    
    def toggle_speed(self):
        """Toggle sampling speed mode"""
        if self.animation_running:
            print("Animation running, please wait or press 's' to stop...")
            return
            
        try:
            self.fast_mode = not self.fast_mode
            mode_text = "FAST" if self.fast_mode else "FULL"
            self.ax1.set_title(f'Workspace ({mode_text} mode) - Click to set goal', fontsize=14)
            print(f"Switched to {'FAST' if self.fast_mode else 'FULL'} sampling mode")
            self.fig.canvas.draw()
        except Exception as e:
            print(f"Error toggling speed: {e}")
    
    def show_help(self):
        """Display help message"""
        print("\n=== Available Commands ===")
        print("Click on workspace - Set goal position")
        print("r - Generate random goal")
        print("c - Clear and reset")
        print("t - Toggle between FAST/FULL mode")
        print("s - Stop current animation")
        print("h - Show this help")
        print("q - Exit demo")
        print("==========================\n")
    
    def set_random_start(self):
        """Set random start position"""
        self.start_joints = np.random.uniform(-np.pi, np.pi, 2)
        start_x, start_y = self.dataset.forward_kinematics(self.start_joints[0], self.start_joints[1])
        self.start_pos = np.array([start_x, start_y])
        
        # Display start position
        self.ax1.clear()
        self.ax1.set_xlim(-2.5, 2.5)
        self.ax1.set_ylim(-2.5, 2.5)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        mode_text = "FAST" if self.fast_mode else "FULL"
        self.ax1.set_title(f'Workspace ({mode_text} mode) - Click to set goal', fontsize=14)
        
        # Display workspace
        circle = plt.Circle((0, 0), 2, fill=False, linestyle='--', color='gray', alpha=0.5)
        self.ax1.add_patch(circle)
        
        # Draw start robot arm
        x1 = np.cos(self.start_joints[0])
        y1 = np.sin(self.start_joints[0])
        x2 = x1 + np.cos(self.start_joints[0] + self.start_joints[1])
        y2 = y1 + np.sin(self.start_joints[0] + self.start_joints[1])
        
        self.ax1.plot([0, x1, x2], [0, y1, y2], 'k-', linewidth=4, alpha=0.7, label='Start Pose')
        self.ax1.plot([0, x1, x2], [0, y1, y2], 'ko', markersize=8)
        self.ax1.plot(start_x, start_y, 'go', markersize=10, label='Start EE')
        
        self.ax1.legend()
        
        # Display start position in joint space
        self.ax2.clear()
        self.ax2.set_xlim(-np.pi, np.pi)
        self.ax2.set_ylim(-np.pi, np.pi)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlabel('θ₁ (rad)')
        self.ax2.set_ylabel('θ₂ (rad)')
        self.ax2.set_title('Joint Space', fontsize=14)
        self.ax2.plot(self.start_joints[0], self.start_joints[1], 'go', markersize=10, label='Start')
        self.ax2.legend()
        
        # Force screen update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def generate_trajectory(self):
        """Generate trajectory with diffusion model"""
        if self.goal_pos is None:
            return
        
        if self.animation_running:
            print("Animation running, please wait...")
            return
        
        mode_text = "FAST" if self.fast_mode else "FULL"
        print(f"Generating trajectory to goal: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f}) [{mode_text} mode]")
        
        # Prepare condition
        condition = torch.FloatTensor(np.concatenate([self.start_joints, self.goal_pos])).unsqueeze(0).to(self.device)
        
        # Generate trajectory
        start_time = time.time()
        with torch.no_grad():
            shape = (1, 2, self.dataset.seq_length)
            
            if self.fast_mode:
                # DDIM fast sampling (20 steps)
                generated_traj = self.ddpm.ddim_sample(shape, condition, self.device, num_steps=100)
            else:
                # Full sampling (1000 steps)
                generated_traj = self.ddpm.sample(shape, condition, self.device)
            
            generated_traj = generated_traj.squeeze().cpu().numpy().T  # (seq_length, 2)
        
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.3f}s")
        
        self.current_trajectory = generated_traj
        
        # Display goal point
        self.ax1.plot(self.goal_pos[0], self.goal_pos[1], 'ro', markersize=10, label='Goal')
        self.ax1.legend()
        
        # Start animation
        self.animate_trajectory()
    
    def animate_trajectory(self):
        """Animate trajectory - progress display version"""
        if self.current_trajectory is None:
            return
        
        self.animation_running = True
        self.stop_animation = False
        print("Starting animation...")
        
        # Pre-calculate EE trajectory
        ee_positions = []
        for joints in self.current_trajectory:
            x, y = self.dataset.forward_kinematics(joints[0], joints[1])
            ee_positions.append([x, y])
        self.ee_positions = np.array(ee_positions)
        
        # Initialize animation objects
        self.arm_line = None
        self.arm_joints = None
        self.ee_trail = None
        self.joint_trail = None
        self.joint_current = None
        
        def animate_func(frame):
            # Check for stop request
            if self.stop_animation or frame >= len(self.current_trajectory):
                if frame >= len(self.current_trajectory):
                    print("Animation completed!")
                else:
                    print("Animation stopped!")
                self.animation_running = False
                return []
            
            joints = self.current_trajectory[frame]
            
            # Create line objects in first frame
            if frame == 0:
                self.arm_line, = self.ax1.plot([], [], 'b-', linewidth=4, alpha=0.8, label='Robot Arm')
                self.arm_joints, = self.ax1.plot([], [], 'bo', markersize=8)
                self.ee_trail, = self.ax1.plot([], [], 'r--', alpha=0.7, linewidth=2, label='EE Trail')
                self.joint_trail, = self.ax2.plot([], [], 'b-', linewidth=2, label='Joint Path')
                self.joint_current, = self.ax2.plot([], [], 'ro', markersize=8, label='Current')
                
                # Clear existing legends and create new ones to avoid duplicates
                self.ax1.clear()
                self.ax1.set_xlim(-2.5, 2.5)
                self.ax1.set_ylim(-2.5, 2.5)
                self.ax1.set_aspect('equal')
                self.ax1.grid(True, alpha=0.3)
                mode_text = "FAST" if self.fast_mode else "FULL"
                self.ax1.set_title(f'Workspace ({mode_text} mode) - Click to set goal', fontsize=14)
                
                # Re-add workspace circle
                circle = plt.Circle((0, 0), 2, fill=False, linestyle='--', color='gray', alpha=0.5)
                self.ax1.add_patch(circle)
                
                # Re-add goal point if it exists
                if self.goal_pos is not None:
                    self.ax1.plot(self.goal_pos[0], self.goal_pos[1], 'ro', markersize=10, label='Goal')
                
                # Re-create line objects after clearing
                self.arm_line, = self.ax1.plot([], [], 'b-', linewidth=4, alpha=0.8, label='Robot Arm')
                self.arm_joints, = self.ax1.plot([], [], 'bo', markersize=8)
                self.ee_trail, = self.ax1.plot([], [], 'r--', alpha=0.7, linewidth=2, label='EE Trail')
                
                # Clear joint space plot
                self.ax2.clear()
                self.ax2.set_xlim(-np.pi, np.pi)
                self.ax2.set_ylim(-np.pi, np.pi)
                self.ax2.grid(True, alpha=0.3)
                self.ax2.set_xlabel('θ₁ (rad)')
                self.ax2.set_ylabel('θ₂ (rad)')
                self.ax2.set_title('Joint Space', fontsize=14)
                
                # Re-create joint space objects
                self.joint_trail, = self.ax2.plot([], [], 'b-', linewidth=2, label='Joint Path')
                self.joint_current, = self.ax2.plot([], [], 'ro', markersize=8, label='Current')
                
                # Update legends only once
                self.ax1.legend()
                self.ax2.legend()
            
            # Draw robot arm
            x1 = np.cos(joints[0])
            y1 = np.sin(joints[0])
            x2 = x1 + np.cos(joints[0] + joints[1])
            y2 = y1 + np.sin(joints[0] + joints[1])
            
            if self.arm_line is not None:
                self.arm_line.set_data([0, x1, x2], [0, y1, y2])
                self.arm_joints.set_data([0, x1, x2], [0, y1, y2])
                
                # End-effector trajectory
                if frame > 0:
                    self.ee_trail.set_data(self.ee_positions[:frame+1, 0], self.ee_positions[:frame+1, 1])
                
                # Joint space trajectory
                if frame > 0:
                    self.joint_trail.set_data(self.current_trajectory[:frame+1, 0], self.current_trajectory[:frame+1, 1])
                self.joint_current.set_data([joints[0]], [joints[1]])
            
            # Progress output (20% intervals)
            progress = (frame + 1) / len(self.current_trajectory) * 100
            if frame % max(1, len(self.current_trajectory) // 5) == 0 or frame == len(self.current_trajectory) - 1:
                print(f"Animation progress: {progress:.1f}%")

            if progress == 100:
                self.animation_running = False
            
            return []
        
        # Create and run animation
        try:
            self.anim = animation.FuncAnimation(
                self.fig, 
                animate_func, 
                frames=len(self.current_trajectory),
                interval=100,  # 100ms per frame
                blit=False,
                repeat=False
            )
            
            # Force drawing
            plt.draw()
            
        except Exception as e:
            print(f"Animation error: {e}")
            self.animation_running = False
            # Fall back to static visualization on animation failure
            self.show_static_result()
    
    def show_static_result(self):
        """Display static result on animation failure"""
        if self.current_trajectory is None:
            return
        
        print("Showing static result...")
        
        # Calculate EE trajectory
        ee_positions = []
        for joints in self.current_trajectory:
            x, y = self.dataset.forward_kinematics(joints[0], joints[1])
            ee_positions.append([x, y])
        ee_positions = np.array(ee_positions)
        
        # Display full trajectory
        self.ax1.plot(ee_positions[:, 0], ee_positions[:, 1], 'r-', linewidth=2, label='EE Path')
        
        # Final robot arm pose
        final_joints = self.current_trajectory[-1]
        x1 = np.cos(final_joints[0])
        y1 = np.sin(final_joints[0])
        x2 = x1 + np.cos(final_joints[0] + final_joints[1])
        y2 = y1 + np.sin(final_joints[0] + final_joints[1])
        
        self.ax1.plot([0, x1, x2], [0, y1, y2], 'b-', linewidth=4, alpha=0.8, label='Final Pose')
        self.ax1.plot([0, x1, x2], [0, y1, y2], 'bo', markersize=8)
        
        # Joint space trajectory
        self.ax2.plot(self.current_trajectory[:, 0], self.current_trajectory[:, 1], 'b-', linewidth=2, label='Joint Path')
        self.ax2.plot(final_joints[0], final_joints[1], 'ro', markersize=8, label='Final')
        
        self.ax1.legend()
        self.ax2.legend()
        plt.draw()
        
        self.animation_running = False
    
    def run(self):
        """Run demo - mixed terminal and GUI approach"""
        print("=== Robot Arm Trajectory Generation Demo ===")
        print("Instructions:")
        print("• Click anywhere in the workspace (left plot) to set a goal")
        print("• Terminal commands:")
        print("  - 'r' or 'random': Generate random goal")
        print("  - 'c' or 'clear': Clear and reset")
        print("  - 't' or 'toggle': Toggle FAST/FULL mode")
        print("  - 's' or 'stop': Stop current animation")
        print("  - 'h' or 'help': Show help")
        print("  - 'q' or 'quit': Exit demo")
        print("  - 'x y' (e.g. '1.5 0.8'): Set goal at coordinates")
        print(f"\nCurrent mode: {'FAST' if self.fast_mode else 'FULL'} sampling")
        print("=" * 50)
        print("Ready! Click on the plot or type commands in terminal.")
        
        # Start thread for terminal input
        import threading
        import queue
        
        self.command_queue = queue.Queue()
        self.running = True
        
        def input_thread():
            """Terminal input processing thread"""
            while self.running:
                try:
                    command = input().strip().lower()
                    if command:
                        self.command_queue.put(command)
                except (EOFError, KeyboardInterrupt):
                    self.command_queue.put('q')
                    break
        
        # Start input thread
        input_thread_obj = threading.Thread(target=input_thread, daemon=True)
        input_thread_obj.start()
        
        # Set up command processing timer
        def process_commands():
            """Process terminal commands"""
            try:
                command = self.command_queue.get_nowait()
                
                if command == 'q' or command == 'quit':
                    print("Quitting...")
                    self.running = False
                    plt.close('all')
                elif command == 'r' or command == 'random':
                    self.random_goal()
                elif command == 'c' or command == 'clear':
                    self.clear_all()
                elif command == 't' or command == 'toggle':
                    self.toggle_speed()
                elif command == 's' or command == 'stop':
                    self.stop_current_animation()
                elif command == 'h' or command == 'help':
                    self.show_help()
                else:
                    # Handle coordinate input (e.g., "1.5 0.8")
                    self.try_parse_coordinates(command)
                    
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Command processing error: {e}")
        
        # Process commands using matplotlib timer
        self.command_timer = self.fig.canvas.new_timer(interval=100)
        self.command_timer.add_callback(process_commands)
        self.command_timer.start()
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            self.running = False
            if hasattr(self, 'command_timer'):
                self.command_timer.stop()
    
    def try_parse_coordinates(self, command):
        """Try to parse coordinate input (e.g., "1.5 0.8")"""
        try:
            parts = command.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                print(f"Setting goal at: ({x:.2f}, {y:.2f})")
                
                # Check if position is reachable
                distance = np.sqrt(x**2 + y**2)
                if distance <= 2.0 and distance >= 0.1:
                    self.goal_pos = np.array([x, y])
                    self.generate_trajectory()
                else:
                    print(f"Goal position ({x:.2f}, {y:.2f}) is not reachable!")
            else:
                print("Unknown command. Type 'h' for help.")
        except ValueError:
            print("Invalid coordinates or unknown command. Type 'h' for help.")

def visualize_trajectory(model, ddpm, dataset, save_path, epoch):
    """Visualize model performance - using saved model"""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Epoch {epoch} - Robot Arm Trajectory Generation', fontsize=16)
    
    with torch.no_grad():
        for i in range(6):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Generate test sample
            start_joints = np.random.uniform(-np.pi, np.pi, 2)
            goal_pos, goal_joints = dataset.sample_reachable_position()
            condition = torch.FloatTensor(np.concatenate([start_joints, goal_pos])).unsqueeze(0).to(device)
            
            # Generate trajectory - using saved model's sampling method
            shape = (1, 2, dataset.seq_length)
            generated_traj = ddpm.sample(shape, condition, device, fast_sampling=False)
            generated_traj = generated_traj.squeeze().cpu().numpy().T  # (seq_length, 2)
            
            # Calculate end-effector trajectory with forward kinematics
            ee_positions = []
            for joints in generated_traj:
                x, y = dataset.forward_kinematics(joints[0], joints[1])
                ee_positions.append([x, y])
            ee_positions = np.array(ee_positions)
            
            # Calculate start position's end-effector
            start_x, start_y = dataset.forward_kinematics(start_joints[0], start_joints[1])
            
            # Plot
            ax.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', label='Generated Path', linewidth=2)
            ax.plot(start_x, start_y, 'go', markersize=8, label='Start')
            ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=8, label='Goal')
            
            # Draw robot arm (final pose)
            final_joints = generated_traj[-1]
            x1 = np.cos(final_joints[0])
            y1 = np.sin(final_joints[0])
            x2 = x1 + np.cos(final_joints[0] + final_joints[1])
            y2 = y1 + np.sin(final_joints[0] + final_joints[1])
            
            ax.plot([0, x1, x2], [0, y1, y2], 'k-', linewidth=3, alpha=0.7)
            ax.plot([0, x1, x2], [0, y1, y2], 'ko', markersize=6)
            
            # Display workspace
            circle = plt.Circle((0, 0), 2, fill=False, linestyle='--', color='gray', alpha=0.5)
            ax.add_patch(circle)
            
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title(f'Test {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Visualize trajectory in joint space as well
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    with torch.no_grad():
        # Visualize joint trajectory with one example
        start_joints = np.random.uniform(-np.pi, np.pi, 2)
        goal_pos, goal_joints = dataset.sample_reachable_position()
        condition = torch.FloatTensor(np.concatenate([start_joints, goal_pos])).unsqueeze(0).to(device)
        
        shape = (1, 2, dataset.seq_length)
        generated_traj = ddpm.sample(shape, condition, device, fast_sampling=False)
        generated_traj = generated_traj.squeeze().cpu().numpy().T
        
        # Joint angle trajectory
        axes[0].plot(generated_traj[:, 0], label='θ₁', linewidth=2)
        axes[0].plot(generated_traj[:, 1], label='θ₂', linewidth=2)
        axes[0].axhline(y=start_joints[0], color='blue', linestyle='--', alpha=0.5, label='θ₁ start')
        axes[0].axhline(y=start_joints[1], color='orange', linestyle='--', alpha=0.5, label='θ₂ start')
        axes[0].axhline(y=goal_joints[0], color='blue', linestyle=':', alpha=0.5, label='θ₁ goal')
        axes[0].axhline(y=goal_joints[1], color='orange', linestyle=':', alpha=0.5, label='θ₂ goal')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Joint Angle (rad)')
        axes[0].set_title('Joint Space Trajectory')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Path in joint space
        axes[1].plot(generated_traj[:, 0], generated_traj[:, 1], 'b-', linewidth=2, label='Generated Path')
        axes[1].plot(start_joints[0], start_joints[1], 'go', markersize=8, label='Start')
        axes[1].plot(goal_joints[0], goal_joints[1], 'ro', markersize=8, label='Goal')
        axes[1].set_xlabel('θ₁ (rad)')
        axes[1].set_ylabel('θ₂ (rad)')
        axes[1].set_title('Joint Space Path')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    joint_save_path = save_path.replace('.png', '_joints.png')
    plt.tight_layout()
    plt.savefig(joint_save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_model():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 200
    seq_length = 50
    
    # Create save directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Dataset and dataloader
    dataset = RobotArmDataset(num_samples=5000, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = UNet1D(input_dim=2, cond_dim=4, dim=64).to(device)
    ddpm = DDPM(model, timesteps=1000)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move DDPM parameters to device
    for name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                 'sqrt_recip_alphas', 'posterior_variance']:
        setattr(ddpm, name, getattr(ddpm, name).to(device))
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            trajectory = batch['trajectory'].to(device)  # (batch, seq_len, 2)
            condition = batch['condition'].to(device)    # (batch, 4)
            
            # (batch, seq_len, 2) -> (batch, 2, seq_len)
            trajectory = trajectory.transpose(1, 2)
            
            # Random timestep
            t = torch.randint(0, ddpm.timesteps, (trajectory.size(0),), device=device)
            
            # Add noise
            noise = torch.randn_like(trajectory)
            x_noisy = ddpm.q_sample(trajectory, t, noise)
            
            # Predict noise
            noise_pred = model(x_noisy, t, condition)
            
            # Calculate loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
        
        # Save model and visualize every 25 epochs
        if (epoch + 1) % 25 == 0:
            # Save model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch {epoch+1}')
            
            # Create separate model and ddpm for evaluation and load checkpoint
            eval_model = UNet1D(input_dim=2, cond_dim=4, dim=64).to(device)
            eval_ddpm = DDPM(eval_model, timesteps=1000)
            
            # Move DDPM parameters to device
            for name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                         'sqrt_recip_alphas', 'posterior_variance']:
                setattr(eval_ddpm, name, getattr(eval_ddpm, name).to(device))
            
            # Load saved checkpoint
            eval_checkpoint = torch.load(f'checkpoints/model_epoch_{epoch+1}.pth', map_location=device)
            eval_model.load_state_dict(eval_checkpoint['model_state_dict'])
            eval_model.eval()
            
            # Visualization
            viz_path = f'visualizations/trajectory_epoch_{epoch+1}.png'
            visualize_trajectory(eval_model, eval_ddpm, dataset, viz_path, epoch+1)
            print(f'Visualization saved: {viz_path}')
    
    print("Training completed!")

def load_latest_model():
    """Find the most recent model"""
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def main():
    parser = argparse.ArgumentParser(description='Robot Arm Trajectory Diffusion Model')
    parser.add_argument('--mode', choices=['train', 'demo'], default='train',
                        help='Run mode: train or demo')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (for demo mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training mode...")
        train_model()
    
    elif args.mode == 'demo':
        model_path = args.model_path
        
        if model_path is None:
            # Auto-load the most recent model
            model_path = load_latest_model()
            if model_path is None:
                print("No trained model found! Please train the model first.")
                print("Run: python robot_arm_diffusion.py --mode train")
                return
            print(f"Loading latest model: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
        
        try:
            demo = InteractiveDemo(model_path, fast_mode=False)  # Use FULL mode as default
            demo.run()
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        except Exception as e:
            print(f"Error running demo: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()