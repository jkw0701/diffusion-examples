import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
from typing import List, Tuple, Optional
import random

# ============================================================================
# 1. Improved A* Pathfinding with Multiple Path Options
# ============================================================================

class ImprovedAStarPathfinder:
    def __init__(self, obstacle_map):
        self.obstacle_map = obstacle_map
        self.height, self.width = obstacle_map.shape
        
    def heuristic(self, a, b):
        """Euclidean distance for better pathfinding"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions with costs"""
        neighbors = []
        # 8-directional movement with different costs
        directions = [
            (-1,-1, 1.414), (-1,0, 1.0), (-1,1, 1.414), 
            (0,-1, 1.0),                  (0,1, 1.0), 
            (1,-1, 1.414),  (1,0, 1.0),   (1,1, 1.414)
        ]
        
        for dx, dy, cost in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            if (0 <= new_x < self.height and 
                0 <= new_y < self.width and 
                self.obstacle_map[new_x, new_y] == 0):
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def find_path(self, start, goal):
        """Improved A* with better heuristic"""
        start, goal = tuple(start), tuple(goal)
        
        if start == goal:
            return [start]
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor, cost in self.get_neighbors(current):
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found

# ============================================================================
# 2. Improved Data Generation with Better Diversity
# ============================================================================

def generate_diverse_obstacle_map(size=64, difficulty='medium'):
    """Generate diverse obstacle maps with different difficulty levels"""
    obstacle_map = np.zeros((size, size), dtype=np.float32)
    
    if difficulty == 'easy':
        num_obstacles = np.random.randint(3, 8)
        max_size = 6
    elif difficulty == 'medium':
        num_obstacles = np.random.randint(5, 12)
        max_size = 8
    else:  # hard
        num_obstacles = np.random.randint(8, 15)
        max_size = 10
    
    for _ in range(num_obstacles):
        # Mix of rectangular and L-shaped obstacles
        if np.random.random() < 0.7:
            # Rectangular obstacles
            x = np.random.randint(2, size-max_size-2)
            y = np.random.randint(2, size-max_size-2)
            w = np.random.randint(2, max_size)
            h = np.random.randint(2, max_size)
            
            obstacle_map[y:y+h, x:x+w] = 1
        else:
            # L-shaped obstacles for more complex navigation
            x = np.random.randint(2, size-max_size-2)
            y = np.random.randint(2, size-max_size-2)
            w1, h1 = np.random.randint(3, max_size, 2)
            w2, h2 = np.random.randint(2, max_size//2, 2)
            
            # Horizontal part
            obstacle_map[y:y+h2, x:x+w1] = 1
            # Vertical part
            obstacle_map[y:y+h1, x:x+w2] = 1
    
    return obstacle_map

def compute_improved_direction(start_pos, goal_pos, obstacle_map):
    """Improved direction computation with multiple strategies"""
    pathfinder = ImprovedAStarPathfinder(obstacle_map)
    path = pathfinder.find_path(start_pos, goal_pos)
    
    if len(path) > 1:
        # Look ahead 2-3 steps for smoother movement
        lookahead = min(3, len(path) - 1)
        target_pos = np.array(path[lookahead])
        direction = target_pos - start_pos
        
        # Normalize direction to unit vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        return direction.astype(np.float32)
    else:
        # Direct direction to goal if no obstacles or already at goal
        direction = goal_pos - start_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0])
        
        return direction.astype(np.float32)

def generate_improved_dataset(num_samples=10000, map_size=64):
    """Generate improved dataset with better diversity"""
    dataset = []
    difficulties = ['easy', 'medium', 'hard']
    
    print(f"Generating {num_samples} improved navigation samples...")
    
    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} samples")
        
        # Vary difficulty
        difficulty = np.random.choice(difficulties)
        obstacle_map = generate_diverse_obstacle_map(map_size, difficulty)
        free_positions = np.argwhere(obstacle_map == 0)
        
        if len(free_positions) < 10:  # Ensure enough free space
            continue
        
        # Strategic positioning: avoid too close start/goal
        max_attempts = 50
        for attempt in range(max_attempts):
            indices = np.random.choice(len(free_positions), 2, replace=False)
            start_pos = free_positions[indices[0]]
            goal_pos = free_positions[indices[1]]
            
            # Ensure minimum distance between start and goal
            distance = np.linalg.norm(goal_pos - start_pos)
            if distance > map_size * 0.2:  # At least 20% of map size
                break
        
        if attempt == max_attempts - 1:
            continue
        
        # Compute improved direction
        direction = compute_improved_direction(start_pos, goal_pos, obstacle_map)
        
        # Skip if direction is zero (edge case)
        if np.linalg.norm(direction) < 0.1:
            continue
        
        # Normalize positions to [0, 1] but keep directions as unit vectors
        norm_start = start_pos.astype(np.float32) / (map_size - 1)
        norm_goal = goal_pos.astype(np.float32) / (map_size - 1)
        
        dataset.append({
            'current_pos': norm_start,
            'goal_pos': norm_goal,
            'obstacle_map': obstacle_map[np.newaxis, :, :],
            'direction': direction  # Keep as unit vector
        })
    
    print(f"Improved dataset generation complete: {len(dataset)} samples")
    return dataset

# ============================================================================
# 3. Dataset Class (Missing from previous code)
# ============================================================================

class NavigationDataset(Dataset):
    """Dataset class for navigation training"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'current_pos': torch.FloatTensor(sample['current_pos']),
            'goal_pos': torch.FloatTensor(sample['goal_pos']),
            'obstacle_map': torch.FloatTensor(sample['obstacle_map']),
            'direction': torch.FloatTensor(sample['direction'])
        }

# ============================================================================
# 4. Enhanced Model Architecture
# ============================================================================

class EnhancedVisionEncoder(nn.Module):
    """Enhanced CNN encoder with attention mechanism"""
    def __init__(self, input_size=64, output_dim=256):
        super(EnhancedVisionEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8)  # -> 8x8
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.feature_proj = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [batch_size, 1, 64, 64]
        conv_features = self.conv_layers(x)
        
        # Apply attention
        attention_weights = self.attention(conv_features)
        attended_features = conv_features * attention_weights
        
        flattened = attended_features.view(attended_features.size(0), -1)
        return self.feature_proj(flattened)

class EnhancedPositionEncoder(nn.Module):
    """Enhanced position encoder with relative encoding"""
    def __init__(self, input_dim=4, output_dim=128):
        super(EnhancedPositionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Relative position encoding
        self.relative_encoder = nn.Sequential(
            nn.Linear(2, 32),  # [dx, dy]
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, current_pos, goal_pos):
        # Absolute positions
        pos_input = torch.cat([current_pos, goal_pos], dim=-1)
        abs_features = self.encoder(pos_input)
        
        # Relative position
        relative_pos = goal_pos - current_pos
        rel_features = self.relative_encoder(relative_pos)
        
        # Combine
        combined = torch.cat([abs_features, rel_features], dim=-1)
        return combined

class ImprovedCrossModalNavigator(nn.Module):
    """Improved navigation model with better fusion"""
    def __init__(self, map_size=64):
        super(ImprovedCrossModalNavigator, self).__init__()
        
        self.map_size = map_size
        
        # Enhanced encoders
        self.position_encoder = EnhancedPositionEncoder(input_dim=4, output_dim=128)
        self.vision_encoder = EnhancedVisionEncoder(input_size=map_size, output_dim=256)
        
        # Multi-layer fusion with residual connections
        self.fusion_layers = nn.ModuleList([
            nn.Linear(128 + 64 + 256, 256),  # pos + rel_pos + vision
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh()  # Constrain output to [-1, 1]
        )
        
        # Confidence prediction (optional)
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_pos, goal_pos, obstacle_map, return_confidence=False):
        # Encode inputs
        pos_features = self.position_encoder(current_pos, goal_pos)
        vision_features = self.vision_encoder(obstacle_map)
        
        # Fuse features with residual connections
        x = torch.cat([pos_features, vision_features], dim=-1)
        
        for i, layer in enumerate(self.fusion_layers):
            if i == 0:
                x = self.activation(layer(x))
            else:
                residual = x
                x = layer(x)
                if x.shape == residual.shape:
                    x = x + residual  # Residual connection
                x = self.activation(x)
            x = self.dropout(x)
        
        # Predict direction
        direction = self.direction_head(x)
        
        if return_confidence:
            confidence = self.confidence_head(x)
            return direction, confidence
        
        return direction

# ============================================================================
# 5. Improved Loss Functions
# ============================================================================

def improved_collision_penalty(predicted_directions, current_positions, obstacle_maps, 
                             map_size=64, penalty_radius=2):
    """Improved collision penalty with safety margin"""
    batch_size = predicted_directions.shape[0]
    device = predicted_directions.device  # Get device from input tensor
    penalties = []
    
    for i in range(batch_size):
        # Convert to map coordinates
        current_map_pos = current_positions[i] * (map_size - 1)
        
        # Predicted next position (scale direction appropriately)
        direction_magnitude = torch.norm(predicted_directions[i])
        if direction_magnitude > 0:
            unit_direction = predicted_directions[i] / direction_magnitude
            # Take a step of size 2-3 pixels
            next_pos = current_map_pos + unit_direction * 2.5
        else:
            next_pos = current_map_pos
        
        # Check multiple points around predicted position for safety margin
        penalty = torch.tensor(0.0, device=device, dtype=torch.float32)  # Initialize as tensor
        # Create tensors on the same device
        offsets = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0], 
            [0.0, 1.0],
            [0.0, -1.0]
        ], device=device, dtype=next_pos.dtype)
        
        num_collisions = 0
        for offset in offsets:
            point = next_pos + offset
            x = torch.clamp(point[0], 0, map_size - 1).long()
            y = torch.clamp(point[1], 0, map_size - 1).long()
            
            if obstacle_maps[i, 0, y, x] > 0.5:
                num_collisions += 1
        
        penalty = torch.tensor(num_collisions / len(offsets), device=device, dtype=torch.float32)
        penalties.append(penalty)
    
    return torch.stack(penalties)

def improved_navigation_loss(predicted_directions, target_directions, current_positions, 
                           goal_positions, obstacle_maps, map_size=64):
    """Improved loss function with better balancing"""
    
    # 1. Direction similarity loss (cosine similarity)
    pred_norm = F.normalize(predicted_directions, p=2, dim=1)
    target_norm = F.normalize(target_directions, p=2, dim=1)
    cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    direction_loss = (1 - cosine_sim).mean()
    
    # 2. Magnitude loss (encourage appropriate step size)
    pred_magnitude = torch.norm(predicted_directions, p=2, dim=1)
    target_magnitude = torch.norm(target_directions, p=2, dim=1)
    magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)
    
    # 3. Improved safety penalty
    collision_penalties = improved_collision_penalty(
        predicted_directions, current_positions, obstacle_maps, map_size
    )
    safety_loss = collision_penalties.mean()
    
    # 4. Goal attraction loss (encourage movement toward goal)
    goal_direction = goal_positions - current_positions
    goal_direction_norm = F.normalize(goal_direction, p=2, dim=1)
    goal_alignment = F.cosine_similarity(pred_norm, goal_direction_norm, dim=1)
    goal_loss = (1 - goal_alignment).mean()
    
    # Balanced combination
    total_loss = (0.4 * direction_loss + 
                  0.2 * magnitude_loss + 
                  0.2 * safety_loss + 
                  0.2 * goal_loss)
    
    return total_loss, direction_loss, magnitude_loss, safety_loss, goal_loss

# ============================================================================
# 6. Enhanced Training Function
# ============================================================================

def train_improved_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """Enhanced training with better optimization"""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    print(f"Enhanced training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = {
            'total': 0, 'direction': 0, 'magnitude': 0, 
            'safety': 0, 'goal': 0, 'batches': 0
        }
        
        for batch in train_loader:
            current_pos = batch['current_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            obstacle_map = batch['obstacle_map'].to(device)
            target_direction = batch['direction'].to(device)
            
            optimizer.zero_grad()
            predicted_direction = model(current_pos, goal_pos, obstacle_map)
            
            total_loss, dir_loss, mag_loss, safety_loss, goal_loss = improved_navigation_loss(
                predicted_direction, target_direction, current_pos, goal_pos, obstacle_map
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_metrics['total'] += total_loss.item()
            train_metrics['direction'] += dir_loss.item()
            train_metrics['magnitude'] += mag_loss.item()
            train_metrics['safety'] += safety_loss.item()
            train_metrics['goal'] += goal_loss.item()
            train_metrics['batches'] += 1
        
        # Validation phase
        model.eval()
        val_metrics = {
            'total': 0, 'direction': 0, 'magnitude': 0, 
            'safety': 0, 'goal': 0, 'batches': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                current_pos = batch['current_pos'].to(device)
                goal_pos = batch['goal_pos'].to(device)
                obstacle_map = batch['obstacle_map'].to(device)
                target_direction = batch['direction'].to(device)
                
                predicted_direction = model(current_pos, goal_pos, obstacle_map)
                total_loss, dir_loss, mag_loss, safety_loss, goal_loss = improved_navigation_loss(
                    predicted_direction, target_direction, current_pos, goal_pos, obstacle_map
                )
                
                val_metrics['total'] += total_loss.item()
                val_metrics['direction'] += dir_loss.item()
                val_metrics['magnitude'] += mag_loss.item()
                val_metrics['safety'] += safety_loss.item()
                val_metrics['goal'] += goal_loss.item()
                val_metrics['batches'] += 1
        
        # Calculate averages
        for key in ['total', 'direction', 'magnitude', 'safety', 'goal']:
            train_metrics[key] /= train_metrics['batches']
            val_metrics[key] /= val_metrics['batches']
        
        train_losses.append(train_metrics['total'])
        val_losses.append(val_metrics['total'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['total'])
        
        # Early stopping
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train={train_metrics['total']:.4f} "
                  f"(Dir:{train_metrics['direction']:.4f}, "
                  f"Mag:{train_metrics['magnitude']:.4f}, "
                  f"Safe:{train_metrics['safety']:.4f}, "
                  f"Goal:{train_metrics['goal']:.4f}) | "
                  f"Val={val_metrics['total']:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return train_losses, val_losses

# ============================================================================
# 7. Enhanced Evaluation
# ============================================================================

def evaluate_improved_model(model, test_loader, device='cpu'):
    """Enhanced evaluation with multiple metrics"""
    model.eval()
    model = model.to(device)
    
    metrics = {
        'direction_cosine_sim': 0,
        'magnitude_error': 0,
        'collision_rate': 0,
        'goal_alignment': 0,
        'total_samples': 0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            current_pos = batch['current_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            obstacle_map = batch['obstacle_map'].to(device)
            target_direction = batch['direction'].to(device)
            
            predicted_direction = model(current_pos, goal_pos, obstacle_map)
            batch_size = current_pos.shape[0]
            
            # Direction similarity
            pred_norm = F.normalize(predicted_direction, p=2, dim=1)
            target_norm = F.normalize(target_direction, p=2, dim=1)
            cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
            metrics['direction_cosine_sim'] += cosine_sim.sum().item()
            
            # Magnitude error
            pred_mag = torch.norm(predicted_direction, p=2, dim=1)
            target_mag = torch.norm(target_direction, p=2, dim=1)
            mag_error = F.mse_loss(pred_mag, target_mag, reduction='sum')
            metrics['magnitude_error'] += mag_error.item()
            
            # Collision rate
            collision_penalties = improved_collision_penalty(
                predicted_direction, current_pos, obstacle_map
            )
            metrics['collision_rate'] += (collision_penalties > 0).sum().item()
            
            # Goal alignment
            goal_direction = goal_pos - current_pos
            goal_direction_norm = F.normalize(goal_direction, p=2, dim=1)
            goal_alignment = F.cosine_similarity(pred_norm, goal_direction_norm, dim=1)
            metrics['goal_alignment'] += goal_alignment.sum().item()
            
            metrics['total_samples'] += batch_size
    
    # Calculate averages
    total_samples = metrics['total_samples']
    avg_cosine_sim = metrics['direction_cosine_sim'] / total_samples
    avg_mag_error = metrics['magnitude_error'] / total_samples
    collision_rate = metrics['collision_rate'] / total_samples
    avg_goal_alignment = metrics['goal_alignment'] / total_samples
    
    print(f"\nEnhanced Evaluation Results:")
    print(f"Direction Cosine Similarity: {avg_cosine_sim:.4f} (1.0 = perfect)")
    print(f"Magnitude Error: {avg_mag_error:.6f}")
    print(f"Collision Rate: {collision_rate:.4f} ({collision_rate*100:.2f}%)")
    print(f"Goal Alignment: {avg_goal_alignment:.4f} (1.0 = perfect)")
    
    return {
        'direction_similarity': avg_cosine_sim,
        'magnitude_error': avg_mag_error,
        'collision_rate': collision_rate,
        'goal_alignment': avg_goal_alignment
    }

# ============================================================================
# 8. Enhanced Visualization
# ============================================================================

def visualize_improved_prediction(model, test_sample, device='cpu'):
    """Enhanced visualization with confidence and attention"""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        current_pos = test_sample['current_pos'].unsqueeze(0).to(device)
        goal_pos = test_sample['goal_pos'].unsqueeze(0).to(device)
        obstacle_map = test_sample['obstacle_map'].unsqueeze(0).to(device)
        target_direction = test_sample['direction']
        
        predicted_direction = model(current_pos, goal_pos, obstacle_map)
        predicted_direction = predicted_direction.cpu().squeeze().numpy()
        
        # Convert to map coordinates
        map_size = 64
        current_map_pos = current_pos.cpu().squeeze().numpy() * (map_size - 1)
        goal_map_pos = goal_pos.cpu().squeeze().numpy() * (map_size - 1)
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        # Obstacle map with positions
        plt.subplot(1, 3, 1)
        obstacle_vis = obstacle_map.cpu().squeeze().numpy()
        plt.imshow(obstacle_vis, cmap='gray_r', origin='lower', alpha=0.8)
        plt.plot(current_map_pos[1], current_map_pos[0], 'go', markersize=12, label='Start')
        plt.plot(goal_map_pos[1], goal_map_pos[0], 'ro', markersize=12, label='Goal')
        plt.title('Environment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Target vs Predicted directions
        plt.subplot(1, 3, 2)
        plt.imshow(obstacle_vis, cmap='gray_r', origin='lower', alpha=0.3)
        plt.plot(current_map_pos[1], current_map_pos[0], 'go', markersize=10)
        
        # Target direction (blue)
        scale = 15
        plt.arrow(current_map_pos[1], current_map_pos[0], 
                 target_direction[1]*scale, target_direction[0]*scale,
                 head_width=2, head_length=2, fc='blue', ec='blue', 
                 alpha=0.7, label='Target', linewidth=3)
        
        # Predicted direction (red)
        plt.arrow(current_map_pos[1], current_map_pos[0], 
                 predicted_direction[1]*scale, predicted_direction[0]*scale,
                 head_width=2, head_length=2, fc='red', ec='red', 
                 alpha=0.7, label='Predicted', linewidth=3)
        
        plt.title('Direction Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics
        plt.subplot(1, 3, 3)
        cosine_sim = np.dot(target_direction, predicted_direction) / (
            np.linalg.norm(target_direction) * np.linalg.norm(predicted_direction)
        )
        
        metrics_text = f"""
        Cosine Similarity: {cosine_sim:.3f}
        
        Target Direction:
        [{target_direction[0]:.3f}, {target_direction[1]:.3f}]
        Magnitude: {np.linalg.norm(target_direction):.3f}
        
        Predicted Direction:
        [{predicted_direction[0]:.3f}, {predicted_direction[1]:.3f}]
        Magnitude: {np.linalg.norm(predicted_direction):.3f}
        """
        
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Metrics')
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# 9. Main Enhanced Training Script
# ============================================================================

def main_improved():
    """Main improved training script"""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate improved dataset
    print("Generating improved dataset...")
    dataset = generate_improved_dataset(num_samples=8000, map_size=64)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create data loaders
    train_dataset = NavigationDataset(train_data)
    val_dataset = NavigationDataset(val_data)
    test_dataset = NavigationDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create improved model
    model = ImprovedCrossModalNavigator(map_size=64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Improved model parameters: {total_params:,}")
    
    # Train model
    print("\nStarting improved training...")
    train_losses, val_losses = train_improved_model(
        model, train_loader, val_loader, 
        num_epochs=80, lr=0.001, device=device
    )
    
    # Evaluate model
    print("\nEvaluating improved model...")
    metrics = evaluate_improved_model(model, test_loader, device=device)
    
    # Visualize predictions
    print("\nVisualizing improved predictions...")
    for i in range(3):
        test_sample = test_dataset[i * 100]  # Spread out samples
        visualize_improved_prediction(model, test_sample, device=device)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    metric_names = ['Direction Sim', 'Goal Align', 'Collision Rate']
    metric_values = [metrics['direction_similarity'], 
                    metrics['goal_alignment'], 
                    metrics['collision_rate']]
    colors = ['green', 'blue', 'red']
    
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Final Performance Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    model = main_improved()