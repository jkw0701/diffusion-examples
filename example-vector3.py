import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from collections import deque
import random
import os
from typing import List, Tuple, Optional
import json
import heapq

# ============================================================================
# 1. A* Pathfinding Algorithm (Integrated from Step 3)
# ============================================================================

class AStarPathfinder:
    """A* pathfinding algorithm for optimal path planning"""
    def __init__(self, obstacle_map):
        self.obstacle_map = obstacle_map
        self.height, self.width = obstacle_map.shape
        
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
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
# 2. Synthetic Real-world Scene Generation
# ============================================================================

class SyntheticSceneGenerator:
    """Generate synthetic indoor/outdoor scenes with semantic labels"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.indoor_colors = {
            'floor': (139, 69, 19),      # Brown floor
            'wall': (128, 128, 128),     # Gray walls  
            'furniture': (101, 67, 33),  # Dark brown furniture
            'door': (160, 82, 45),       # Saddle brown doors
            'background': (245, 245, 220) # Beige background
        }
        
        self.outdoor_colors = {
            'ground': (34, 139, 34),     # Forest green ground
            'path': (184, 134, 11),      # Dark goldenrod path
            'tree': (0, 100, 0),         # Dark green trees
            'building': (105, 105, 105), # Dim gray buildings
            'sky': (135, 206, 235),      # Sky blue
            'rock': (105, 105, 105)      # Gray rocks
        }
    
    def generate_indoor_scene(self):
        """Generate synthetic indoor scene"""
        img = Image.new('RGB', self.image_size, self.indoor_colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Floor
        floor_height = self.image_size[1] // 3
        draw.rectangle([0, self.image_size[1] - floor_height, 
                       self.image_size[0], self.image_size[1]], 
                      fill=self.indoor_colors['floor'])
        
        # Walls (obstacles)
        wall_positions = []
        num_walls = np.random.randint(2, 5)
        
        for _ in range(num_walls):
            if np.random.random() < 0.5:  # Vertical walls
                x = np.random.randint(20, self.image_size[0] - 40)
                y1 = np.random.randint(0, self.image_size[1] - 60)
                y2 = y1 + np.random.randint(30, 80)
                thickness = np.random.randint(10, 20)
                draw.rectangle([x, y1, x + thickness, y2], 
                              fill=self.indoor_colors['wall'])
                wall_positions.append(('vertical', x, y1, thickness, y2 - y1))
            else:  # Horizontal walls
                y = np.random.randint(20, self.image_size[1] - 40)
                x1 = np.random.randint(0, self.image_size[0] - 60)
                x2 = x1 + np.random.randint(30, 80)
                thickness = np.random.randint(10, 20)
                draw.rectangle([x1, y, x2, y + thickness], 
                              fill=self.indoor_colors['wall'])
                wall_positions.append(('horizontal', x1, y, x2 - x1, thickness))
        
        # Furniture (obstacles)
        furniture_positions = []
        num_furniture = np.random.randint(1, 4)
        
        for _ in range(num_furniture):
            x = np.random.randint(10, self.image_size[0] - 30)
            y = np.random.randint(10, self.image_size[1] - 30)
            w = np.random.randint(15, 25)
            h = np.random.randint(15, 25)
            draw.rectangle([x, y, x + w, y + h], 
                          fill=self.indoor_colors['furniture'])
            furniture_positions.append((x, y, w, h))
        
        return img, {'walls': wall_positions, 'furniture': furniture_positions}
    
    def generate_outdoor_scene(self):
        """Generate synthetic outdoor scene"""
        img = Image.new('RGB', self.image_size, self.outdoor_colors['sky'])
        draw = ImageDraw.Draw(img)
        
        # Ground
        ground_height = self.image_size[1] // 2
        draw.rectangle([0, self.image_size[1] - ground_height, 
                       self.image_size[0], self.image_size[1]], 
                      fill=self.outdoor_colors['ground'])
        
        # Path (free space)
        path_width = np.random.randint(40, 80)
        path_x = np.random.randint(20, self.image_size[0] - path_width - 20)
        draw.rectangle([path_x, self.image_size[1] - ground_height, 
                       path_x + path_width, self.image_size[1]], 
                      fill=self.outdoor_colors['path'])
        
        # Trees and buildings (obstacles)
        obstacle_positions = []
        num_obstacles = np.random.randint(3, 7)
        
        for _ in range(num_obstacles):
            if np.random.random() < 0.6:  # Trees (circular)
                x = np.random.randint(15, self.image_size[0] - 15)
                y = np.random.randint(15, self.image_size[1] - ground_height - 15)
                radius = np.random.randint(8, 20)
                
                # Avoid path area
                if not (path_x - radius < x < path_x + path_width + radius):
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                                fill=self.outdoor_colors['tree'])
                    obstacle_positions.append(('tree', x, y, radius))
            else:  # Buildings (rectangular)
                w = np.random.randint(20, 40)
                h = np.random.randint(30, 60)
                x = np.random.randint(0, self.image_size[0] - w)
                y = np.random.randint(0, self.image_size[1] - ground_height - h)
                
                # Avoid path area
                if not (path_x - 10 < x + w//2 < path_x + path_width + 10):
                    draw.rectangle([x, y, x + w, y + h], 
                                  fill=self.outdoor_colors['building'])
                    obstacle_positions.append(('building', x, y, w, h))
        
        return img, {'path': (path_x, path_width), 'obstacles': obstacle_positions}
    
    def create_semantic_mask(self, scene_type, scene_info):
        """Create semantic segmentation mask"""
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if scene_type == 'indoor':
            # 0: free space, 1: obstacles
            for wall_type, x, y, w, h in scene_info['walls']:
                if wall_type == 'vertical':
                    mask[y:y+h, x:x+w] = 1
                else:  # horizontal
                    mask[y:y+h, x:x+w] = 1
            
            for x, y, w, h in scene_info['furniture']:
                mask[y:y+h, x:x+w] = 1
                
        else:  # outdoor
            # Mark everything as obstacle initially
            mask[:, :] = 1
            
            # Mark path as free space
            path_x, path_width = scene_info['path']
            ground_start = self.image_size[1] // 2
            mask[ground_start:, path_x:path_x+path_width] = 0
            
            # Add obstacle details
            for obs_type, x, y, *params in scene_info['obstacles']:
                if obs_type == 'tree':
                    radius = params[0]
                    y_coords, x_coords = np.ogrid[:self.image_size[0], :self.image_size[1]]
                    dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
                    mask[dist_from_center <= radius] = 1
                else:  # building
                    w, h = params
                    mask[y:y+h, x:x+w] = 1
        
        return mask

def generate_navigation_scene():
    """Generate a complete navigation scene"""
    generator = SyntheticSceneGenerator()
    
    # Randomly choose indoor or outdoor
    scene_type = np.random.choice(['indoor', 'outdoor'])
    
    if scene_type == 'indoor':
        rgb_image, scene_info = generator.generate_indoor_scene()
    else:
        rgb_image, scene_info = generator.generate_outdoor_scene()
    
    # Create semantic mask
    semantic_mask = generator.create_semantic_mask(scene_type, scene_info)
    
    return rgb_image, semantic_mask, scene_type

# ============================================================================
# 3. Real-world Navigation Dataset
# ============================================================================

def generate_realworld_dataset(num_samples=5000, image_size=(224, 224)):
    """Generate real-world style navigation dataset"""
    dataset = []
    
    print(f"Generating {num_samples} real-world navigation samples...")
    
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"Generated {i}/{num_samples} samples")
        
        # Generate scene
        rgb_image, semantic_mask, scene_type = generate_navigation_scene()
        
        # Find free positions
        free_positions = np.argwhere(semantic_mask == 0)
        
        if len(free_positions) < 10:
            continue
        
        # Random start and goal positions
        indices = np.random.choice(len(free_positions), 2, replace=False)
        start_pos = free_positions[indices[0]]
        goal_pos = free_positions[indices[1]]
        
        # Ensure minimum distance
        distance = np.linalg.norm(goal_pos - start_pos)
        if distance < image_size[0] * 0.15:
            continue
        
        # Compute direction using A* on semantic mask
        pathfinder = AStarPathfinder(semantic_mask)
        path = pathfinder.find_path(start_pos, goal_pos)
        
        if len(path) > 1:
            # Lookahead strategy
            lookahead = min(3, len(path) - 1)
            target_pos = np.array(path[lookahead])
            direction = target_pos - start_pos
            
            # Normalize direction
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                continue
        else:
            # Direct direction to goal
            direction = goal_pos - start_pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                continue
        
        # Normalize positions to [0, 1]
        norm_start = start_pos.astype(np.float32) / np.array(image_size)
        norm_goal = goal_pos.astype(np.float32) / np.array(image_size)
        
        # Convert RGB image to tensor
        rgb_array = np.array(rgb_image)
        rgb_tensor = torch.FloatTensor(rgb_array).permute(2, 0, 1) / 255.0
        
        dataset.append({
            'rgb_image': rgb_tensor,
            'semantic_mask': semantic_mask[np.newaxis, :, :].astype(np.float32),
            'current_pos': norm_start,
            'goal_pos': norm_goal,
            'direction': direction.astype(np.float32),
            'scene_type': scene_type
        })
    
    print(f"Real-world dataset generation complete: {len(dataset)} samples")
    return dataset

class RealWorldNavigationDataset(Dataset):
    """Dataset for real-world navigation"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        rgb_image = sample['rgb_image']
        
        # Apply transforms if provided
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        return {
            'rgb_image': rgb_image,
            'semantic_mask': torch.FloatTensor(sample['semantic_mask']),
            'current_pos': torch.FloatTensor(sample['current_pos']),
            'goal_pos': torch.FloatTensor(sample['goal_pos']),
            'direction': torch.FloatTensor(sample['direction']),
            'scene_type': sample['scene_type']
        }

# ============================================================================
# 4. Advanced Vision Encoder with Semantic Segmentation
# ============================================================================

class SemanticSegmentationEncoder(nn.Module):
    """Advanced encoder with semantic segmentation capability"""
    
    def __init__(self, backbone='resnet50', pretrained=True, output_dim=512):
        super(SemanticSegmentationEncoder, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_dim = 2048
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained).features
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Semantic segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),  # 2 classes: free space, obstacle
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        )
        
        # Feature extraction for navigation
        self.nav_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(backbone_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
        # Cross-attention for position-aware features
        self.position_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, dropout=0.1
        )
        
    def forward(self, rgb_image, current_pos=None, goal_pos=None, return_segmentation=False):
        # Extract features from backbone
        features = self.backbone(rgb_image)  # [B, C, H, W]
        
        # Semantic segmentation
        seg_logits = self.seg_head(features)  # [B, 2, 224, 224]
        
        # Navigation features
        nav_features = self.nav_head(features)  # [B, output_dim]
        
        # Position-aware attention (if positions provided)
        if current_pos is not None and goal_pos is not None:
            # Create positional encoding
            pos_encoding = torch.cat([current_pos, goal_pos], dim=-1)  # [B, 4]
            pos_encoding = pos_encoding.unsqueeze(0)  # [1, B, 4]
            
            # Expand nav_features for attention
            nav_features_expanded = nav_features.unsqueeze(0)  # [1, B, output_dim]
            
            # Apply attention (simplified - in practice, need proper pos encoding)
            attended_features, _ = self.position_attention(
                nav_features_expanded, nav_features_expanded, nav_features_expanded
            )
            nav_features = attended_features.squeeze(0)  # [B, output_dim]
        
        if return_segmentation:
            return nav_features, seg_logits
        else:
            return nav_features

class RealWorldNavigator(nn.Module):
    """Complete real-world navigation model"""
    
    def __init__(self, backbone='resnet50', pretrained=True):
        super(RealWorldNavigator, self).__init__()
        
        # Vision encoder with semantic segmentation
        self.vision_encoder = SemanticSegmentationEncoder(
            backbone=backbone, pretrained=pretrained, output_dim=512
        )
        
        # Position encoder (enhanced from Step 3)
        self.position_encoder = nn.Sequential(
            nn.Linear(4, 128),  # [current_x, current_y, goal_x, goal_y]
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Cross-modal fusion
        self.fusion_layers = nn.ModuleList([
            nn.Linear(512 + 128, 512),  # vision + position
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # Navigation head
        self.nav_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh()  # Constrain to [-1, 1]
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_image, current_pos, goal_pos, return_all=False):
        # Extract vision features
        vision_features, seg_logits = self.vision_encoder(
            rgb_image, current_pos, goal_pos, return_segmentation=True
        )
        
        # Extract position features
        pos_input = torch.cat([current_pos, goal_pos], dim=-1)
        pos_features = self.position_encoder(pos_input)
        
        # Fuse features
        x = torch.cat([vision_features, pos_features], dim=-1)
        
        for layer in self.fusion_layers:
            residual = x if x.shape[-1] == layer.out_features else None
            x = layer(x)
            if residual is not None:
                x = x + residual
            x = self.activation(x)
            x = self.dropout(x)
        
        # Predict direction and confidence
        direction = self.nav_head(x)
        confidence = self.confidence_head(x)
        
        if return_all:
            return direction, confidence, seg_logits
        else:
            return direction

# ============================================================================
# 5. Advanced Loss Functions
# ============================================================================

def semantic_segmentation_loss(pred_seg, target_seg):
    """Semantic segmentation loss"""
    return F.cross_entropy(pred_seg, target_seg.long())

def realworld_navigation_loss(predicted_directions, target_directions, current_positions, 
                            goal_positions, pred_seg, target_seg, confidences=None):
    """Advanced loss for real-world navigation"""
    
    # 1. Direction similarity loss
    pred_norm = F.normalize(predicted_directions, p=2, dim=1)
    target_norm = F.normalize(target_directions, p=2, dim=1)
    cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    direction_loss = (1 - cosine_sim).mean()
    
    # 2. Magnitude loss
    pred_magnitude = torch.norm(predicted_directions, p=2, dim=1)
    target_magnitude = torch.norm(target_directions, p=2, dim=1)
    magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)
    
    # 3. Semantic segmentation loss
    seg_loss = semantic_segmentation_loss(pred_seg, target_seg.squeeze(1))
    
    # 4. Goal alignment loss
    goal_direction = goal_positions - current_positions
    goal_direction_norm = F.normalize(goal_direction, p=2, dim=1)
    goal_alignment = F.cosine_similarity(pred_norm, goal_direction_norm, dim=1)
    goal_loss = (1 - goal_alignment).mean()
    
    # 5. Confidence loss (if provided)
    if confidences is not None:
        # High confidence for good predictions
        prediction_quality = cosine_sim.detach()
        confidence_loss = F.mse_loss(confidences.squeeze(), prediction_quality)
    else:
        confidence_loss = torch.tensor(0.0, device=direction_loss.device)
    
    # Weighted combination
    total_loss = (0.3 * direction_loss +
                  0.1 * magnitude_loss +
                  0.3 * seg_loss +
                  0.2 * goal_loss +
                  0.1 * confidence_loss)
    
    return total_loss, direction_loss, magnitude_loss, seg_loss, goal_loss, confidence_loss

# ============================================================================
# 6. Training Function
# ============================================================================

def train_realworld_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """Train real-world navigation model"""
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    print(f"Training real-world model on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = {
            'total': 0, 'direction': 0, 'magnitude': 0, 
            'segmentation': 0, 'goal': 0, 'confidence': 0, 'batches': 0
        }
        
        for batch in train_loader:
            rgb_image = batch['rgb_image'].to(device)
            semantic_mask = batch['semantic_mask'].to(device)
            current_pos = batch['current_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            target_direction = batch['direction'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_direction, confidence, seg_logits = model(
                rgb_image, current_pos, goal_pos, return_all=True
            )
            
            # Compute loss
            total_loss, dir_loss, mag_loss, seg_loss, goal_loss, conf_loss = realworld_navigation_loss(
                predicted_direction, target_direction, current_pos, goal_pos,
                seg_logits, semantic_mask, confidence
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            train_metrics['total'] += total_loss.item()
            train_metrics['direction'] += dir_loss.item()
            train_metrics['magnitude'] += mag_loss.item()
            train_metrics['segmentation'] += seg_loss.item()
            train_metrics['goal'] += goal_loss.item()
            train_metrics['confidence'] += conf_loss.item()
            train_metrics['batches'] += 1
        
        # Validation phase
        model.eval()
        val_metrics = {
            'total': 0, 'direction': 0, 'magnitude': 0, 
            'segmentation': 0, 'goal': 0, 'confidence': 0, 'batches': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                rgb_image = batch['rgb_image'].to(device)
                semantic_mask = batch['semantic_mask'].to(device)
                current_pos = batch['current_pos'].to(device)
                goal_pos = batch['goal_pos'].to(device)
                target_direction = batch['direction'].to(device)
                
                predicted_direction, confidence, seg_logits = model(
                    rgb_image, current_pos, goal_pos, return_all=True
                )
                
                total_loss, dir_loss, mag_loss, seg_loss, goal_loss, conf_loss = realworld_navigation_loss(
                    predicted_direction, target_direction, current_pos, goal_pos,
                    seg_logits, semantic_mask, confidence
                )
                
                val_metrics['total'] += total_loss.item()
                val_metrics['direction'] += dir_loss.item()
                val_metrics['magnitude'] += mag_loss.item()
                val_metrics['segmentation'] += seg_loss.item()
                val_metrics['goal'] += goal_loss.item()
                val_metrics['confidence'] += conf_loss.item()
                val_metrics['batches'] += 1
        
        # Calculate averages
        for key in ['total', 'direction', 'magnitude', 'segmentation', 'goal', 'confidence']:
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
                  f"Seg:{train_metrics['segmentation']:.4f}, "
                  f"Goal:{train_metrics['goal']:.4f}) | "
                  f"Val={val_metrics['total']:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return train_losses, val_losses

# ============================================================================
# 7. Evaluation and Visualization
# ============================================================================

def evaluate_realworld_model(model, test_loader, device='cpu'):
    """Evaluate real-world navigation model"""
    model.eval()
    model = model.to(device)
    
    metrics = {
        'direction_cosine_sim': 0,
        'magnitude_error': 0,
        'goal_alignment': 0,
        'confidence_accuracy': 0,
        'segmentation_accuracy': 0,
        'total_samples': 0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            rgb_image = batch['rgb_image'].to(device)
            semantic_mask = batch['semantic_mask'].to(device)
            current_pos = batch['current_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            target_direction = batch['direction'].to(device)
            
            predicted_direction, confidence, seg_logits = model(
                rgb_image, current_pos, goal_pos, return_all=True
            )
            
            batch_size = current_pos.shape[0]
            
            # Direction metrics
            pred_norm = F.normalize(predicted_direction, p=2, dim=1)
            target_norm = F.normalize(target_direction, p=2, dim=1)
            cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
            metrics['direction_cosine_sim'] += cosine_sim.sum().item()
            
            # Magnitude error
            pred_mag = torch.norm(predicted_direction, p=2, dim=1)
            target_mag = torch.norm(target_direction, p=2, dim=1)
            mag_error = F.mse_loss(pred_mag, target_mag, reduction='sum')
            metrics['magnitude_error'] += mag_error.item()
            
            # Goal alignment
            goal_direction = goal_pos - current_pos
            goal_direction_norm = F.normalize(goal_direction, p=2, dim=1)
            goal_alignment = F.cosine_similarity(pred_norm, goal_direction_norm, dim=1)
            metrics['goal_alignment'] += goal_alignment.sum().item()
            
            # Segmentation accuracy
            seg_pred = torch.argmax(seg_logits, dim=1)
            seg_target = semantic_mask.squeeze(1).long()
            seg_accuracy = (seg_pred == seg_target).float().mean()
            metrics['segmentation_accuracy'] += seg_accuracy.item() * batch_size
            
            # Confidence accuracy
            prediction_quality = cosine_sim.detach()
            conf_error = F.mse_loss(confidence.squeeze(), prediction_quality, reduction='sum')
            metrics['confidence_accuracy'] += conf_error.item()
            
            metrics['total_samples'] += batch_size
    
    # Calculate averages
    total_samples = metrics['total_samples']
    results = {
        'direction_similarity': metrics['direction_cosine_sim'] / total_samples,
        'magnitude_error': metrics['magnitude_error'] / total_samples,
        'goal_alignment': metrics['goal_alignment'] / total_samples,
        'segmentation_accuracy': metrics['segmentation_accuracy'] / total_samples,
        'confidence_error': metrics['confidence_accuracy'] / total_samples
    }
    
    print(f"\nReal-world Navigation Evaluation:")
    print(f"Direction Similarity: {results['direction_similarity']:.4f}")
    print(f"Goal Alignment: {results['goal_alignment']:.4f}")
    print(f"Segmentation Accuracy: {results['segmentation_accuracy']:.4f}")
    print(f"Magnitude Error: {results['magnitude_error']:.6f}")
    print(f"Confidence Error: {results['confidence_error']:.6f}")
    
    return results

def visualize_realworld_prediction(model, test_sample, device='cpu'):
    """Visualize real-world navigation prediction"""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        rgb_image = test_sample['rgb_image'].unsqueeze(0).to(device)
        semantic_mask = test_sample['semantic_mask']
        current_pos = test_sample['current_pos'].unsqueeze(0).to(device)
        goal_pos = test_sample['goal_pos'].unsqueeze(0).to(device)
        target_direction = test_sample['direction']
        scene_type = test_sample['scene_type']
        
        predicted_direction, confidence, seg_logits = model(
            rgb_image, current_pos, goal_pos, return_all=True
        )
        
        # Convert to numpy
        rgb_np = rgb_image.cpu().squeeze().permute(1, 2, 0).numpy()
        semantic_np = semantic_mask.squeeze().numpy()
        seg_pred_np = torch.argmax(seg_logits, dim=1).cpu().squeeze().numpy()
        predicted_direction_np = predicted_direction.cpu().squeeze().numpy()
        confidence_val = confidence.cpu().squeeze().item()
        
        # Convert positions back to image coordinates
        img_size = rgb_np.shape[:2]
        current_img_pos = current_pos.cpu().squeeze().numpy() * np.array(img_size)
        goal_img_pos = goal_pos.cpu().squeeze().numpy() * np.array(img_size)
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original RGB image
        axes[0, 0].imshow(rgb_np)
        axes[0, 0].plot(current_img_pos[1], current_img_pos[0], 'go', markersize=12, label='Start')
        axes[0, 0].plot(goal_img_pos[1], goal_img_pos[0], 'ro', markersize=12, label='Goal')
        axes[0, 0].set_title(f'RGB Image ({scene_type})')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # Ground truth semantic mask
        axes[0, 1].imshow(semantic_np, cmap='RdYlBu_r', alpha=0.8)
        axes[0, 1].plot(current_img_pos[1], current_img_pos[0], 'go', markersize=10)
        axes[0, 1].plot(goal_img_pos[1], goal_img_pos[0], 'ro', markersize=10)
        axes[0, 1].set_title('Ground Truth Segmentation')
        axes[0, 1].axis('off')
        
        # Predicted semantic mask
        axes[0, 2].imshow(seg_pred_np, cmap='RdYlBu_r', alpha=0.8)
        axes[0, 2].plot(current_img_pos[1], current_img_pos[0], 'go', markersize=10)
        axes[0, 2].plot(goal_img_pos[1], goal_img_pos[0], 'ro', markersize=10)
        axes[0, 2].set_title('Predicted Segmentation')
        axes[0, 2].axis('off')
        
        # Direction comparison
        axes[1, 0].imshow(rgb_np, alpha=0.7)
        scale = 30
        # Target direction (blue)
        axes[1, 0].arrow(current_img_pos[1], current_img_pos[0], 
                        target_direction[1]*scale, target_direction[0]*scale,
                        head_width=8, head_length=8, fc='blue', ec='blue', 
                        linewidth=4, alpha=0.8, label='Target')
        # Predicted direction (red)
        axes[1, 0].arrow(current_img_pos[1], current_img_pos[0], 
                        predicted_direction_np[1]*scale, predicted_direction_np[0]*scale,
                        head_width=8, head_length=8, fc='red', ec='red', 
                        linewidth=4, alpha=0.8, label='Predicted')
        axes[1, 0].set_title('Direction Comparison')
        axes[1, 0].legend()
        axes[1, 0].axis('off')
        
        # Metrics
        cosine_sim = np.dot(target_direction, predicted_direction_np) / (
            np.linalg.norm(target_direction) * np.linalg.norm(predicted_direction_np)
        )
        
        seg_accuracy = (semantic_np == seg_pred_np).mean()
        
        metrics_text = f"""
        Scene Type: {scene_type}
        
        Direction Metrics:
        Cosine Similarity: {cosine_sim:.3f}
        Confidence: {confidence_val:.3f}
        
        Target: [{target_direction[0]:.3f}, {target_direction[1]:.3f}]
        Predicted: [{predicted_direction_np[0]:.3f}, {predicted_direction_np[1]:.3f}]
        
        Segmentation:
        Accuracy: {seg_accuracy:.3f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Metrics')
        
        # Error analysis
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Model Confidence')
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# 8. Main Training Script
# ============================================================================

def main_realworld():
    """Main real-world navigation training script"""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("Generating real-world navigation dataset...")
    dataset = generate_realworld_dataset(num_samples=3000, image_size=(224, 224))
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create data loaders
    train_dataset = RealWorldNavigationDataset(train_data, transform=train_transform)
    val_dataset = RealWorldNavigationDataset(val_data, transform=val_transform)
    test_dataset = RealWorldNavigationDataset(test_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    model = RealWorldNavigator(backbone='resnet50', pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Real-world model parameters: {total_params:,}")
    
    # Train model
    print("\nStarting real-world navigation training...")
    train_losses, val_losses = train_realworld_model(
        model, train_loader, val_loader, 
        num_epochs=60, lr=0.0005, device=device
    )
    
    # Evaluate model
    print("\nEvaluating real-world model...")
    results = evaluate_realworld_model(model, test_loader, device=device)
    
    # Visualize predictions
    print("\nVisualizing real-world predictions...")
    for i in range(3):
        test_sample = test_dataset[i * 50]
        visualize_realworld_prediction(model, test_sample, device=device)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Real-world Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    metric_names = ['Dir Sim', 'Goal Align', 'Seg Acc']
    metric_values = [results['direction_similarity'], 
                    results['goal_alignment'], 
                    results['segmentation_accuracy']]
    colors = ['green', 'blue', 'orange']
    
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.5, f"""
    Step 4: Real-world Navigation
    
    Key Achievements:
    ✓ RGB Image Processing
    ✓ Semantic Segmentation  
    ✓ Multi-modal Fusion
    ✓ Real-world Scene Understanding
    
    Next: Deploy to Robot! 🤖
    """, fontsize=12, verticalalignment='center',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Achievement Summary')
    
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    model = main_realworld()