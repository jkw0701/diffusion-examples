import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
from typing import List, Tuple, Optional
import heapq
import os
import argparse
import time
from pathlib import Path

# HuggingFace Diffusers imports
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1. Enhanced Scene Generation with Better Obstacle Patterns
# ============================================================================

class AdvancedObstacleGenerator:
    """다양한 복잡한 장애물 패턴을 생성하는 클래스"""
    
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        
    def generate_maze_pattern(self):
        """미로 형태의 장애물 패턴 생성"""
        obstacle_map = np.zeros(self.image_size, dtype=np.uint8)
        
        wall_thickness = 3
        corridor_width = 8
        
        # 수직 벽 생성
        for x in range(corridor_width, self.image_size[0], corridor_width + wall_thickness):
            obstacle_map[:, x:x+wall_thickness] = 1
            
        # 수평 벽 생성 (구멍 포함)
        for y in range(corridor_width, self.image_size[1], corridor_width + wall_thickness):
            wall_row = obstacle_map[y:y+wall_thickness, :]
            wall_row[:] = 1
            # 벽에 구멍 생성
            gap_positions = np.random.choice(self.image_size[0]//2, 2, replace=False) * 2
            for gap in gap_positions:
                gap_start = max(0, gap - 2)
                gap_end = min(self.image_size[0], gap + 3)
                wall_row[:, gap_start:gap_end] = 0
        
        return obstacle_map
    
    def generate_scattered_obstacles(self):
        """산재된 원형 및 사각형 장애물 생성"""
        obstacle_map = np.zeros(self.image_size, dtype=np.uint8)
        
        num_obstacles = np.random.randint(8, 15)
        
        for _ in range(num_obstacles):
            if np.random.random() < 0.6:  # 원형 장애물
                center_x = np.random.randint(5, self.image_size[0] - 5)
                center_y = np.random.randint(5, self.image_size[1] - 5)
                radius = np.random.randint(3, 8)
                
                y, x = np.ogrid[:self.image_size[0], :self.image_size[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                obstacle_map[mask] = 1
                
            else:  # 사각형 장애물
                x = np.random.randint(2, self.image_size[0] - 8)
                y = np.random.randint(2, self.image_size[1] - 8)
                w = np.random.randint(3, 6)
                h = np.random.randint(3, 6)
                obstacle_map[y:y+h, x:x+w] = 1
        
        return obstacle_map
    
    def generate_corridor_pattern(self):
        """좁은 복도 패턴 생성"""
        obstacle_map = np.ones(self.image_size, dtype=np.uint8)
        
        # 메인 복도 생성
        corridor_width = np.random.randint(4, 8)
        corridor_y = self.image_size[1] // 2
        corridor_start = corridor_y - corridor_width // 2
        corridor_end = corridor_y + corridor_width // 2
        
        obstacle_map[corridor_start:corridor_end, :] = 0
        
        # 분기 복도 생성
        num_branches = np.random.randint(2, 4)
        for _ in range(num_branches):
            branch_x = np.random.randint(10, self.image_size[0] - 10)
            branch_length = np.random.randint(8, 15)
            branch_width = np.random.randint(3, 5)
            
            if np.random.random() < 0.5:  # 수직 분기
                branch_start = max(0, corridor_y - branch_length // 2)
                branch_end = min(self.image_size[1], corridor_y + branch_length // 2)
                obstacle_map[branch_start:branch_end, 
                           branch_x:branch_x+branch_width] = 0
            else:  # 수평 확장
                obstacle_map[corridor_start:corridor_end, 
                           branch_x:branch_x+branch_length] = 0
        
        return obstacle_map

def generate_challenging_navigation_scene(pattern_type=None):
    """도전적인 네비게이션 장면 생성"""
    generator = AdvancedObstacleGenerator()
    
    if pattern_type is None:
        pattern_type = np.random.choice(['maze', 'scattered', 'corridor'])
    
    if pattern_type == 'maze':
        obstacle_map = generator.generate_maze_pattern()
    elif pattern_type == 'scattered':
        obstacle_map = generator.generate_scattered_obstacles()
    else:
        obstacle_map = generator.generate_corridor_pattern()
    
    # RGB 시각화 생성
    rgb_image = np.ones((64, 64, 3), dtype=np.uint8) * 255  # 흰색 배경
    rgb_image[obstacle_map == 1] = [50, 50, 50]  # 어두운 장애물
    
    return rgb_image, obstacle_map, pattern_type

# ============================================================================
# 2. A* Pathfinder for Ground Truth Generation
# ============================================================================

class AStarPathfinder:
    """최적 경로 생성을 위한 A* 경로 찾기 알고리즘"""
    
    def __init__(self, obstacle_map):
        self.obstacle_map = obstacle_map
        self.height, self.width = obstacle_map.shape
        
    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos):
        neighbors = []
        directions = [(-1,-1, 1.414), (-1,0, 1.0), (-1,1, 1.414), 
                     (0,-1, 1.0), (0,1, 1.0), 
                     (1,-1, 1.414), (1,0, 1.0), (1,1, 1.414)]
        
        for dx, dy, cost in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            if (0 <= new_x < self.height and 0 <= new_y < self.width and 
                self.obstacle_map[new_x, new_y] == 0):
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def find_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
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
        
        return []

# ============================================================================
# 3. Collision Detection Functions (FIXED)
# ============================================================================

def sample_obstacle_map_soft_gradients(point, obstacle_map, map_size):
    """그래디언트를 보존하는 장애물 맵 샘플링 (수정됨)"""
    # 정규화된 좌표를 맵 좌표로 변환
    x_continuous = point[0] * (map_size - 1)
    y_continuous = point[1] * (map_size - 1)
    
    # 유효 범위로 클램프 (미분 가능)
    x_continuous = torch.clamp(x_continuous, 0, map_size - 1)
    y_continuous = torch.clamp(y_continuous, 0, map_size - 1)
    
    # 그리드 샘플링 형식으로 변환 [-1, 1]
    grid_x = (x_continuous / (map_size - 1)) * 2 - 1
    grid_y = (y_continuous / (map_size - 1)) * 2 - 1
    
    # 단일 포인트를 위한 샘플링 그리드 [1, 1, 1, 2] 생성
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    
    # 장애물 맵에 배치 및 채널 차원 추가 [1, 1, H, W]
    obstacle_map_batch = obstacle_map.unsqueeze(0).unsqueeze(0)
    
    # 미분 가능한 샘플링을 위해 F.grid_sample 사용
    try:
        sampled_value = F.grid_sample(
            obstacle_map_batch, grid, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        return sampled_value.squeeze()
    except Exception:
        # 폴백: 간단한 이중선형 보간
        return sample_obstacle_map_bilinear_fallback(point, obstacle_map, map_size)

def sample_obstacle_map_bilinear_fallback(point, obstacle_map, map_size):
    """그리드 샘플링 실패 시 폴백 함수"""
    x_continuous = point[0] * (map_size - 1)
    y_continuous = point[1] * (map_size - 1)
    
    x_continuous = torch.clamp(x_continuous, 0, map_size - 1)
    y_continuous = torch.clamp(y_continuous, 0, map_size - 1)
    
    x_floor = torch.floor(x_continuous).long()
    y_floor = torch.floor(y_continuous).long()
    x_ceil = torch.clamp(x_floor + 1, 0, map_size - 1)
    y_ceil = torch.clamp(y_floor + 1, 0, map_size - 1)
    
    x_frac = x_continuous - x_floor.float()
    y_frac = y_continuous - y_floor.float()
    
    try:
        top_left = obstacle_map[y_floor, x_floor]
        top_right = obstacle_map[y_floor, x_ceil]
        bottom_left = obstacle_map[y_ceil, x_floor]
        bottom_right = obstacle_map[y_ceil, x_ceil]
        
        top = top_left * (1 - x_frac) + top_right * x_frac
        bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac
        collision_value = top * (1 - y_frac) + bottom * y_frac
        
        return collision_value
    except:
        return torch.tensor(0.0, device=point.device, requires_grad=True)

def compute_noise_based_collision_penalty(noise_pred, noisy_paths, timesteps, scheduler, obstacle_maps, device):
    """노이즈 예측에서 직접 충돌 페널티 계산 (수정됨)"""
    batch_size = noise_pred.shape[0]
    map_size = obstacle_maps.shape[-1]
    
    # 타임스텝 처리 수정
    if isinstance(timesteps, torch.Tensor):
        if len(timesteps.shape) == 0:  # 스칼라 텐서
            timesteps = timesteps.unsqueeze(0).expand(batch_size)
        elif len(timesteps) == 1:  # 전체 배치에 대한 단일 타임스텝
            timesteps = timesteps.expand(batch_size)
    else:
        timesteps = torch.tensor([timesteps] * batch_size, device=device)
    
    # 각 배치 샘플에 대한 노이즈 스케줄 값 가져오기
    alpha_t = scheduler.alphas_cumprod[timesteps].to(device)
    sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1)
    
    # DDPM 디노이징 공식: x_0 ≈ (x_t - sqrt(1-α_t) * ε_θ) / sqrt(α_t)
    predicted_original = (noisy_paths - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    
    total_penalty = torch.tensor(0.0, device=device, requires_grad=True)
    
    for b in range(batch_size):
        path = predicted_original[b]
        obstacle_map = obstacle_maps[b].float()
        
        batch_penalty = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 포인트 충돌 (그래디언트 포함!)
        for i in range(path.shape[0]):
            point_collision = sample_obstacle_map_soft_gradients(path[i], obstacle_map, map_size)
            batch_penalty = batch_penalty + point_collision
        
        # 선분 충돌 (그래디언트 포함!)
        for i in range(path.shape[0] - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # 각 선분을 따라 3개 포인트 샘플링
            for j in range(1, 4):
                t = j / 4
                interpolated_point = start_point * (1 - t) + end_point * t
                line_collision = sample_obstacle_map_soft_gradients(interpolated_point, obstacle_map, map_size)
                batch_penalty = batch_penalty + 0.7 * line_collision
        
        total_penalty = total_penalty + batch_penalty / (path.shape[0] + 3 * (path.shape[0] - 1) * 0.7)
    
    return total_penalty / batch_size

# ============================================================================
# 4. Diffusion Path Planning Model
# ============================================================================

class PathDiffusionModel(nn.Module):
    """HuggingFace Diffusers를 사용한 경로 계획용 확산 모델"""
    
    def __init__(self, max_path_length=16, path_dim=2):
        super(PathDiffusionModel, self).__init__()
        
        self.max_path_length = max_path_length
        self.path_dim = path_dim
        
        # 환경 인코더 (장애물 맵 + 시작/목표 처리)
        self.env_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.AdaptiveAvgPool2d(8),  # -> 8x8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 시작/목표 위치 인코더
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [start_x, start_y, goal_x, goal_y]
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 확산을 위한 시간 임베딩
        self.time_embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # UNet을 위한 경로 투영
        self.path_projection = nn.Linear(max_path_length * path_dim, 16 * 16)
        self.path_unprojection = nn.Linear(16 * 16, max_path_length * path_dim)
        
        # UNet 초기화 (오류 처리 포함)
        try:
            self.unet = UNet2DConditionModel(
                sample_size=16,
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(64, 128, 256),
                down_block_types=(
                    "DownBlock2D",
                    "CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D"
                ),
                up_block_types=(
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D"
                ),
                cross_attention_dim=256 + 64,
                attention_head_dim=32
            )
            self.use_conditioning = True
            print("조건부 UNet2DConditionModel 사용")
        except Exception as e:
            print(f"조건부 UNet 실패: {e}")
            try:
                self.unet = UNet2DConditionModel(
                    sample_size=16,
                    in_channels=1,
                    out_channels=1,
                    block_out_channels=(64, 128, 256),
                    layers_per_block=2,
                    cross_attention_dim=320
                )
                self.use_conditioning = True
                print("단순 UNet2DConditionModel 사용")
            except Exception as e2:
                print(f"단순 UNet도 실패: {e2}")
                from diffusers import UNet2DModel
                self.unet = UNet2DModel(
                    sample_size=16,
                    in_channels=1,
                    out_channels=1,
                    block_out_channels=(64, 128, 256),
                    layers_per_block=2
                )
                self.use_conditioning = False
                print("기본 UNet2DModel 사용 (조건부 없음)")
        
        # 노이즈 스케줄러
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
    
    def get_time_embedding(self, timesteps):
        """정현파 시간 임베딩 생성"""
        half_dim = 128
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if emb.shape[1] < 256:
            emb = F.pad(emb, (0, 256 - emb.shape[1]))
        elif emb.shape[1] > 256:
            emb = emb[:, :256]
            
        return self.time_embedding(emb)
    
    def encode_conditions(self, obstacle_map, start_pos, goal_pos):
        """환경 조건 인코딩"""
        env_features = self.env_encoder(obstacle_map.unsqueeze(1))
        pos_input = torch.cat([start_pos, goal_pos], dim=-1)
        pos_features = self.pos_encoder(pos_input)
        condition_embedding = torch.cat([env_features, pos_features], dim=-1)
        return condition_embedding
    
    def forward(self, noisy_paths, timesteps, obstacle_map, start_pos, goal_pos):
        """훈련용 순전파"""
        batch_size = noisy_paths.shape[0]
        
        # 시간 임베딩 가져오기
        time_emb = self.get_time_embedding(timesteps)
        
        # 조건 인코딩
        condition_emb = self.encode_conditions(obstacle_map, start_pos, goal_pos)
        
        # UNet을 위해 경로를 이미지 형식으로 변환
        path_flat = noisy_paths.view(batch_size, -1)
        path_img = self.path_projection(path_flat).view(batch_size, 1, 16, 16)
        
        # 적절한 조건부 처리로 UNet 디노이징
        if self.use_conditioning:
            try:
                noise_pred = self.unet(
                    sample=path_img,
                    timestep=timesteps,
                    encoder_hidden_states=condition_emb.unsqueeze(1),
                    return_dict=False
                )[0]
            except Exception:
                try:
                    noise_pred = self.unet(
                        path_img,
                        timesteps,
                        condition_emb.unsqueeze(1),
                        return_dict=False
                    )[0]
                except Exception:
                    noise_pred = self.unet(path_img, timesteps, return_dict=False)[0]
        else:
            noise_pred = self.unet(path_img, timesteps, return_dict=False)[0]
        
        # 경로 형식으로 다시 변환
        noise_pred_flat = noise_pred.view(batch_size, -1)
        noise_pred_path = self.path_unprojection(noise_pred_flat)
        noise_pred_path = noise_pred_path.view(batch_size, self.max_path_length, self.path_dim)
        
        return noise_pred_path
    
    @torch.no_grad()
    def generate_path(self, obstacle_map, start_pos, goal_pos, num_inference_steps=50):
        """확산 샘플링을 사용한 경로 생성"""
        batch_size = obstacle_map.shape[0]
        device = obstacle_map.device
        
        # 랜덤 노이즈로 시작
        path_shape = (batch_size, self.max_path_length, self.path_dim)
        path = torch.randn(path_shape, device=device)
        
        # 스케줄러 설정
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 조건 인코딩 (조건부 사용 시)
        if self.use_conditioning:
            condition_emb = self.encode_conditions(obstacle_map, start_pos, goal_pos)
        
        # 디노이징 루프
        for timestep in self.scheduler.timesteps:
            timesteps = timestep.expand(batch_size).to(device)
            
            # UNet을 위해 변환
            path_flat = path.view(batch_size, -1)
            path_img = self.path_projection(path_flat).view(batch_size, 1, 16, 16)
            
            # 적절한 오류 처리로 UNet 예측
            if self.use_conditioning:
                try:
                    noise_pred = self.unet(
                        sample=path_img,
                        timestep=timesteps,
                        encoder_hidden_states=condition_emb.unsqueeze(1),
                        return_dict=False
                    )[0]
                except Exception:
                    try:
                        noise_pred = self.unet(
                            path_img,
                            timesteps,
                            condition_emb.unsqueeze(1),
                            return_dict=False
                        )[0]
                    except Exception:
                        noise_pred = self.unet(path_img, timesteps, return_dict=False)[0]
            else:
                noise_pred = self.unet(path_img, timesteps, return_dict=False)[0]
            
            # 다시 변환
            noise_pred_flat = noise_pred.view(batch_size, -1)
            noise_pred_path = self.path_unprojection(noise_pred_flat)
            noise_pred_path = noise_pred_path.view(batch_size, self.max_path_length, self.path_dim)
            
            # 스케줄러 단계
            path = self.scheduler.step(noise_pred_path, timestep, path, return_dict=False)[0]
        
        # 경로가 시작 위치에서 시작하고 목표 위치에서 끝나도록 보장
        # path[:, 0] = start_pos
        # path[:, -1] = goal_pos
        
        # 중간 점을 보간하여 경로 부드럽게 하기
        for i in range(1, self.max_path_length - 1):
            alpha = i / (self.max_path_length - 1)
            linear_interp = start_pos * (1 - alpha) + goal_pos * alpha
            path[:, i] = 0.9 * path[:, i] + 0.1 * linear_interp
        
        return path

# ============================================================================
# 5. Dataset Generation and Management
# ============================================================================

def generate_diffusion_dataset(num_samples=2000, image_size=64, max_path_length=16):
    """확산 경로 계획을 위한 데이터셋 생성"""
    dataset = []
    
    print(f"확산 경로 계획 샘플 {num_samples}개 생성 중...")
    
    successful_samples = 0
    attempts = 0
    max_attempts = num_samples * 3
    
    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1
        
        if attempts % 500 == 0:
            print(f"시도: {attempts}, 성공: {successful_samples}/{num_samples}")
        
        # 도전적인 장애물 패턴 생성
        try:
            rgb_image, obstacle_map, pattern_type = generate_challenging_navigation_scene()
        except:
            continue
        
        # 자유 위치 찾기
        free_positions = np.argwhere(obstacle_map == 0)
        
        if len(free_positions) < 20:
            continue
        
        # 충분한 거리로 시작과 목표 선택
        max_pos_attempts = 20
        valid_pair = False
        
        for _ in range(max_pos_attempts):
            indices = np.random.choice(len(free_positions), 2, replace=False)
            start_pos = free_positions[indices[0]]
            goal_pos = free_positions[indices[1]]
            
            distance = np.linalg.norm(goal_pos - start_pos)
            if distance > image_size * 0.3:
                valid_pair = True
                break
        
        if not valid_pair:
            continue
        
        # A*를 사용하여 최적 경로 생성
        try:
            pathfinder = AStarPathfinder(obstacle_map)
            optimal_path = pathfinder.find_path(start_pos, goal_pos)
        except:
            continue
        
        if len(optimal_path) < 3:
            continue
        
        # 고정 길이로 경로 포인트 샘플링
        if len(optimal_path) > max_path_length:
            indices = np.linspace(0, len(optimal_path) - 1, max_path_length, dtype=int)
            sampled_path = [optimal_path[i] for i in indices]
        else:
            sampled_path = []
            for i in range(max_path_length):
                t = i / (max_path_length - 1)
                idx = min(int(t * (len(optimal_path) - 1)), len(optimal_path) - 1)
                sampled_path.append(optimal_path[idx])
        
        # numpy로 변환하고 정규화
        path_array = np.array(sampled_path, dtype=np.float32)
        norm_start = start_pos.astype(np.float32) / image_size
        norm_goal = goal_pos.astype(np.float32) / image_size
        norm_path = path_array / image_size
        
        # 텐서 데이터 생성
        obstacle_tensor = torch.FloatTensor(obstacle_map.astype(np.float32))
        rgb_tensor = torch.FloatTensor(rgb_image).permute(2, 0, 1) / 255.0
        path_tensor = torch.FloatTensor(norm_path)
        start_tensor = torch.FloatTensor(norm_start)
        goal_tensor = torch.FloatTensor(norm_goal)
        
        dataset.append({
            'obstacle_map': obstacle_tensor,
            'rgb_image': rgb_tensor,
            'start_pos': start_tensor,
            'goal_pos': goal_tensor,
            'optimal_path': path_tensor,
            'pattern_type': pattern_type
        })
        
        successful_samples += 1
    
    print(f"확산 데이터셋 생성 완료: {len(dataset)} 샘플")
    return dataset

class DiffusionPathDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# 6. Training Loop (FIXED)
# ============================================================================

class SimpleEMA:
    """간단한 EMA 구현"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data = self.shadow[name]

def train_diffusion_model(model, train_loader, val_loader, val_data, num_epochs=100, lr=1e-4, device='cpu', save_dir='./models'):
    """확산 경로 계획 모델 훈련 (수정됨)"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = SimpleEMA(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    
    print(f"디바이스에서 확산 모델 훈련: {device}")
    print(f"모델은 25 에포크마다 저장됩니다: {save_dir}")
    
    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        train_loss_sum = 0
        collision_loss_sum = 0
        
        for batch_idx, batch in enumerate(train_loader):
            obstacle_map = batch['obstacle_map'].to(device)
            start_pos = batch['start_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            target_path = batch['optimal_path'].to(device)
            
            batch_size = obstacle_map.shape[0]
            
            # 랜덤 타임스텝 샘플링
            timesteps = torch.randint(
                0, model.scheduler.num_train_timesteps, 
                (batch_size,), device=device
            ).long()
            
            # 타겟 경로에 노이즈 추가
            noise = torch.randn_like(target_path)
            noisy_paths = model.scheduler.add_noise(target_path, noise, timesteps)
            
            # 노이즈 예측
            optimizer.zero_grad()
            noise_pred = model(noisy_paths, timesteps, obstacle_map, start_pos, goal_pos)
            
            # 기본 디노이징 손실
            denoising_loss = F.mse_loss(noise_pred, noise)
            
            # 강건한 충돌 감지
            try:
                collision_penalty = compute_noise_based_collision_penalty(
                    noise_pred, noisy_paths, timesteps, model.scheduler, obstacle_map, device
                )
                
                if batch_idx % 20 == 0 and epoch % 5 == 0:
                    print(f"  에포크 {epoch}, 배치 {batch_idx}")
                    print(f"    디노이징: {denoising_loss.item():.6f}")
                    print(f"    충돌: {collision_penalty.item():.6f}")
                    
            except Exception as e:
                print(f"❌ 충돌 페널티 실패: {e}")
                collision_penalty = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 결합 손실
            total_loss = denoising_loss + 1.0 * collision_penalty
            
            if not total_loss.requires_grad:
                print("⚠️ 경고: 총 손실이 그래디언트를 요구하지 않습니다!")
                total_loss = denoising_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.step()
            
            train_loss_sum += denoising_loss.item()
            collision_loss_sum += collision_penalty.item()
        
        scheduler.step()
        
        # 검증 단계
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                obstacle_map = batch['obstacle_map'].to(device)
                start_pos = batch['start_pos'].to(device)
                goal_pos = batch['goal_pos'].to(device)
                target_path = batch['optimal_path'].to(device)
                
                batch_size = obstacle_map.shape[0]
                timesteps = torch.randint(
                    0, model.scheduler.num_train_timesteps,
                    (batch_size,), device=device
                ).long()
                
                noise = torch.randn_like(target_path)
                noisy_paths = model.scheduler.add_noise(target_path, noise, timesteps)
                noise_pred = model(noisy_paths, timesteps, obstacle_map, start_pos, goal_pos)
                
                val_loss = F.mse_loss(noise_pred, noise)
                val_loss_sum += val_loss.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_collision = collision_loss_sum / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 진행 상황 출력
        if epoch % 10 == 0:
            print(f"에포크 {epoch:3d}: "
                  f"훈련: {avg_train_loss:.4f}, "
                  f"검증: {avg_val_loss:.4f}, "
                  f"충돌: {avg_collision:.4f}")
        
        # 25 에포크마다 체크포인트 저장
        if (epoch + 1) % 25 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'collision_loss': avg_collision
            }, checkpoint_path)
            print(f"✅ 체크포인트 저장됨: {checkpoint_path}")
    
    # EMA 가중치 적용
    ema.apply_shadow()
    
    # 최종 모델 저장
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"✅ 최종 모델 저장됨: {final_path}")
    
    return train_losses, val_losses

# ============================================================================
# 7. Evaluation Functions
# ============================================================================

def calculate_path_smoothness(path):
    """경로 부드러움 계산 (낮을수록 부드러움)"""
    if len(path) < 3:
        return 0
    
    curvatures = []
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0

def evaluate_path_quality(generated_path, target_path, obstacle_map):
    """경로 품질 평가"""
    img_size = 64
    
    # 이미지 좌표로 변환
    generated_path_img = generated_path * img_size
    target_path_img = target_path * img_size
    
    metrics = {}
    
    # 1. 충돌률 계산
    collisions = 0
    for point in generated_path_img:
        x, y = int(np.clip(point[1], 0, 63)), int(np.clip(point[0], 0, 63))
        if obstacle_map[y, x] > 0.5:
            collisions += 1
    metrics['collision_rate'] = collisions / len(generated_path_img)
    
    # 2. 경로 길이 비율
    path_length_optimal = np.sum(np.linalg.norm(np.diff(target_path_img, axis=0), axis=1))
    path_length_generated = np.sum(np.linalg.norm(np.diff(generated_path_img, axis=0), axis=1))
    metrics['length_ratio'] = path_length_generated / max(path_length_optimal, 1e-6)
    
    # 3. 부드러움 비율
    smoothness_optimal = calculate_path_smoothness(target_path_img)
    smoothness_generated = calculate_path_smoothness(generated_path_img)
    metrics['smoothness_ratio'] = smoothness_generated / max(smoothness_optimal, 1e-6)
    
    # 4. 목표 오차
    goal_error = np.linalg.norm(generated_path_img[-1] - target_path_img[-1])
    metrics['goal_error'] = goal_error
    
    return metrics

def load_model_checkpoint(model_path, device='cpu'):
    """모델 체크포인트 로드"""
    print(f"모델 로딩: {model_path}")
    
    # 모델 생성
    model = PathDiffusionModel(max_path_length=16)
    
    # 체크포인트 로드
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # 체크포인트인지 모델 상태만인지 확인
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'Unknown')
            train_loss = checkpoint.get('train_loss', 'Unknown')
            print(f"체크포인트 로드됨: 에포크 {epoch}, 훈련 손실: {train_loss}")
        else:
            # 모델 상태만 있는 경우
            model.load_state_dict(checkpoint)
            print("모델 상태 로드됨")
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    model.to(device)
    model.eval()
    
    return model

def evaluate_model_on_dataset(model, dataset, device='cpu', num_samples=50):
    """데이터셋에서 모델 평가"""
    model.eval()
    
    all_metrics = {
        'collision_rate': [],
        'length_ratio': [],
        'smoothness_ratio': [],
        'goal_error': []
    }
    
    pattern_metrics = {
        'maze': {'collision_rate': [], 'length_ratio': [], 'smoothness_ratio': [], 'goal_error': []},
        'scattered': {'collision_rate': [], 'length_ratio': [], 'smoothness_ratio': [], 'goal_error': []},
        'corridor': {'collision_rate': [], 'length_ratio': [], 'smoothness_ratio': [], 'goal_error': []}
    }
    
    print(f"모델 평가 중... {num_samples} 샘플")
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            
            try:
                # 경로 생성
                obstacle_map = sample['obstacle_map'].unsqueeze(0).to(device)
                start_pos = sample['start_pos'].unsqueeze(0).to(device)
                goal_pos = sample['goal_pos'].unsqueeze(0).to(device)
                target_path = sample['optimal_path'].numpy()
                pattern_type = sample['pattern_type']
                
                generated_path = model.generate_path(obstacle_map, start_pos, goal_pos, num_inference_steps=200)
                generated_path = generated_path.squeeze(0).cpu().numpy()
                
                # 품질 평가
                obstacle_np = obstacle_map.squeeze().cpu().numpy()
                metrics = evaluate_path_quality(generated_path, target_path, obstacle_np)
                
                # 결과 저장
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                    pattern_metrics[pattern_type][key].append(value)
                
                if (i + 1) % 10 == 0:
                    print(f"  처리됨: {i + 1}/{num_samples}")
                    
            except Exception as e:
                print(f"샘플 {i} 평가 실패: {e}")
                continue
    
    # 평균 계산
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        else:
            avg_metrics[key] = float('inf')
            avg_metrics[f'{key}_std'] = 0
    
    # 패턴별 평균 계산
    pattern_avg_metrics = {}
    for pattern, metrics in pattern_metrics.items():
        pattern_avg_metrics[pattern] = {}
        for key, values in metrics.items():
            if values:
                pattern_avg_metrics[pattern][key] = np.mean(values)
                pattern_avg_metrics[pattern][f'{key}_std'] = np.std(values)
            else:
                pattern_avg_metrics[pattern][key] = float('inf')
                pattern_avg_metrics[pattern][f'{key}_std'] = 0
    
    return avg_metrics, pattern_avg_metrics, all_metrics

def visualize_evaluation_results(model, dataset, device='cpu', num_examples=6, save_path=None):
    """평가 결과 시각화"""
    model.eval()
    
    # 그리드 설정
    cols = 3
    rows = num_examples // cols + (num_examples % cols > 0)
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 6, rows * 8))
    
    if rows == 1:
        axes = axes.reshape(2, cols)
    
    examples_shown = 0
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            if examples_shown >= num_examples:
                break
                
            sample = dataset[i * (len(dataset) // num_examples)]
            
            try:
                # 경로 생성
                obstacle_map = sample['obstacle_map'].unsqueeze(0).to(device)
                start_pos = sample['start_pos'].unsqueeze(0).to(device)
                goal_pos = sample['goal_pos'].unsqueeze(0).to(device)
                target_path = sample['optimal_path']
                pattern_type = sample['pattern_type']
                
                generated_path = model.generate_path(obstacle_map, start_pos, goal_pos, num_inference_steps=200)
                generated_path = generated_path.squeeze(0).cpu().numpy()
                
                # 이미지 좌표로 변환
                img_size = 64
                obstacle_np = obstacle_map.squeeze().cpu().numpy()
                start_img = start_pos.squeeze().cpu().numpy() * img_size
                goal_img = goal_pos.squeeze().cpu().numpy() * img_size
                target_path_img = target_path.numpy() * img_size
                generated_path_img = generated_path * img_size
                
                # 품질 메트릭 계산
                metrics = evaluate_path_quality(generated_path, target_path.numpy(), obstacle_np)
                
                # 플롯 위치 계산
                row_idx = (examples_shown // cols) * 2
                col_idx = examples_shown % cols
                
                # 첫 번째 행: 경로 비교
                ax1 = axes[row_idx, col_idx]
                ax1.imshow(1 - obstacle_np, cmap='gray', alpha=0.8)
                ax1.plot(target_path_img[:, 1], target_path_img[:, 0], 'b-', 
                        linewidth=3, alpha=0.7, label='A* 최적')
                ax1.plot(generated_path_img[:, 1], generated_path_img[:, 0], 'r--', 
                        linewidth=3, alpha=0.7, label='확산 생성')
                ax1.plot(start_img[1], start_img[0], 'go', markersize=10, label='시작')
                ax1.plot(goal_img[1], goal_img[0], 'ro', markersize=10, label='목표')
                ax1.set_title(f'{pattern_type}\n충돌률: {metrics["collision_rate"]:.3f}')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 64)
                ax1.set_ylim(0, 64)
                
                # 두 번째 행: 메트릭 정보
                ax2 = axes[row_idx + 1, col_idx]
                metric_text = f"""
품질 메트릭:

충돌률: {metrics['collision_rate']:.3f}
길이 비율: {metrics['length_ratio']:.2f}
부드러움 비율: {metrics['smoothness_ratio']:.2f}
목표 오차: {metrics['goal_error']:.2f}

패턴: {pattern_type}
"""
                ax2.text(0.1, 0.5, metric_text, fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="lightcyan" if metrics['collision_rate'] < 0.1 else "lightcoral"))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
                
                examples_shown += 1
                
            except Exception as e:
                print(f"시각화 실패 (샘플 {i}): {e}")
                continue
    
    # 빈 축 숨기기
    for i in range(examples_shown, rows * cols):
        row_idx = (i // cols) * 2
        col_idx = i % cols
        axes[row_idx, col_idx].axis('off')
        axes[row_idx + 1, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 저장됨: {save_path}")
    
    plt.show()

def print_evaluation_report(avg_metrics, pattern_metrics):
    """평가 보고서 출력"""
    print("\n" + "="*60)
    print("📊 모델 평가 보고서")
    print("="*60)
    
    # 전체 메트릭
    print("\n🎯 전체 성능 메트릭:")
    print(f"  충돌률:     {avg_metrics['collision_rate']:.3f} ± {avg_metrics['collision_rate_std']:.3f}")
    print(f"  길이 비율:  {avg_metrics['length_ratio']:.2f} ± {avg_metrics['length_ratio_std']:.2f}")
    print(f"  부드러움:   {avg_metrics['smoothness_ratio']:.2f} ± {avg_metrics['smoothness_ratio_std']:.2f}")
    print(f"  목표 오차:  {avg_metrics['goal_error']:.2f} ± {avg_metrics['goal_error_std']:.2f}")
    
    # 성능 등급
    collision_grade = "우수" if avg_metrics['collision_rate'] < 0.05 else "양호" if avg_metrics['collision_rate'] < 0.15 else "개선 필요"
    length_grade = "우수" if 0.9 <= avg_metrics['length_ratio'] <= 1.3 else "양호" if 0.8 <= avg_metrics['length_ratio'] <= 1.5 else "개선 필요"
    
    print(f"\n📈 성능 등급:")
    print(f"  충돌 회피:  {collision_grade}")
    print(f"  경로 효율:  {length_grade}")
    
    # 패턴별 성능
    print(f"\n🏗️ 패턴별 성능:")
    for pattern, metrics in pattern_metrics.items():
        print(f"\n  {pattern.upper()} 패턴:")
        print(f"    충돌률:   {metrics['collision_rate']:.3f}")
        print(f"    길이 비율: {metrics['length_ratio']:.2f}")
        print(f"    부드러움:  {metrics['smoothness_ratio']:.2f}")
        print(f"    목표 오차: {metrics['goal_error']:.2f}")
    
    print("\n" + "="*60)

# ============================================================================
# 8. Main Functions
# ============================================================================

def main_training():
    """메인 훈련 함수"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터셋 생성
    print("확산 경로 계획 데이터셋 생성 중...")
    dataset = generate_diffusion_dataset(num_samples=1500, max_path_length=16)
    
    if len(dataset) < 100:
        print(f"❌ {len(dataset)}개 샘플만 생성됨. 더 많은 샘플이 필요합니다.")
        return None
    
    # 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]
    
    print(f"데이터셋 분할: 훈련={len(train_data)}, 검증={len(val_data)}")
    
    # 데이터 로더 생성
    train_dataset = DiffusionPathDataset(train_data)
    val_dataset = DiffusionPathDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 모델 생성
    model = PathDiffusionModel(max_path_length=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"확산 모델 파라미터: {total_params:,}")
    
    # 모델 훈련
    print("\n확산 경로 계획 훈련 시작...")
    train_losses, val_losses = train_diffusion_model(
        model, train_loader, val_loader, val_data,
        num_epochs=100, lr=1e-4, device=device
    )
    
    print("\n🎉 훈련 완료!")
    return model

def main_evaluation(model_path, num_samples=100, num_visualize=6):
    """메인 평가 함수"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    try:
        model = load_model_checkpoint(model_path, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 평가용 데이터셋 생성
    print("\n평가용 데이터셋 생성 중...")
    eval_dataset = generate_diffusion_dataset(num_samples=200, max_path_length=16)
    
    if len(eval_dataset) < 50:
        print(f"❌ 평가용 데이터셋이 부족합니다: {len(eval_dataset)} 샘플")
        return
    
    print(f"평가용 데이터셋: {len(eval_dataset)} 샘플")
    
    # 모델 평가
    print("\n📊 모델 성능 평가 중...")
    avg_metrics, pattern_metrics, all_metrics = evaluate_model_on_dataset(
        model, eval_dataset, device, num_samples=num_samples
    )
    
    # 평가 보고서 출력
    print_evaluation_report(avg_metrics, pattern_metrics)
    
    # 시각화
    print(f"\n🎨 {num_visualize}개 샘플 시각화 중...")
    timestamp = str(int(os.path.getctime(model_path)))
    save_path = f"./evaluation_results_{timestamp}.png"
    
    visualize_evaluation_results(
        model, eval_dataset, device, 
        num_examples=num_visualize, save_path=save_path
    )
    
    return avg_metrics, pattern_metrics

def main():
    """메인 함수 - 명령행 인수 처리"""
    parser = argparse.ArgumentParser(description='확산 기반 경로 계획 - 훈련 및 평가')
    parser.add_argument('--mode', choices=['train', 'eval'], required=True,
                       help='실행 모드: train (훈련) 또는 eval (평가)')
    parser.add_argument('--model_path', type=str, default='./models/final_model.pth',
                       help='평가할 모델 경로 (평가 모드용)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='평가할 샘플 수')
    parser.add_argument('--num_visualize', type=int, default=6,
                       help='시각화할 예제 수')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 훈련 모드 시작")
        main_training()
        
    elif args.mode == 'eval':
        print("📊 평가 모드 시작")
        if not os.path.exists(args.model_path):
            print(f"❌ 모델 파일이 존재하지 않습니다: {args.model_path}")
            
            # 사용 가능한 모델들 찾기
            models_dir = Path('./models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pth'))
                if model_files:
                    print("\n사용 가능한 모델들:")
                    for i, model_file in enumerate(model_files):
                        print(f"  {i+1}. {model_file}")
                    
                    choice = input("\n사용할 모델 번호를 입력하세요 (기본값: 1): ").strip()
                    try:
                        choice_idx = int(choice) - 1 if choice else 0
                        args.model_path = str(model_files[choice_idx])
                        print(f"선택된 모델: {args.model_path}")
                    except (ValueError, IndexError):
                        print("잘못된 선택입니다. 첫 번째 모델을 사용합니다.")
                        args.model_path = str(model_files[0])
                else:
                    print("❌ ./models 폴더에 모델 파일이 없습니다.")
                    return
            else:
                print("❌ ./models 폴더가 존재하지 않습니다.")
                return
        
        main_evaluation(args.model_path, args.num_samples, args.num_visualize)

def interactive_evaluation():
    """대화형 평가 모드"""
    print("🎮 대화형 평가 모드")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 사용 가능한 모델 찾기
    models_dir = Path('./models')
    if not models_dir.exists():
        print("❌ ./models 폴더가 존재하지 않습니다.")
        return
    
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        print("❌ ./models 폴더에 모델 파일이 없습니다.")
        return
    
    print("\n📁 사용 가능한 모델들:")
    for i, model_file in enumerate(model_files):
        # 파일 정보 가져오기
        stat = model_file.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = os.path.getctime(str(model_file))
        
        print(f"  {i+1}. {model_file.name}")
        print(f"      크기: {size_mb:.1f}MB")
        print(f"      생성: {np.datetime64(int(mtime), 's')}")
        print()
    
    # 모델 선택
    while True:
        try:
            choice = input("평가할 모델 번호를 입력하세요 (1-{}): ".format(len(model_files))).strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_model = str(model_files[choice_idx])
                break
            else:
                print("❌ 잘못된 번호입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    print(f"\n✅ 선택된 모델: {Path(selected_model).name}")
    
    # 모델 로드
    try:
        model = load_model_checkpoint(selected_model, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("🎯 평가 옵션을 선택하세요:")
        print("1. 빠른 평가 (50 샘플)")
        print("2. 표준 평가 (100 샘플)")
        print("3. 정밀 평가 (200 샘플)")
        print("4. 시각화만 (6 예제)")
        print("5. 커스텀 설정")
        print("0. 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            num_samples, num_visualize = 50, 6
        elif choice == '2':
            num_samples, num_visualize = 100, 6
        elif choice == '3':
            num_samples, num_visualize = 200, 9
        elif choice == '4':
            num_samples, num_visualize = 10, 6
        elif choice == '5':
            try:
                num_samples = int(input("평가 샘플 수 (기본 100): ") or "100")
                num_visualize = int(input("시각화 예제 수 (기본 6): ") or "6")
                num_samples = max(10, min(500, num_samples))
                num_visualize = max(3, min(12, num_visualize))
            except ValueError:
                print("❌ 잘못된 입력입니다. 기본값을 사용합니다.")
                num_samples, num_visualize = 100, 6
        else:
            print("❌ 잘못된 선택입니다.")
            continue
        
        print(f"\n🚀 평가 시작: {num_samples} 샘플, {num_visualize} 시각화")
        
        # 평가용 데이터셋 생성
        eval_dataset = generate_diffusion_dataset(num_samples=num_samples + 50, max_path_length=16)
        
        if len(eval_dataset) < 20:
            print(f"❌ 평가용 데이터셋 생성 실패")
            continue
        
        # 평가 실행
        try:
            avg_metrics, pattern_metrics, all_metrics = evaluate_model_on_dataset(
                model, eval_dataset, device, num_samples=num_samples
            )
            
            # 결과 출력
            print_evaluation_report(avg_metrics, pattern_metrics)
            
            # 시각화 여부 확인
            vis_choice = input("\n시각화를 표시하시겠습니까? (y/N): ").strip().lower()
            if vis_choice in ['y', 'yes']:
                timestamp = str(int(time.time()))
                save_path = f"./evaluation_{timestamp}.png"
                
                visualize_evaluation_results(
                    model, eval_dataset, device, 
                    num_examples=num_visualize, save_path=save_path
                )
            
            # 결과 저장 여부 확인
            save_choice = input("\n결과를 파일로 저장하시겠습니까? (y/N): ").strip().lower()
            if save_choice in ['y', 'yes']:
                save_evaluation_results(avg_metrics, pattern_metrics, all_metrics, selected_model)
                
        except Exception as e:
            print(f"❌ 평가 실행 실패: {e}")
            continue

def save_evaluation_results(avg_metrics, pattern_metrics, all_metrics, model_path):
    """평가 결과를 파일로 저장"""
    import json
    import time
    
    timestamp = int(time.time())
    results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'average_metrics': avg_metrics,
        'pattern_metrics': pattern_metrics,
        'detailed_metrics': {key: [float(v) for v in values] for key, values in all_metrics.items()}
    }
    
    # JSON 파일로 저장
    results_file = f"./evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 평가 결과 저장됨: {results_file}")
    
    # 텍스트 보고서도 생성
    report_file = f"./evaluation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("📊 모델 평가 보고서\n")
        f.write("="*60 + "\n")
        f.write(f"모델 경로: {model_path}\n")
        f.write(f"평가 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n\n")
        
        f.write("🎯 전체 성능 메트릭:\n")
        f.write(f"  충돌률:     {avg_metrics['collision_rate']:.3f} ± {avg_metrics['collision_rate_std']:.3f}\n")
        f.write(f"  길이 비율:  {avg_metrics['length_ratio']:.2f} ± {avg_metrics['length_ratio_std']:.2f}\n")
        f.write(f"  부드러움:   {avg_metrics['smoothness_ratio']:.2f} ± {avg_metrics['smoothness_ratio_std']:.2f}\n")
        f.write(f"  목표 오차:  {avg_metrics['goal_error']:.2f} ± {avg_metrics['goal_error_std']:.2f}\n\n")
        
        f.write("🏗️ 패턴별 성능:\n")
        for pattern, metrics in pattern_metrics.items():
            f.write(f"\n  {pattern.upper()} 패턴:\n")
            f.write(f"    충돌률:   {metrics['collision_rate']:.3f}\n")
            f.write(f"    길이 비율: {metrics['length_ratio']:.2f}\n")
            f.write(f"    부드러움:  {metrics['smoothness_ratio']:.2f}\n")
            f.write(f"    목표 오차: {metrics['goal_error']:.2f}\n")
    
    print(f"✅ 텍스트 보고서 저장됨: {report_file}")

# 사용성 개선을 위한 추가 유틸리티 함수들

def find_best_model(models_dir='./models'):
    """최고 성능 모델 자동 찾기"""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None
    
    model_files = list(models_path.glob('*.pth'))
    if not model_files:
        return None
    
    # final_model.pth가 있으면 우선 선택
    final_model = models_path / 'final_model.pth'
    if final_model.exists():
        return str(final_model)
    
    # 그렇지 않으면 가장 높은 에포크의 체크포인트 선택
    checkpoint_files = [f for f in model_files if 'checkpoint_epoch_' in f.name]
    if checkpoint_files:
        # 에포크 번호로 정렬
        checkpoint_files.sort(key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
        return str(checkpoint_files[-1])  # 가장 높은 에포크
    
    # 그것도 없으면 가장 최근 파일
    model_files.sort(key=lambda x: x.stat().st_mtime)
    return str(model_files[-1])

def quick_demo(model_path=None):
    """빠른 데모 실행"""
    print("🎮 빠른 데모 모드")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 자동 찾기 또는 지정된 경로 사용
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("❌ 사용 가능한 모델이 없습니다.")
            return
        print(f"📁 자동 선택된 모델: {Path(model_path).name}")
    
    # 모델 로드
    try:
        model = load_model_checkpoint(model_path, device)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 간단한 평가 데이터셋 생성
    print("\n📊 간단한 성능 테스트 실행 중...")
    eval_dataset = generate_diffusion_dataset(num_samples=50, max_path_length=16)
    
    # 빠른 평가
    avg_metrics, _, _ = evaluate_model_on_dataset(model, eval_dataset, device, num_samples=30)
    
    # 간단한 결과 출력
    print(f"\n🎯 성능 요약:")
    print(f"  충돌률: {avg_metrics['collision_rate']:.3f} ({'우수' if avg_metrics['collision_rate'] < 0.05 else '양호' if avg_metrics['collision_rate'] < 0.15 else '개선필요'})")
    print(f"  경로 효율: {avg_metrics['length_ratio']:.2f} ({'우수' if 0.9 <= avg_metrics['length_ratio'] <= 1.3 else '양호' if 0.8 <= avg_metrics['length_ratio'] <= 1.5 else '개선필요'})")
    
    # 시각화
    print("\n🎨 시각화 생성 중...")
    visualize_evaluation_results(model, eval_dataset, device, num_examples=3)

# 메인 실행 부분 업데이트
if __name__ == "__main__":
    import sys
    import time
    
    # 명령행 인수가 없으면 대화형 모드
    if len(sys.argv) == 1:
        print("🎮 대화형 모드로 실행합니다.")
        print("명령행 옵션을 원하시면: python script.py --help")
        
        print("\n실행 모드를 선택하세요:")
        print("1. 훈련 (Training)")
        print("2. 대화형 평가 (Interactive Evaluation)")  
        print("3. 빠른 데모 (Quick Demo)")
        print("0. 종료")
        
        choice = input("\n선택 (0-3): ").strip()
        
        if choice == '1':
            main_training()
        elif choice == '2':
            interactive_evaluation()
        elif choice == '3':
            quick_demo()
        elif choice == '0':
            print("종료합니다.")
        else:
            print("잘못된 선택입니다.")
    else:
        # 명령행 인수가 있으면 기존 방식 사용
        main()