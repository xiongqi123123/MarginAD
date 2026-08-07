import math
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

@PLUGIN_LAYERS.register_module()
class PhysicsKinematicModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=6,
        max_acceleration=3.0,
        max_deceleration=5.0,
        max_steering_angle=0.6,
        max_curvature=0.2,
        wheelbase=2.7,
        use_velocity_constraint=True,
        use_acceleration_constraint=True,
        use_steering_constraint=True,
        use_curvature_constraint=True,
        alpha_init=0.5,
    ):
        super(PhysicsKinematicModule, self).__init__()
        
        self.embed_dims = embed_dims
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_curvature = max_curvature
        
        self.use_velocity_constraint = use_velocity_constraint
        self.use_acceleration_constraint = use_acceleration_constraint
        self.use_steering_constraint = use_steering_constraint
        self.use_curvature_constraint = use_curvature_constraint
        
        if self.use_velocity_constraint:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(3, embed_dims // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dims // 2),
                nn.Linear(embed_dims // 2, embed_dims),
            )
        else:
            self.velocity_encoder = None
    
        if self.use_acceleration_constraint:
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(3, embed_dims // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dims // 2),
                nn.Linear(embed_dims // 2, embed_dims),
            )
        else:
            self.acceleration_encoder = None
        
        if self.use_steering_constraint:
            self.steering_encoder = nn.Sequential(
                nn.Linear(2, embed_dims // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dims // 2),
                nn.Linear(embed_dims // 2, embed_dims),
            )
        else:
            self.steering_encoder = None
            
        if self.use_curvature_constraint:
            self.curvature_encoder = nn.Sequential(
                nn.Linear(2, embed_dims // 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dims // 2),
                nn.Linear(embed_dims // 2, embed_dims),
            )
        else:
            self.curvature_encoder = None
        
        fusion_input_dim = embed_dims 
        if use_velocity_constraint:
            fusion_input_dim += embed_dims
        if use_acceleration_constraint:
            fusion_input_dim += embed_dims
        if use_steering_constraint:
            fusion_input_dim += embed_dims
        if use_curvature_constraint:
            fusion_input_dim += embed_dims
            
        self.physics_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dims * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dims * 2),
            nn.Dropout(0.1),
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(),
            nn.LayerNorm(embed_dims),
        )
        if 0 < alpha_init < 1:
            raw = math.log(alpha_init / (1.0 - alpha_init))
        else:
            raw = 0.0
        self.alpha = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        
    def compute_velocity_features(self, trajectory):
        dt = 0.5
        velocities = torch.diff(trajectory, dim=-2) / dt
        v_mag = torch.sqrt(velocities[..., 0] ** 2 + velocities[..., 1] ** 2)
        v_start = v_mag[..., 0:1]
        v_end = v_mag[..., -1:]
        v_mean = v_mag.mean(dim=-1, keepdim=True)
        return torch.cat([v_start, v_end, v_mean], dim=-1)
    
    def compute_acceleration_features(self, trajectory):
        dt = 0.5
        velocities = torch.diff(trajectory, dim=-2) / dt
        accelerations = torch.diff(velocities, dim=-2) / dt
        a_mag = torch.sqrt(accelerations[..., 0] ** 2 + accelerations[..., 1] ** 2)
        a_start = a_mag[..., 0:1]
        a_end = a_mag[..., -1:]
        a_mean = a_mag.mean(dim=-1, keepdim=True)
        return torch.cat([a_start, a_end, a_mean], dim=-1)
    
    def compute_steering_features(self, trajectory):
        dt = 0.5
        velocities = torch.diff(trajectory, dim=-2) / dt
        headings = torch.atan2(velocities[..., 1], velocities[..., 0])
        heading_rates = torch.diff(headings, dim=-1) / dt
        
        v = torch.sqrt(velocities[..., 0]**2 + velocities[..., 1]**2)
        v_safe = torch.clamp(v[:, :, :, :-1], min=0.1)
        
        curvature = heading_rates / v_safe
        steering_angle = torch.atan(self.wheelbase * curvature)
        
        steering_angle_mean = steering_angle.mean(dim=-1, keepdim=True)
        steering_rate = torch.diff(steering_angle, dim=-1).mean(dim=-1, keepdim=True)
        
        steering_angle_clipped = torch.clamp(
            steering_angle_mean, -self.max_steering_angle, self.max_steering_angle
        )
        return torch.cat([steering_angle_clipped, steering_rate], dim=-1)
    
    def compute_curvature_features(self, trajectory):
        dx = torch.diff(trajectory[..., 0], dim=-1)
        dy = torch.diff(trajectory[..., 1], dim=-1)
        ddx = torch.diff(dx, dim=-1)
        ddy = torch.diff(dy, dim=-1)
        
        numerator = torch.abs(dx[:, :, :, :-1] * ddy - dy[:, :, :, :-1] * ddx)
        denominator = torch.pow(dx[:, :, :, :-1]**2 + dy[:, :, :, :-1]**2, 1.5)
        denominator = torch.clamp(denominator, min=1e-6)
        
        curvature = numerator / denominator
        curvature_mean = curvature.mean(dim=-1, keepdim=True)
        curvature_rate = torch.diff(curvature, dim=-1).mean(dim=-1, keepdim=True)
        
        curvature_clipped = torch.clamp(curvature_mean, 0, self.max_curvature)
        return torch.cat([curvature_clipped, curvature_rate], dim=-1)
    
    def forward(self, plan_query, planning_prediction):
        B, _, num_modes, D = plan_query.shape
        trajectory = planning_prediction.cumsum(dim=-2)
        
        physics_features = []
        
        if self.use_velocity_constraint and self.velocity_encoder is not None:
            velocity_feat = self.compute_velocity_features(trajectory)
            velocity_embed = self.velocity_encoder(velocity_feat)
            physics_features.append(velocity_embed)
        
        if self.use_acceleration_constraint and self.acceleration_encoder is not None:
            acceleration_feat = self.compute_acceleration_features(trajectory)
            acceleration_embed = self.acceleration_encoder(acceleration_feat)
            physics_features.append(acceleration_embed)
        
        if self.use_steering_constraint and self.steering_encoder is not None:
            steering_feat = self.compute_steering_features(trajectory)
            steering_embed = self.steering_encoder(steering_feat)
            physics_features.append(steering_embed)
        
        if self.use_curvature_constraint and self.curvature_encoder is not None:
            curvature_feat = self.compute_curvature_features(trajectory)
            curvature_embed = self.curvature_encoder(curvature_feat)
            physics_features.append(curvature_embed)
        
        fusion_input = torch.cat([plan_query] + physics_features, dim=-1)
        
        fused_features = self.physics_fusion(fusion_input)
        
        alpha = torch.sigmoid(self.alpha)
        physics_enhanced_query = alpha * plan_query + (1 - alpha) * fused_features
        
        return physics_enhanced_query

@PLUGIN_LAYERS.register_module()
class PhysicsConstraintLoss(nn.Module):
    def __init__(
        self,
        max_acceleration=3.0,
        max_deceleration=5.0,
        max_steering_angle=0.6,
        max_curvature=0.2,
        wheelbase=2.7,  
        loss_weight_acceleration=0.1,
        loss_weight_steering=0.1,
        loss_weight_curvature=0.1,
        forward_only=False,
        stage_weights=None,
    ):
        super(PhysicsConstraintLoss, self).__init__()
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_steering_angle = max_steering_angle
        self.max_curvature = max_curvature
        self.wheelbase = wheelbase
        self.loss_weight_acceleration = loss_weight_acceleration
        self.loss_weight_steering = loss_weight_steering
        self.loss_weight_curvature = loss_weight_curvature
        self.forward_only = forward_only
        self.stage_weights = stage_weights

    def compute_acceleration_violation(self, trajectory, forward_only=None):
        forward_only = self.forward_only if forward_only is None else forward_only
        dt = 0.5
        velocities = torch.diff(trajectory, dim=-2) / dt
        accelerations = torch.diff(velocities, dim=-2) / dt
        if forward_only:
            v_for_a = velocities[..., :-1, :]
            v_norm = torch.linalg.norm(v_for_a, dim=-1, keepdim=True).clamp(min=1e-6)
            heading = v_for_a / v_norm
            a_along = (accelerations * heading).sum(dim=-1)
            acc_violation = (
                torch.clamp(a_along - self.max_acceleration, min=0.0)
                + torch.clamp(-a_along - self.max_deceleration, min=0.0)
            )
        else:
            a_magnitude = torch.sqrt(accelerations[..., 0]**2 + accelerations[..., 1]**2)
            if self.training and torch.rand(1).item() < 0.001:
                print(f"[Physics Debug] Acc Max: {a_magnitude.max():.2f}")
            acc_violation = torch.clamp(a_magnitude - self.max_acceleration, min=0.0)
        return acc_violation.mean()
    
    def compute_steering_violation(self, trajectory):
        dt = 0.5
        velocities = torch.diff(trajectory, dim=-2) / dt
        headings = torch.atan2(velocities[..., 1], velocities[..., 0])
        heading_rates = torch.diff(headings, dim=-1) / dt
        
        v = torch.sqrt(velocities[..., 0]**2 + velocities[..., 1]**2)
        v_safe = torch.clamp(v[:, :, :, :-1], min=0.1)
        
        curvature = heading_rates / v_safe
        steering_angle = torch.atan(self.wheelbase * curvature)
        
        steering_violation = torch.clamp(
            torch.abs(steering_angle) - self.max_steering_angle, 
            min=0.0
        )
        return steering_violation.mean()
    
    def compute_curvature_violation(self, trajectory):
        dx = torch.diff(trajectory[..., 0], dim=-1)
        dy = torch.diff(trajectory[..., 1], dim=-1)
        ddx = torch.diff(dx, dim=-1)
        ddy = torch.diff(dy, dim=-1)
        
        numerator = torch.abs(dx[:, :, :, :-1] * ddy - dy[:, :, :, :-1] * ddx)
        denominator = torch.pow(dx[:, :, :, :-1]**2 + dy[:, :, :, :-1]**2, 1.5)
        denominator = torch.clamp(denominator, min=1e-6)
        
        curvature = numerator / denominator
        curvature_violation = torch.clamp(curvature - self.max_curvature, min=0.0)
        return curvature_violation.mean()
    
    def _compute_loss_for_trajectory(self, trajectory):
        loss = trajectory.new_tensor(0.0)
        if self.loss_weight_acceleration > 0:
            loss = loss + self.loss_weight_acceleration * self.compute_acceleration_violation(trajectory)
        if self.loss_weight_steering > 0:
            loss = loss + self.loss_weight_steering * self.compute_steering_violation(trajectory)
        if self.loss_weight_curvature > 0:
            loss = loss + self.loss_weight_curvature * self.compute_curvature_violation(trajectory)
        return loss

    def forward(self, planning_prediction):
        trajectory = planning_prediction.cumsum(dim=-2)
        loss_dict = {}

        if self.stage_weights is not None:
            w_near, w_far = self.stage_weights
            T = trajectory.shape[-2]
            mid = max(1, T // 2)
            traj_near = trajectory[..., :mid + 1, :]
            traj_far = trajectory[..., mid:, :]
            loss_near = self._compute_loss_for_trajectory(traj_near)
            loss_far = self._compute_loss_for_trajectory(traj_far)
            total = w_near * loss_near + w_far * loss_far
            loss_dict['loss_physics_constraint'] = total
            return loss_dict
        if self.loss_weight_acceleration > 0:
            loss_dict['loss_acceleration'] = (
                self.loss_weight_acceleration * self.compute_acceleration_violation(trajectory)
            )
        if self.loss_weight_steering > 0:
            loss_dict['loss_steering'] = (
                self.loss_weight_steering * self.compute_steering_violation(trajectory)
            )
        if self.loss_weight_curvature > 0:
            loss_dict['loss_curvature'] = (
                self.loss_weight_curvature * self.compute_curvature_violation(trajectory)
            )
        return loss_dict