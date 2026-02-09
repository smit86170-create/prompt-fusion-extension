import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from modules import scripts, shared, sd_samplers, sd_samplers_common, sd_samplers_kdiffusion
from modules.script_callbacks import on_cfg_denoiser
import k_diffusion.sampling
from k_diffusion.sampling import to_d, default_noise_sampler, get_ancestral_step
from tqdm.auto import trange
import math
import numpy as np
from collections import OrderedDict
import os
import sys

from PIL import Image

# === НАЧАЛО ИСПРАВЛЕНИЯ ===
# Получаем путь к папке, где лежит этот файл (asymmetric_tiling_UNIFIED.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Добавляем этот путь в sys.path, чтобы Python мог найти соседние файлы
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)
# === КОНЕЦ ИСПРАВЛЕНИЯ ===

from improved_tiling_functions import (
    compute_advanced_blend_padding,
    compute_blend_fade_to_black,
    apply_multires_blend,
    validate_blend_params,
    validate_multires_params,
    BlendStrategy,
    MultiResStrategy
)
from advanced_zoom_extension import (
    apply_unified_zoom,
    validate_zoom_params,
    ZoomMode,
    BlendMode
)
# ========================================================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ========================================================================
MODE_OFF = "Default (Off)"
MODE_CIRCULAR = "Circular"
MODE_MIRROR = "Mirror (Reflect)"
MODE_HEXAGONAL = "Hexagonal (Staggered)"
MODE_PANORAMA = "Panorama 360°"
MODE_CUBEMAP = "Cubemap (3D)"
MODE_BLEND = "Soft Blend Edges"
MODE_ANISOTROPIC = "Anisotropic (Directional)"
MODE_POLAR = "Polar (Sphere Correct)"

# NEW MODES - Advanced Features from ENHANCED
MODE_VORONOI = "Voronoi (Organic)"
MODE_PERLIN = "Perlin Noise Distortion"
MODE_FRACTAL = "Fractal Recursive"
MODE_ADAPTIVE = "Adaptive Smart"

# Глобальное хранилище
_ORIGINAL_METHODS_CACHE = {}
_MASK_CACHE = {}
_CUBEMAP_GRID_CACHE = {}
_SAMPLER_REGISTERED = False


# Panorama Live caches
_PANO_GRID_CACHE = {}
_BLUR_KERNEL_CACHE = {}
# ========================================================================
# KOHAKU LONYU YOG SAMPLER IMPLEMENTATION
# ========================================================================

@torch.no_grad()
def sample_kohaku_lonyu_yog(model, x, sigmas, extra_args=None, callback=None,
                            disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'),
                            s_noise=1., noise_sampler=None, eta=1.):
    """
    Kohaku_LoNyu_Yog Sampler - Geometric Second-Order Method
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    steps_total = len(sigmas) - 1
    halfway_point = steps_total // 2

    for i in trange(steps_total, disable=disable, desc="Kohaku Sampling"):
        gamma = min(s_churn / steps_total, 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        dt = sigma_down - sigmas[i]

        if i <= halfway_point:
            x_antipode = -x

            denoised2 = model(x_antipode, sigma_hat * s_in, **extra_args)
            d2 = to_d(x_antipode, sigma_hat, denoised2)

            v_down = (d + d2) / 2
            x_closer = x + v_down * dt

            denoised3 = model(x_closer, sigma_hat * s_in, **extra_args)
            d3 = to_d(x_closer, sigma_hat, denoised3)

            real_d = (d + d3) / 2
            x = x + real_d * dt

            if sigma_up > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
            if sigma_up > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigmas[i],
                'sigma_hat': sigma_hat,
                'denoised': denoised
            })

    return x

# ========================================================================
# LRU CACHE SYSTEM - Enhanced memory management
# ========================================================================

class LRUCache:
    """Thread-safe LRU cache with size limits and memory management"""
    def __init__(self, max_size=50, max_memory_mb=500):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
    
    def _estimate_size_mb(self, tensor):
        """Estimate tensor size in MB"""
        if isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement() / (1024 * 1024)
        return 0
    
    def get(self, key):
        """Get cached value, move to end (most recent)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Set value with automatic eviction if needed"""
        size_mb = self._estimate_size_mb(value)
        
        # Evict old entries if needed
        while (len(self.cache) >= self.max_size or 
               self.current_memory_mb + size_mb > self.max_memory_mb):
            if len(self.cache) == 0:
                break
            old_key, old_value = self.cache.popitem(last=False)
            self.current_memory_mb -= self._estimate_size_mb(old_value)
            if isinstance(old_value, torch.Tensor) and old_value.is_cuda:
                del old_value
        
        self.cache[key] = value
        self.cache.move_to_end(key)
        self.current_memory_mb += size_mb
    
    def clear(self):
        """Clear all cache"""
        for value in self.cache.values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                del value
        self.cache.clear()
        self.current_memory_mb = 0

# Enhanced cache instances (keep compatibility with old dict-based caches)
_VORONOI_CACHE = LRUCache(max_size=20, max_memory_mb=100)

# ========================================================================
# VALIDATION UTILITIES - Protect against NaN/Inf
# ========================================================================

def validate_float(value, min_val=None, max_val=None, default=0.0, name="parameter"):
    """Safely validate and clamp float values"""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            print(f"⚠ Warning: {name} is NaN/Inf, using default {default}")
            return default
        if min_val is not None and val < min_val:
            return min_val
        if max_val is not None and val > max_val:
            return max_val
        return val
    except (ValueError, TypeError):
        print(f"⚠ Warning: Invalid {name}, using default {default}")
        return default

def validate_int(value, min_val=None, max_val=None, default=0, name="parameter"):
    """Safely validate and clamp integer values"""
    try:
        val = int(value)
        if min_val is not None and val < min_val:
            return min_val
        if max_val is not None and val > max_val:
            return max_val
        return val
    except (ValueError, TypeError):
        print(f"⚠ Warning: Invalid {name}, using default {default}")
        return default

def validate_tensor(tensor, name="tensor"):
    """Check tensor for NaN/Inf values"""
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            print(f"⚠ Warning: {name} contains NaN values!")
            return torch.nan_to_num(tensor, nan=0.0)
        if torch.isinf(tensor).any():
            print(f"⚠ Warning: {name} contains Inf values!")
            return torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
    return tensor

# ========================================================================
# РАСШИРЕННЫЕ ФУНКЦИИ ПАДДИНГА
# ========================================================================

def get_or_create_mask(h, w, device):
    """Кэширование масок для оптимизации"""
    key = (h, w, str(device))
    if key not in _MASK_CACHE:
        row_indices = torch.arange(h, device=device).view(1, 1, h, 1)
        _MASK_CACHE[key] = (row_indices % 2 == 1)
    return _MASK_CACHE[key]

def compute_anisotropic_padding(input_tensor, pad_h, pad_w, angle_deg=45, angle_deg2=None, angle_mix=1.0):
    """
    Анизотропный паддинг - разное поведение по диагоналям.
    Эмулирует направленные материалы (дерево, металл, волокна).
    """
    b, c, h, w = input_tensor.shape

    # Базовые паддинги: circular и reflect
    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    padded_reflect = _safe_pad4d(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # Размеры уже с учетом паддинга
    _, H, W = padded.shape[1:]

    # Преобразуем угол в радианы
    angle_rad = math.radians(angle_deg)

    # Координаты в нормализованной системе для всего padded-тензора
    y_coords = torch.linspace(-1.0, 1.0, steps=H, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 1, H, 1)
    x_coords = torch.linspace(-1.0, 1.0, steps=W, device=input_tensor.device, dtype=input_tensor.dtype).view(1, 1, 1, W)

    # Проекция на направление волокон
    directional_component = x_coords * math.cos(angle_rad) + y_coords * math.sin(angle_rad)
    directional_strength = directional_component.abs()

    # Optional second direction (advanced): blend two angle fields
    if angle_deg2 is not None:
        angle_mix = float(max(0.0, min(float(angle_mix), 1.0)))
        angle_rad2 = math.radians(float(angle_deg2))
        directional_component2 = x_coords * math.cos(angle_rad2) + y_coords * math.sin(angle_rad2)
        directional_strength2 = directional_component2.abs()
        directional_strength = directional_strength * angle_mix + directional_strength2 * (1.0 - angle_mix)

    # Альфа-блендинг: вдоль направления больше circular, поперек больше reflect
    alpha = directional_strength.clamp(0.0, 1.0)
    result = padded * alpha + padded_reflect * (1.0 - alpha)

    return result

def compute_polar_padding(input_tensor, pad_h, pad_w):
    """
    Полярный паддинг для сферических проекций.
    """
    b, c, h, w = input_tensor.shape

    # X-axis: стандартный circular (долгота замыкается)
    x = F.pad(input_tensor, (pad_w, pad_w, 0, 0), mode='circular')

    # Y-axis: полярная коррекция (широта через полюса)
    shift = w // 2

    # Верхний паддинг (Северный полюс)
    top_strip = x[:, :, :pad_h, :]
    top_pad = torch.roll(top_strip, shifts=shift, dims=3)
    top_pad = torch.flip(top_pad, dims=[2])

    # Нижний паддинг (Южный полюс)
    bot_strip = x[:, :, -pad_h:, :]
    bot_pad = torch.roll(bot_strip, shifts=shift, dims=3)
    bot_pad = torch.flip(bot_pad, dims=[2])

    result = torch.cat([top_pad, x, bot_pad], dim=2)
    return result


# ========================================================================
# NEW ENHANCED MODES - Voronoi, Perlin, Fractal, Adaptive
# ========================================================================

# ===== EDGE DETECTION for Adaptive Mode =====

def detect_edges_sobel(tensor):
    """
    Detect edges using Sobel operator for adaptive padding
    Returns edge map [0, 1]
    """
    # Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    
    # Convert to grayscale if needed
    if tensor.shape[1] > 1:
        gray = tensor.mean(dim=1, keepdim=True)
    else:
        gray = tensor
    
    # Apply Sobel
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    # Magnitude
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edges = edges / (edges.max() + 1e-8)  # Normalize
    
    return edges

# ===== VORONOI ORGANIC TILING =====

def compute_voronoi_padding(input_tensor, pad_h, pad_w, num_cells=8, seed=42):
    """
    Voronoi diagram-based organic tiling
    Creates natural-looking tile patterns based on cellular structures
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # Validate parameters
    num_cells = validate_int(num_cells, min_val=4, max_val=32, default=8, name="voronoi_cells")
    seed = validate_int(seed, min_val=0, max_val=99999, default=42, name="voronoi_seed")
    
    # Cache key
    cache_key = (h, w, pad_h, pad_w, num_cells, seed, str(device))
    cached_map = _VORONOI_CACHE.get(cache_key)
    
    if cached_map is None:
        # Create extended canvas
        H_ext = h + 2 * pad_h
        W_ext = w + 2 * pad_w
        
        # Generate random Voronoi cell centers
        torch.manual_seed(seed)
        centers_y = torch.rand(num_cells, device=device) * H_ext
        centers_x = torch.rand(num_cells, device=device) * W_ext
        
        # Create coordinate grids
        y_grid = torch.arange(H_ext, device=device, dtype=torch.float32).view(-1, 1)
        x_grid = torch.arange(W_ext, device=device, dtype=torch.float32).view(1, -1)
        
        # Compute distances to all centers
        distances = []
        for i in range(num_cells):
            dist = torch.sqrt((y_grid - centers_y[i])**2 + (x_grid - centers_x[i])**2)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=0)
        
        # Find nearest cell for each pixel
        nearest_cell = torch.argmin(distances, dim=0)
        
        # Create mapping from extended to source coordinates
        voronoi_map = torch.zeros(2, H_ext, W_ext, device=device, dtype=torch.float32)
        
        for cell_id in range(num_cells):
            mask = (nearest_cell == cell_id)
            # Map this region to corresponding source region (with wrapping)
            y_offset = (centers_y[cell_id] % h) - h // 2
            x_offset = (centers_x[cell_id] % w) - w // 2
            
            y_coords_masked = y_grid.expand_as(mask)[mask]
            x_coords_masked = x_grid.expand_as(mask)[mask]
            
            voronoi_map[0][mask] = (y_coords_masked - y_offset) % h
            voronoi_map[1][mask] = (x_coords_masked - x_offset) % w
        
        _VORONOI_CACHE.set(cache_key, voronoi_map)
        cached_map = voronoi_map
    
    # Apply mapping using grid_sample
    # Normalize coordinates to [-1, 1]
    grid = cached_map.permute(1, 2, 0).unsqueeze(0)  # [1, H_ext, W_ext, 2]
    grid[..., 0] = (grid[..., 0] / (h - 1)) * 2 - 1  # y
    grid[..., 1] = (grid[..., 1] / (w - 1)) * 2 - 1  # x
    
    # Sample from input
    result = F.grid_sample(
        input_tensor, 
        grid.expand(b, -1, -1, -1),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return result

# ===== PERLIN NOISE DISTORTION =====

def generate_perlin_noise(height, width, scale=10.0, octaves=4, persistence=0.5, device='cuda'):
    """
    Generate Perlin-like noise using multiple octaves of smoothed noise
    """
    def smooth_noise(noise):
        return F.avg_pool2d(
            F.pad(noise, (1, 1, 1, 1), mode='reflect'),
            kernel_size=3, stride=1, padding=0
        )
    
    total_noise = torch.zeros(1, 1, height, width, device=device)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        # Generate random noise at this frequency
        noise_h = int(height / scale * frequency) + 1
        noise_w = int(width / scale * frequency) + 1
        
        noise = torch.rand(1, 1, noise_h, noise_w, device=device) * 2 - 1
        
        # Smooth and upscale
        for _ in range(3):  # Multiple smoothing passes
            noise = smooth_noise(noise)
        
        # Resize to target size
        noise = F.interpolate(noise, size=(height, width), mode='bilinear', align_corners=False)
        
        total_noise += noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= 2.0
    
    # Normalize
    total_noise /= max_value
    return total_noise

def compute_perlin_padding(input_tensor, pad_h, pad_w, strength=0.3, scale=10.0):
    """
    Apply Perlin noise distortion for natural-looking tiling
    """
    # Validate parameters
    strength = validate_float(strength, min_val=0.0, max_val=1.0, default=0.3, name="perlin_strength")
    scale = validate_float(scale, min_val=1.0, max_val=50.0, default=10.0, name="perlin_scale")
    
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # Start with circular padding
    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    H_ext, W_ext = padded.shape[-2:]
    
    # Generate Perlin noise for displacement
    noise_y = generate_perlin_noise(H_ext, W_ext, scale=scale, device=device)
    noise_x = generate_perlin_noise(H_ext, W_ext, scale=scale * 1.3, device=device)
    
    # Create coordinate grid
    y_coords = torch.linspace(0, h - 1, H_ext, device=device).view(1, 1, H_ext, 1).expand(1, 1, H_ext, W_ext)
    x_coords = torch.linspace(0, w - 1, W_ext, device=device).view(1, 1, 1, W_ext).expand(1, 1, H_ext, W_ext)
    
    # Apply distortion
    distortion_strength = strength * max(pad_h, pad_w)
    y_distorted = (y_coords + noise_y * distortion_strength) % h
    x_distorted = (x_coords + noise_x * distortion_strength) % w
    
    # Create sampling grid
    # FIX V3.6: Убираем лишнюю размерность канала .squeeze(1)
    grid = torch.stack([x_distorted, y_distorted], dim=-1).squeeze(1)  # Становится (1, H_ext, W_ext, 2)
    
    grid[..., 0] = (grid[..., 0] / (w - 1)) * 2 - 1  # Normalize x to [-1, 1]
    grid[..., 1] = (grid[..., 1] / (h - 1)) * 2 - 1  # Normalize y to [-1, 1]
    
    # Sample with distortion
    result = F.grid_sample(
        input_tensor,
        grid.expand(b, -1, -1, -1),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return result

# ===== FRACTAL RECURSIVE TILING =====

def compute_fractal_padding(input_tensor, pad_h, pad_w, iterations=2, scale_factor=0.6):
    """
    Fractal recursive tiling - creates self-similar patterns
    Each iteration adds scaled version of the pattern
    """
    iterations = validate_int(iterations, min_val=1, max_val=4, default=2, name="fractal_iterations")
    scale_factor = validate_float(scale_factor, min_val=0.3, max_val=0.9, default=0.6, name="fractal_scale")
    
    # Base padding
    result = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    b, c, h_ext, w_ext = result.shape
    
    # Add fractal details
    for i in range(1, iterations + 1):
        scale = scale_factor ** i
        
        # Scale down the pattern
        scaled_h = max(8, int(h_ext * scale))
        scaled_w = max(8, int(w_ext * scale))
        
        scaled_pattern = F.interpolate(
            result,
            size=(scaled_h, scaled_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Tile it across the result
        tiles_y = (h_ext + scaled_h - 1) // scaled_h
        tiles_x = (w_ext + scaled_w - 1) // scaled_w
        
        tiled = scaled_pattern.repeat(1, 1, tiles_y, tiles_x)
        tiled = tiled[:, :, :h_ext, :w_ext]
        
        # Blend with decreasing strength
        alpha = 0.3 / (i + 1)
        result = result * (1 - alpha) + tiled * alpha
    
    return result

# ===== ADAPTIVE SMART MODE =====

def compute_adaptive_padding(input_tensor, pad_h, pad_w, edge_threshold=0.1):
    """
    Adaptive padding that analyzes image content and chooses best mode
    Uses edge detection to decide between circular, reflect, or blend
    """
    edge_threshold = validate_float(edge_threshold, min_val=0.0, max_val=1.0, default=0.1, name="adaptive_threshold")
    
    # Detect edges
    edges = detect_edges_sobel(input_tensor)
    
    # Analyze edge intensity at borders
    top_edge = edges[:, :, :min(pad_h, edges.shape[2]), :].mean()
    bottom_edge = edges[:, :, -min(pad_h, edges.shape[2]):, :].mean()
    left_edge = edges[:, :, :, :min(pad_w, edges.shape[3])].mean()
    right_edge = edges[:, :, :, -min(pad_w, edges.shape[3]):].mean()
    
    avg_edge = (top_edge + bottom_edge + left_edge + right_edge) / 4
    
    # Decide mode based on edge strength
    if avg_edge < edge_threshold:
        # Low edge content -> circular works well
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    elif avg_edge > 0.5:
        # High edge content -> use blend mode
        return compute_blend_padding(input_tensor, pad_h, pad_w, strength=0.7)
    else:
        # Medium edge content -> adaptive blend
        blend_strength = float(avg_edge) * 1.5  # Scale to [0, 0.75]
        return compute_blend_padding(input_tensor, pad_h, pad_w, strength=blend_strength)

# ===== ENHANCED BLEND MODE (improved version) =====

def create_edge_blend_mask(h, w, blend_width, device, dtype=torch.float32):
    """
    Create gradient mask for soft edge blending
    Returns mask with smooth falloff at borders
    """
    mask = torch.ones(h, w, device=device, dtype=dtype)
    
    if blend_width <= 0:
        return mask.unsqueeze(0).unsqueeze(0)
    
    blend_w = min(blend_width, w // 4)
    blend_h = min(blend_width, h // 4)
    
    # Horizontal edges (left and right)
    for i in range(blend_w):
        alpha = (i + 1) / (blend_w + 1)
        alpha = alpha ** 0.5  # Gamma correction
        mask[:, i] = alpha
        mask[:, -(i + 1)] = alpha
    
    # Vertical edges (top and bottom)
    for i in range(blend_h):
        alpha = (i + 1) / (blend_h + 1)
        alpha = alpha ** 0.5
        mask[i, :] = torch.minimum(mask[i, :], torch.tensor(alpha, device=device, dtype=dtype))
        mask[-(i + 1), :] = torch.minimum(mask[-(i + 1), :], torch.tensor(alpha, device=device, dtype=dtype))
    
    return mask.unsqueeze(0).unsqueeze(0)

def compute_blend_padding(input_tensor, pad_h, pad_w, strength=0.5, blend_width=None):
    """
    Enhanced Soft Blend Edges implementation
    Blends between circular and reflect padding modes at the borders
    """
    strength = validate_float(strength, min_val=0.0, max_val=1.0, default=0.5, name="blend_strength")
    
    # Auto blend width if not specified
    if blend_width is None:
        blend_width = max(pad_h, pad_w)
    
    # Create both padding modes
    circular = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    reflect = _safe_pad4d(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    
    # Create blend mask
    h, w = circular.shape[-2:]
    mask = create_edge_blend_mask(h, w, blend_width, input_tensor.device, input_tensor.dtype)
    
    # Apply blending with strength control
    alpha = mask * strength + (1 - strength) * 0.5
    alpha = alpha.expand_as(circular)
    
    result = circular * alpha + reflect * (1 - alpha)
    return result


# ===================================
# CUBEMAP (3D) — Engine A (Fast) + Engine B (Seam-Blend)
# ===================================

def _safe_pad4d(x, pad, mode='reflect', value=0.0):
    """
    Safe wrapper around F.pad for 4D tensors.
    - For mode='reflect', PyTorch requires pad < input_size.
      If invalid, we fall back to 'replicate' to avoid runtime errors.
    pad: (left, right, top, bottom)
    """
    if not isinstance(pad, (tuple, list)) or len(pad) != 4:
        return F.pad(x, pad, mode=mode, value=value) if mode == 'constant' else F.pad(x, pad, mode=mode)

    l, r, t, b = pad
    if mode == 'reflect':
        h = int(x.shape[-2])
        w = int(x.shape[-1])
        if (l >= w) or (r >= w) or (t >= h) or (b >= h):
            mode = 'replicate'

    if mode == 'constant':
        return F.pad(x, (l, r, t, b), mode=mode, value=value)
    return F.pad(x, (l, r, t, b), mode=mode)


def _cubemap_split_faces(x):
    """
    Splits a 3x2 cubemap net into faces.
    Layout expected (top row / bottom row):
        S | E | N
        B | T | W
    Returns tuple (S, E, N, B, T, W), each (B,C,h,w)
    """
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 3 != 0:
        raise ValueError("Cubemap expects H%2==0 and W%3==0 (3x2 net).")
    h, w = H // 2, W // 3
    S = x[:, :, 0:h, 0:w]
    E = x[:, :, 0:h, w:2*w]
    N = x[:, :, 0:h, 2*w:3*w]
    Bm = x[:, :, h:2*h, 0:w]
    T = x[:, :, h:2*h, w:2*w]
    Wf = x[:, :, h:2*h, 2*w:3*w]
    return S, E, N, Bm, T, Wf


def _cubemap_stitch_faces(S, E, N, Bm, T, Wf):
    """Stitches faces back into a 3x2 net (S/E/N over B/T/W)."""
    B, C, h, w = S.shape
    out = torch.zeros((B, C, h * 2, w * 3), device=S.device, dtype=S.dtype)
    out[:, :, 0:h, 0:w] = S
    out[:, :, 0:h, w:2*w] = E
    out[:, :, 0:h, 2*w:3*w] = N
    out[:, :, h:2*h, 0:w] = Bm
    out[:, :, h:2*h, w:2*w] = T
    out[:, :, h:2*h, 2*w:3*w] = Wf
    return out


def _cubemap_pad_with_adjoint(O, L, R, U, D, pL, pR, pU, pD, pad_mode='replicate',
                             seam_strength=0.0, seam_width=0):
    """
    Pads a face O with neighbor strips L/R/U/D (already extracted from adjacent faces).
    Supports optional seam blending (Engine B) by mixing neighbor padding with O edge.
    """
    B, C, h, w = O.shape
    Hp = h + pU + pD
    Wp = w + pL + pR
    Z = torch.zeros((B, C, Hp, Wp), device=O.device, dtype=O.dtype)
    Z[:, :, pU:pU + h, pL:pL + w] = O

    if pL == 0 and pR == 0 and pU == 0 and pD == 0:
        return Z

    # Helper: create ramp for seam_width (0 at boundary, 1 at outer pad)
    def _make_ramp(n, seam_w, device, dtype):
        if n <= 0:
            return None
        seam_w = int(max(0, min(seam_w, n)))
        if seam_w == 0:
            return torch.ones((n,), device=device, dtype=dtype)
        if seam_w == 1:
            ramp = torch.ones((n,), device=device, dtype=dtype)
            ramp[0] = 0.0
            return ramp
        ramp = torch.ones((n,), device=device, dtype=dtype)
        ramp[:seam_w] = torch.linspace(0.0, 1.0, steps=seam_w, device=device, dtype=dtype)
        return ramp

    # Fill left/right strips
    if pL > 0:
        Lp = _safe_pad4d(L, (0, 0, pU, pD), mode=pad_mode)
        strip = Lp
        if seam_strength > 0.0:
            Oedge = O[:, :, :, :min(pL, w)]
            Oedge = _safe_pad4d(Oedge, (0, max(0, pL - Oedge.shape[-1]), pU, pD), mode='replicate')
            ramp = _make_ramp(pL, seam_width, O.device, O.dtype).view(1, 1, 1, pL)
            blend_scheme = Oedge * (1.0 - ramp) + strip * ramp
            strip = strip * (1.0 - seam_strength) + blend_scheme * seam_strength
        Z[:, :, :, :pL] = strip

    if pR > 0:
        Rp = _safe_pad4d(R, (0, 0, pU, pD), mode=pad_mode)
        strip = Rp
        if seam_strength > 0.0:
            Oedge = O[:, :, :, max(0, w - pR):w]
            need = pR - Oedge.shape[-1]
            Oedge = _safe_pad4d(Oedge, (max(0, need), 0, pU, pD), mode='replicate')
            ramp = _make_ramp(pR, seam_width, O.device, O.dtype).view(1, 1, 1, pR).flip(-1)
            blend_scheme = Oedge * (1.0 - ramp) + strip * ramp
            strip = strip * (1.0 - seam_strength) + blend_scheme * seam_strength
        Z[:, :, :, -pR:] = strip

    # Fill top/bottom strips
    if pU > 0:
        Up = _safe_pad4d(U, (pL, pR, 0, 0), mode=pad_mode)
        strip = Up
        if seam_strength > 0.0:
            Oedge = O[:, :, :min(pU, h), :]
            Oedge = _safe_pad4d(Oedge, (pL, pR, 0, max(0, pU - Oedge.shape[-2])), mode='replicate')
            ramp = _make_ramp(pU, seam_width, O.device, O.dtype).view(1, 1, pU, 1)
            blend_scheme = Oedge * (1.0 - ramp) + strip * ramp
            strip = strip * (1.0 - seam_strength) + blend_scheme * seam_strength
        Z[:, :, :pU, :] = strip

    if pD > 0:
        Dp = _safe_pad4d(D, (pL, pR, 0, 0), mode=pad_mode)
        strip = Dp
        if seam_strength > 0.0:
            Oedge = O[:, :, max(0, h - pD):h, :]
            need = pD - Oedge.shape[-2]
            Oedge = _safe_pad4d(Oedge, (pL, pR, max(0, need), 0), mode='replicate')
            ramp = _make_ramp(pD, seam_width, O.device, O.dtype).view(1, 1, pD, 1).flip(-2)
            blend_scheme = Oedge * (1.0 - ramp) + strip * ramp
            strip = strip * (1.0 - seam_strength) + blend_scheme * seam_strength
        Z[:, :, -pD:, :] = strip

    # Fix corners overlapping (same as cubemap(3).py logic)
    if pU and pL:
        Z[:, :, :pU, :pL] /= 2
    if pU and pR:
        Z[:, :, :pU, -pR:] /= 2
    if pD and pL:
        Z[:, :, -pD:, :pL] /= 2
    if pD and pR:
        Z[:, :, -pD:, -pR:] /= 2

    return Z


def conv2d_cubemap_batched(input_tensor, weight, bias, stride, dilation, groups,
                          pad_h, pad_w, pad_mode='replicate',
                          engine='A (Fast)', seam_width=0, seam_strength=0.0):
    """
    Cubemap convolution for a 3x2 cubemap net (S/E/N over B/T/W), using 1 conv call:
      - Engine A: neighbor padding (fast, like cubemap(3).py but batched)
      - Engine B: same, but with seam-aware blending inside padding regions.
    NOTE: Requires square faces (h == w) to keep rotations consistent.
    """
    if pad_h != pad_w:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    if pad_h == 0 and pad_w == 0:
        return F.conv2d(input_tensor, weight, bias, stride, (0, 0), dilation, groups)

    try:
        S, E, N, Bm, T, Wf = _cubemap_split_faces(input_tensor)
    except Exception:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    B, C, h, w = S.shape
    if h != w:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    p = int(pad_h)
    pL = pR = pU = pD = p

    seam_strength = float(max(0.0, min(seam_strength, 1.0))) if (engine or '').startswith('B') else 0.0
    seam_width = int(max(0, seam_width))

    ZS = _cubemap_pad_with_adjoint(
        S,
        L=Wf[:, :, :, -pL:],
        R=E[:, :, :, :pR],
        U=T[:, :, -pU:, :],
        D=Bm[:, :, :pD, :],
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    ZE = _cubemap_pad_with_adjoint(
        E,
        L=S[:, :, :, -pL:],
        R=N[:, :, :, :pR],
        U=torch.rot90(T[:, :, :, -pU:], k=-1, dims=[2, 3]),
        D=torch.rot90(Bm[:, :, :, -pD:], k=+1, dims=[2, 3]),
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    ZN = _cubemap_pad_with_adjoint(
        N,
        L=E[:, :, :, -pL:],
        R=Wf[:, :, :, :pR],
        U=T[:, :, :pU, :].flip(-1),
        D=Bm[:, :, -pD:, :].flip(-1),
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    ZB = _cubemap_pad_with_adjoint(
        Bm,
        L=torch.rot90(Wf[:, :, -pL:, :], k=+1, dims=[2, 3]),
        R=torch.rot90(E[:, :, -pR:, :], k=-1, dims=[2, 3]),
        U=S[:, :, -pU:, :],
        D=N[:, :, -pD:, :].flip(-1),
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    ZT = _cubemap_pad_with_adjoint(
        T,
        L=torch.rot90(Wf[:, :, :pL, :], k=-1, dims=[2, 3]),
        R=torch.rot90(E[:, :, :pR, :], k=+1, dims=[2, 3]),
        U=N[:, :, :pU, :].flip(-1),
        D=S[:, :, :pD, :],
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    ZW = _cubemap_pad_with_adjoint(
        Wf,
        L=N[:, :, :, -pL:],
        R=S[:, :, :, :pR],
        U=torch.rot90(T[:, :, :, :pL], k=+1, dims=[2, 3]),
        D=torch.rot90(Bm[:, :, :, :pD], k=-1, dims=[2, 3]),
        pL=pL, pR=pR, pU=pU, pD=pD,
        pad_mode=pad_mode,
        seam_strength=seam_strength,
        seam_width=seam_width
    )

    Z = torch.cat([ZS, ZE, ZN, ZB, ZT, ZW], dim=0)
    Y = F.conv2d(Z, weight, bias, stride, (0, 0), dilation, groups)
    YS, YE, YN, YB, YT, YW = Y.chunk(6, dim=0)

    return _cubemap_stitch_faces(YS, YE, YN, YB, YT, YW)

# ===================================
# CUBEMAP (3D) — Engine C (GridSample / True 3D mapping)
# ===================================

def _ypr_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float, device, dtype):
    """
    Builds a rotation matrix from yaw/pitch/roll angles (degrees).
    Convention:
      - yaw   around +Y axis
      - pitch around +X axis
      - roll  around +Z axis
    Applied as: R = Rz(roll) @ Rx(pitch) @ Ry(yaw)
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Ry (yaw)
    Ry = torch.tensor([[cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], device=device, dtype=dtype)

    # Rx (pitch)
    Rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, cp, -sp],
                       [0.0, sp, cp]], device=device, dtype=dtype)

    # Rz (roll)
    Rz = torch.tensor([[cr, -sr, 0.0],
                       [sr,  cr, 0.0],
                       [0.0, 0.0, 1.0]], device=device, dtype=dtype)

    return (Rz @ Rx @ Ry)


def _cubemap_dirs_from_face_uv(face_id: int, u, v):
    """
    Maps face-local (u,v) to 3D direction vectors BEFORE normalization.
    Faces in our atlas mapping:
      0: Front  (+Z)  -> S
      1: Right  (+X)  -> E
      2: Back   (-Z)  -> N
      3: Bottom (-Y)  -> Bm
      4: Top    (+Y)  -> T
      5: Left   (-X)  -> Wf
    u, v are broadcastable tensors, typically shaped (Hp, Wp) or (1,1,Hp,Wp)
    """
    if face_id == 0:      # +Z (Front)
        x, y, z = u, -v, torch.ones_like(u)
    elif face_id == 1:    # +X (Right)
        x, y, z = torch.ones_like(u), -v, -u
    elif face_id == 2:    # -Z (Back)
        x, y, z = -u, -v, -torch.ones_like(u)
    elif face_id == 3:    # -Y (Bottom)
        x, y, z = u, -torch.ones_like(u), -v
    elif face_id == 4:    # +Y (Top)
        x, y, z = u, torch.ones_like(u), v
    elif face_id == 5:    # -X (Left)
        x, y, z = -torch.ones_like(u), -v, u
    else:
        raise ValueError("Invalid face_id for cubemap.")
    return x, y, z


def _cubemap_dir_to_atlas_grid(x, y, z, face_h: int, face_w: int, device, dtype):
    """
    Converts 3D direction vectors to a single atlas (3x2 net) sampling grid in [-1,1].
    Returns grid shaped (..., 2) with last dim [x_norm, y_norm].
    """
    # Normalize directions (avoid divide-by-zero)
    eps = torch.tensor(1e-8, device=device, dtype=dtype)
    inv_len = torch.rsqrt(torch.clamp(x * x + y * y + z * z, min=eps))
    x = x * inv_len
    y = y * inv_len
    z = z * inv_len

    ax = x.abs()
    ay = y.abs()
    az = z.abs()

    # Major axis selection
    is_x = (ax >= ay) & (ax >= az)
    is_y = (ay >= ax) & (ay >= az)
    is_z = ~(is_x | is_y)

    # Face index map: 0..5
    face_idx = torch.empty_like(x, dtype=torch.int64)

    # Defaults (placeholders)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    # +X / -X
    mask = is_x & (x >= 0)
    face_idx[mask] = 1
    u[mask] = -z[mask] / (ax[mask] + eps)
    v[mask] = -y[mask] / (ax[mask] + eps)

    mask = is_x & (x < 0)
    face_idx[mask] = 5
    u[mask] = z[mask] / (ax[mask] + eps)
    v[mask] = -y[mask] / (ax[mask] + eps)

    # +Y / -Y
    mask = is_y & (y >= 0)
    face_idx[mask] = 4
    u[mask] = x[mask] / (ay[mask] + eps)
    v[mask] = z[mask] / (ay[mask] + eps)

    mask = is_y & (y < 0)
    face_idx[mask] = 3
    u[mask] = x[mask] / (ay[mask] + eps)
    v[mask] = -z[mask] / (ay[mask] + eps)

    # +Z / -Z
    mask = is_z & (z >= 0)
    face_idx[mask] = 0
    u[mask] = x[mask] / (az[mask] + eps)
    v[mask] = -y[mask] / (az[mask] + eps)

    mask = is_z & (z < 0)
    face_idx[mask] = 2
    u[mask] = -x[mask] / (az[mask] + eps)
    v[mask] = -y[mask] / (az[mask] + eps)

    # Atlas tile offsets (col,row) for each face_idx
    # 0:F -> (0,0), 1:R -> (1,0), 2:B -> (2,0), 3:Bo -> (0,1), 4:T -> (1,1), 5:L -> (2,1)
    col = torch.zeros_like(u)
    row = torch.zeros_like(v)

    col = torch.where(face_idx == 0, torch.tensor(0.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 0, torch.tensor(0.0, device=device, dtype=dtype), row)

    col = torch.where(face_idx == 1, torch.tensor(1.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 1, torch.tensor(0.0, device=device, dtype=dtype), row)

    col = torch.where(face_idx == 2, torch.tensor(2.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 2, torch.tensor(0.0, device=device, dtype=dtype), row)

    col = torch.where(face_idx == 3, torch.tensor(0.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 3, torch.tensor(1.0, device=device, dtype=dtype), row)

    col = torch.where(face_idx == 4, torch.tensor(1.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 4, torch.tensor(1.0, device=device, dtype=dtype), row)

    col = torch.where(face_idx == 5, torch.tensor(2.0, device=device, dtype=dtype), col)
    row = torch.where(face_idx == 5, torch.tensor(1.0, device=device, dtype=dtype), row)

    # Convert (u,v) [-1,1] -> atlas pixel coords -> normalized coords [-1,1]
    H_atlas = int(face_h * 2)
    W_atlas = int(face_w * 3)

    # align_corners=True mapping uses (W-1)/(H-1)
    x_pix = col * face_w + (u + 1.0) * 0.5 * (face_w - 1)
    y_pix = row * face_h + (v + 1.0) * 0.5 * (face_h - 1)

    x_norm = (x_pix / max(W_atlas - 1, 1)) * 2.0 - 1.0
    y_norm = (y_pix / max(H_atlas - 1, 1)) * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)
    return grid


def _build_cubemap_engine_c_grids(face_h: int, face_w: int, pad: int,
                                  yaw: float, pitch: float, roll: float,
                                  coord_mode: str = "Cartesian (Face UV)",
                                  twist_deg: float = 0.0,
                                  polar_scale: float = 1.0,
                                  polar_power: float = 1.0,
                                  swirl_deg: float = 0.0,
                                  swirl_power: float = 1.0,
                                  device=None, dtype=None,
                                  antipode: bool = False,
                                  angle_quant: float = 0.5):
    """
    Builds and caches per-face sampling grids (Engine C) for cubemap atlas.
    Grids map each pixel in a padded face to the correct location in the 3x2 atlas.
    """
    if face_h <= 1 or face_w <= 1:
        return None

    # Quantize angles to stabilize caching
    q = float(angle_quant)
    q_milli = int(round(float(q) * 1000.0))
    if q_milli <= 0: q_milli = 1
    yaw_t = int(round(float(yaw) / q))
    pitch_t = int(round(float(pitch) / q))
    roll_t = int(round(float(roll) / q))
    twist_t = int(round(float(twist_deg) / q))
    swirl_t = int(round(float(swirl_deg) / q))
    yaw_q = float(yaw_t) * q
    pitch_q = float(pitch_t) * q
    roll_q = float(roll_t) * q
    twist_q = float(twist_t) * q
    swirl_q = float(swirl_t) * q

    # Quantize continuous params a bit for caching
    polar_scale_q = round(float(polar_scale) * 100.0) / 100.0
    polar_power_q = round(float(polar_power) * 100.0) / 100.0
    swirl_power_q = round(float(swirl_power) * 100.0) / 100.0

    dev_type = getattr(device, "type", None)
    dev_index = getattr(device, "index", None)
    key = (
           str(dev_type) if dev_type is not None else str(device),
           int(dev_index) if dev_index is not None else -1,
           str(dtype), int(face_h), int(face_w), int(pad),
           str(coord_mode),
           int(yaw_t), int(pitch_t), int(roll_t),
           int(twist_t),
           int(round(float(polar_scale_q) * 100.0)), int(round(float(polar_power_q) * 100.0)),
           int(swirl_t), int(round(float(swirl_power_q) * 100.0)),
           bool(antipode), int(q_milli))
    cached = _CUBEMAP_GRID_CACHE.get(key, None)
    if cached is not None:
        return cached

    p = int(max(0, pad))
    Hp = int(face_h + 2 * p)
    Wp = int(face_w + 2 * p)

    # Face-local u,v coordinate system (padded)
    # u maps across width, v maps across height. Center region uses u,v in [-1,1].
    j = torch.arange(Wp, device=device, dtype=dtype)
    i = torch.arange(Hp, device=device, dtype=dtype)
    denom_w = float(max(face_w - 1, 1))
    denom_h = float(max(face_h - 1, 1))
    u = 2.0 * ((j - p) / denom_w) - 1.0
    v = 2.0 * ((i - p) / denom_h) - 1.0

    # Broadcast to (Hp,Wp)
    u2 = u.view(1, Wp).expand(Hp, Wp)
    v2 = v.view(Hp, 1).expand(Hp, Wp)

    # Advanced UV transform (twist / polar warp / swirl)
    if coord_mode is None:
        coord_mode = "Cartesian (Face UV)"
    cm = str(coord_mode)
    twist_rad = float(twist_q) * (math.pi / 180.0)
    swirl_rad = float(swirl_q) * (math.pi / 180.0)
    do_polar = cm.startswith("Polar")
    if abs(twist_rad) > 1e-9 or abs(swirl_rad) > 1e-9 or do_polar:
        r = torch.sqrt(u2 * u2 + v2 * v2)
        # clamp radius for numerical stability in padding areas
        r_clamped = torch.clamp(r, 0.0, 2.0)
        theta = torch.atan2(v2, u2)
        theta = theta + twist_rad
        if abs(swirl_rad) > 1e-9:
            sp = float(swirl_power_q)
            theta = theta + swirl_rad * torch.pow(r_clamped, sp)
        if do_polar:
            ps = float(polar_scale_q)
            pp = float(polar_power_q)
            r2 = torch.pow(torch.clamp(r_clamped * ps, min=0.0), pp)
        else:
            r2 = r
        u2 = r2 * torch.cos(theta)
        v2 = r2 * torch.sin(theta)

    R = _ypr_rotation_matrix(yaw_q, pitch_q, roll_q, device=device, dtype=dtype)

    grids = []
    for face_id in range(6):
        x, y, z = _cubemap_dirs_from_face_uv(face_id, u2, v2)

        # Rotate directions
        # (Hp,Wp,3) @ (3,3)^T
        dirs = torch.stack([x, y, z], dim=-1)
        dirs = torch.matmul(dirs, R.transpose(0, 1))

        if antipode:
            dirs = -dirs

        grid = _cubemap_dir_to_atlas_grid(
            dirs[..., 0], dirs[..., 1], dirs[..., 2],
            face_h=face_h, face_w=face_w,
            device=device, dtype=dtype
        )
        grids.append(grid)

    grids = torch.stack(grids, dim=0)  # (6,Hp,Wp,2)
    _CUBEMAP_GRID_CACHE[key] = grids
    return grids


def _grid_sample_geoaa(atlas, grid, samples: int = 1, radius_px: float = 0.0,
                     mode: str = "bilinear", padding_mode: str = "border"):
    """
    Optional geometric AA (multi-sampling) for Engine C.
    - samples: 1..4
    - radius_px: pixel radius in atlas space (approx)
    """
    samples = int(max(1, min(int(samples), 4)))
    radius_px = float(max(0.0, radius_px))
    # sanitize grid_sample args
    if mode not in ("bilinear", "nearest"):
        mode = "bilinear"
    if padding_mode not in ("border", "reflection", "zeros"):
        padding_mode = "border"


    if samples == 1 or radius_px <= 0.0:
        return F.grid_sample(atlas, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    B, C, H, W = atlas.shape
    # normalize radius to grid space (align_corners=True => 1px == 2/(W-1))
    dx = (radius_px * 2.0) / max(W - 1, 1)
    dy = (radius_px * 2.0) / max(H - 1, 1)

    offsets = [(0.0, 0.0)]
    if samples >= 2:
        offsets.append((dx, dy))
    if samples >= 3:
        offsets.append((-dx, dy))
    if samples >= 4:
        offsets.append((dx, -dy))

    acc = None
    for ox, oy in offsets:
        g = grid.clone()
        g[..., 0] = (g[..., 0] + ox).clamp(-1.0, 1.0)
        g[..., 1] = (g[..., 1] + oy).clamp(-1.0, 1.0)
        y = F.grid_sample(atlas, g, mode=mode, padding_mode=padding_mode, align_corners=True)
        acc = y if acc is None else (acc + y)

    return acc / float(len(offsets))


def conv2d_cubemap_gridsample(input_tensor, weight, bias, stride, dilation, groups,
                             pad_h, pad_w,
                             yaw=0.0, pitch=0.0, roll=0.0,
                             coord_mode="Cartesian (Face UV)", twist_deg=0.0,
                             polar_scale=1.0, polar_power=1.0,
                             swirl_deg=0.0, swirl_power=1.0,
                             grid_interp="bilinear", grid_padding="border",
                             cache_angle_quant=0.5,
                             geoaa_samples=1, geoaa_radius_px=0.0,
                             antipode_strength=0.0):
    """
    Engine C: True 3D cubemap mapping using grid_sample.
    - Builds padded faces by sampling from the full 3x2 atlas via direction mapping.
    - Supports yaw/pitch/roll rotation of the sampling directions.
    - Optional geometric AA (multi-sampling) and Kohaku-inspired antipode mixing.
    """
    if pad_h != pad_w:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    p = int(pad_h)
    if p <= 0:
        return F.conv2d(input_tensor, weight, bias, stride, (0, 0), dilation, groups)

    B, C, H, W = input_tensor.shape
    if H % 2 != 0 or W % 3 != 0:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    face_h = H // 2
    face_w = W // 3
    if face_h != face_w:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    device = input_tensor.device
    # grid_sample expects float grid; use float32 for stability if input is fp16/bf16
    grid_dtype = torch.float32 if input_tensor.dtype in (torch.float16, torch.bfloat16) else input_tensor.dtype

    grids = _build_cubemap_engine_c_grids(face_h, face_w, p, yaw, pitch, roll,
                                          coord_mode, twist_deg, polar_scale, polar_power,
                                          swirl_deg, swirl_power,
                                          device, grid_dtype,
                                          antipode=False,
                                          angle_quant=cache_angle_quant)
    if grids is None:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    antipode_strength = float(max(0.0, min(float(antipode_strength), 1.0)))

    if antipode_strength > 0.0:
        grids_anti = _build_cubemap_engine_c_grids(face_h, face_w, p, yaw, pitch, roll,
                                          coord_mode, twist_deg, polar_scale, polar_power,
                                          swirl_deg, swirl_power,
                                          device, grid_dtype,
                                          antipode=True,
                                          angle_quant=cache_angle_quant)
    else:
        grids_anti = None

    Hp = int(face_h + 2 * p)
    Wp = int(face_w + 2 * p)

    faces_padded = []
    for face_id in range(6):
        g = grids[face_id].to(device=device)
        gB = g.unsqueeze(0).expand(B, Hp, Wp, 2).contiguous()

        # grid_sample uses input dtype, but grid dtype may differ; that's ok.
        y0 = _grid_sample_geoaa(input_tensor, gB, samples=geoaa_samples, radius_px=geoaa_radius_px, mode=grid_interp, padding_mode=grid_padding)

        if grids_anti is not None:
            ga = grids_anti[face_id].to(device=device)
            gaB = ga.unsqueeze(0).expand(B, Hp, Wp, 2).contiguous()
            y1 = _grid_sample_geoaa(input_tensor, gaB, samples=geoaa_samples, radius_px=geoaa_radius_px, mode=grid_interp, padding_mode=grid_padding)
            y0 = y0 * (1.0 - antipode_strength) + y1 * antipode_strength

        faces_padded.append(y0)

    Z = torch.cat(faces_padded, dim=0)  # (6B,C,Hp,Wp)
    Y = F.conv2d(Z, weight, bias, stride, (0, 0), dilation, groups)
    YS, YE, YN, YB, YT, YW = Y.chunk(6, dim=0)

    return _cubemap_stitch_faces(YS, YE, YN, YB, YT, YW)



# ========================================================================
# PANORAMA LIVE (Equirectangular) — Engine C (3D grid_sample)
# ========================================================================

def _get_blur_kernel_1d(radius: int, device, dtype):
    """Depthwise 1D blur kernel along X (width)."""
    r = int(max(0, radius))
    if r <= 0:
        return None
    k = 2 * r + 1
    dev_type = getattr(device, "type", None)
    dev_index = getattr(device, "index", None)
    key = (int(k), str(dev_type) if dev_type is not None else str(device), int(dev_index) if dev_index is not None else -1, str(dtype))
    ker = _BLUR_KERNEL_CACHE.get(key, None)
    if ker is not None:
        return ker
    # Simple box kernel (stable, cheap). You can swap to Gaussian if needed.
    w = torch.ones((k,), device=device, dtype=dtype) / float(k)
    ker = w.view(1, 1, 1, k)  # (1,1,1,k)
    _BLUR_KERNEL_CACHE[key] = ker
    return ker


def _apply_pole_blur_smoothing(x, strength: float = 0.0, radius: int = 0, power: float = 1.0):
    """
    Applies circular horizontal blur near poles (top/bottom) with a smooth mask.
    x: (B,C,H,W)
    """
    strength = float(max(0.0, min(float(strength), 1.0)))
    radius = int(max(0, int(radius)))
    power = float(max(0.25, min(float(power), 4.0)))

    if strength <= 0.0 or radius <= 0:
        return x

    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    ker = _get_blur_kernel_1d(radius, device, dtype)
    if ker is None:
        return x

    # Pole mask: 1 near top/bottom, 0 near equator
    yy = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype).view(1, 1, H, 1)
    t = torch.abs(yy - 0.5) * 2.0  # 0 at equator, 1 at poles
    pole_mask = torch.pow(torch.clamp(t, 0.0, 1.0), power)  # (1,1,H,1)

    # Circular pad along X then depthwise conv
    xp = F.pad(x, (radius, radius, 0, 0), mode="circular")
    # Depthwise conv: expand kernel per-channel
    weight = ker.expand(C, 1, 1, ker.shape[-1]).contiguous()
    blurred = F.conv2d(xp, weight, bias=None, stride=1, padding=0, groups=C)

    m = pole_mask * strength
    return x * (1.0 - m) + blurred * m


def _build_panorama_engine_c_grid(H: int, W: int, pad_h: int, pad_w: int,
                                  yaw: float, pitch: float, roll: float,
                                  coord_mode: str = "Cartesian (lon/lat)",
                                  polar_scale: float = 1.0,
                                  polar_power: float = 1.0,
                                  twist_deg: float = 0.0,
                                  twist_power: float = 1.0,
                                  swirl_deg: float = 0.0,
                                  swirl_power: float = 1.0,
                                  pole_ease_power: float = 1.0,
                                  antipode: bool = False,
                                  angle_quant: float = 0.5,
                                  device=None, dtype=None):
    """
    Builds/caches a sampling grid for equirectangular panoramas.
    Grid maps output pixels in a padded canvas to source coords in the original panorama.
    Uses true 3D spherical mapping (yaw/pitch/roll) and optional UV warps.
    """
    if H <= 1 or W <= 1:
        return None

    ph = int(max(0, pad_h))
    pw = int(max(0, pad_w))
    Hp = int(H + 2 * ph)
    Wp = int(W + 2 * pw)

    q = float(max(0.1, float(angle_quant)))
    q_milli = int(round(float(q) * 1000.0))
    if q_milli <= 0: q_milli = 1
    yaw_t = int(round(float(yaw) / q))
    pitch_t = int(round(float(pitch) / q))
    roll_t = int(round(float(roll) / q))
    twist_t = int(round(float(twist_deg) / q))
    swirl_t = int(round(float(swirl_deg) / q))
    yaw_q = float(yaw_t) * q
    pitch_q = float(pitch_t) * q
    roll_q = float(roll_t) * q
    twist_q = float(twist_t) * q
    swirl_q = float(swirl_t) * q

    polar_scale_q = round(float(polar_scale) * 100.0) / 100.0
    polar_power_q = round(float(polar_power) * 100.0) / 100.0
    twist_power_q = round(float(twist_power) * 100.0) / 100.0
    swirl_power_q = round(float(swirl_power) * 100.0) / 100.0
    pole_ease_q = round(float(pole_ease_power) * 100.0) / 100.0

    dev_type = getattr(device, "type", None)
    dev_index = getattr(device, "index", None)
    key = (
           str(dev_type) if dev_type is not None else str(device),
           int(dev_index) if dev_index is not None else -1,
           str(dtype), int(H), int(W), int(ph), int(pw),
           str(coord_mode),
           int(yaw_t), int(pitch_t), int(roll_t),
           int(twist_t), int(round(float(twist_power_q) * 100.0)),
           int(swirl_t), int(round(float(swirl_power_q) * 100.0)),
           int(round(float(polar_scale_q) * 100.0)), int(round(float(polar_power_q) * 100.0)),
           int(round(float(pole_ease_q) * 100.0)),
           bool(antipode), int(q_milli))
    cached = _PANO_GRID_CACHE.get(key, None)
    if cached is not None:
        return cached

    # Output pixel -> base lon/lat (can extend beyond [0,1] in padding; that's OK)
    j = torch.arange(Wp, device=device, dtype=dtype)
    i = torch.arange(Hp, device=device, dtype=dtype)

    denom_w = float(max(W - 1, 1))
    denom_h = float(max(H - 1, 1))

    u = (j - pw) / denom_w  # 0..1 over original image
    v = (i - ph) / denom_h

    u2 = u.view(1, Wp).expand(Hp, Wp)
    v2 = v.view(Hp, 1).expand(Hp, Wp)

    # lon in radians (wrap naturally via sin/cos); lat in radians (can go beyond poles)
    lon = (u2 - 0.5) * (2.0 * math.pi)
    lat = (0.5 - v2) * math.pi

    cm = str(coord_mode or "Cartesian (lon/lat)")
    do_polar = cm.startswith("Polar")

    # --- Optional twist & swirl in (lon,lat) domain ---
    # Twist is latitude-dependent longitude shift ("roll" feel along parallels)
    tr = float(twist_q) * (math.pi / 180.0)
    tp = float(max(0.25, min(float(twist_power_q), 4.0)))
    if abs(tr) > 1e-9:
        t = torch.clamp(torch.abs(lat) / (0.5 * math.pi), 0.0, 1.0)
        lon = lon + tr * torch.sign(lat) * torch.pow(t, tp)

    sr = float(swirl_q) * (math.pi / 180.0)
    sp = float(max(0.25, min(float(swirl_power_q), 4.0)))
    if abs(sr) > 1e-9:
        # Swirl strongest near poles by default
        t = torch.clamp(torch.abs(lat) / (0.5 * math.pi), 0.0, 1.0)
        lon = lon + sr * torch.pow(t, sp)

    # --- Polar mode: radial warp around poles via latitude reparameterization ---
    if do_polar:
        ps = float(max(0.01, float(polar_scale_q)))
        pp = float(max(0.25, min(float(polar_power_q), 6.0)))
        # t=0 at equator, t=1 at poles
        t = torch.clamp(torch.abs(lat) / (0.5 * math.pi), 0.0, 1.0)
        r = 1.0 - t  # r=1 at equator, 0 at poles
        r2 = torch.pow(torch.clamp(r * ps, min=0.0, max=1.0), pp)
        t2 = 1.0 - r2
        lat = torch.sign(lat) * t2 * (0.5 * math.pi)

    # Convert (lon,lat) to 3D direction
    cl = torch.cos(lon)
    sl = torch.sin(lon)
    ca = torch.cos(lat)
    sa = torch.sin(lat)

    x = sl * ca
    y = sa
    z = cl * ca

    # Apply global rotation
    R = _ypr_rotation_matrix(yaw_q, pitch_q, roll_q, device=device, dtype=dtype)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = torch.matmul(dirs, R.transpose(0, 1))

    if antipode:
        dirs = -dirs

    # Back to lon/lat
    x2 = dirs[..., 0]
    y2 = torch.clamp(dirs[..., 1], -1.0, 1.0)
    z2 = dirs[..., 2]

    lon2 = torch.atan2(x2, z2)  # [-pi,pi]
    lat2 = torch.asin(y2)       # [-pi/2,pi/2]

    # Pole easing curve (power) on latitude magnitude
    pe = float(max(0.25, min(float(pole_ease_q), 6.0)))
    if abs(pe - 1.0) > 1e-6:
        t = torch.clamp(torch.abs(lat2) / (0.5 * math.pi), 0.0, 1.0)
        t = torch.pow(t, pe)
        lat2 = torch.sign(lat2) * t * (0.5 * math.pi)

    # Convert to source UV [0,1) with X wrap
    u_src = (lon2 / (2.0 * math.pi)) + 0.5
    u_src = torch.remainder(u_src, 1.0)  # wrap horizontally
    v_src = 0.5 - (lat2 / math.pi)       # 0..1

    # to normalized grid_sample coords [-1,1]
    x_norm = u_src * 2.0 - 1.0
    y_norm = v_src * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # (Hp,Wp,2)
    _PANO_GRID_CACHE[key] = grid
    return grid


def conv2d_panorama_gridsample(input_tensor, weight, bias, stride, dilation, groups,
                               pad_h, pad_w,
                               yaw=0.0, pitch=0.0, roll=0.0,
                               coord_mode="Cartesian (lon/lat)",
                               polar_scale=1.0, polar_power=1.0,
                               twist_deg=0.0, twist_power=1.0,
                               swirl_deg=0.0, swirl_power=1.0,
                               pole_ease_power=1.0,
                               grid_interp="bilinear", grid_padding="border",
                               cache_angle_quant=0.5,
                               geoaa_samples=1, geoaa_radius_px=0.0,
                               antipode_strength=0.0,
                               pole_blur_strength=0.0, pole_blur_radius=0, pole_blur_power=1.0):
    """
    Panorama Live Engine C:
      - Builds a padded panorama by sampling the original via 3D spherical mapping.
      - Runs conv2d without extra padding.
      - Optional Kohaku-style antipode mixing and pole blur smoothing.
    """
    ph = int(max(0, int(pad_h)))
    pw = int(max(0, int(pad_w)))
    if ph <= 0 and pw <= 0:
        return F.conv2d(input_tensor, weight, bias, stride, (0, 0), dilation, groups)

    B, C, H, W = input_tensor.shape
    device = input_tensor.device
    grid_dtype = torch.float32 if input_tensor.dtype in (torch.float16, torch.bfloat16) else input_tensor.dtype

    grid = _build_panorama_engine_c_grid(
        H, W, ph, pw,
        yaw=yaw, pitch=pitch, roll=roll,
        coord_mode=coord_mode,
        polar_scale=polar_scale, polar_power=polar_power,
        twist_deg=twist_deg, twist_power=twist_power,
        swirl_deg=swirl_deg, swirl_power=swirl_power,
        pole_ease_power=pole_ease_power,
        antipode=False,
        angle_quant=cache_angle_quant,
        device=device, dtype=grid_dtype
    )
    if grid is None:
        return F.conv2d(input_tensor, weight, bias, stride, (pad_h, pad_w), dilation, groups)

    Hp = int(H + 2 * ph)
    Wp = int(W + 2 * pw)

    gB = grid.unsqueeze(0).expand(B, Hp, Wp, 2).contiguous()
    y0 = _grid_sample_geoaa(input_tensor, gB, samples=geoaa_samples, radius_px=geoaa_radius_px,
                            mode=grid_interp, padding_mode=grid_padding)

    antipode_strength = float(max(0.0, min(float(antipode_strength), 1.0)))
    if antipode_strength > 0.0:
        grid_a = _build_panorama_engine_c_grid(
            H, W, ph, pw,
            yaw=yaw, pitch=pitch, roll=roll,
            coord_mode=coord_mode,
            polar_scale=polar_scale, polar_power=polar_power,
            twist_deg=twist_deg, twist_power=twist_power,
            swirl_deg=swirl_deg, swirl_power=swirl_power,
            pole_ease_power=pole_ease_power,
            antipode=True,
            angle_quant=cache_angle_quant,
            device=device, dtype=grid_dtype
        )
        gaB = grid_a.unsqueeze(0).expand(B, Hp, Wp, 2).contiguous()
        y1 = _grid_sample_geoaa(input_tensor, gaB, samples=geoaa_samples, radius_px=geoaa_radius_px,
                                mode=grid_interp, padding_mode=grid_padding)
        y0 = y0 * (1.0 - antipode_strength) + y1 * antipode_strength

    # Optional pole blur
    y0 = _apply_pole_blur_smoothing(y0,
                                   strength=pole_blur_strength,
                                   radius=pole_blur_radius,
                                   power=pole_blur_power)

    return F.conv2d(y0, weight, bias, stride, (0, 0), dilation, groups)

def compute_stereoscopic_padding(input_tensor, pad_h, pad_w, eye='left',
                                 convergence=0.05, separation=0.065):
    """
    Стереоскопический паддинг для 3D изображений.
    eye: 'left', 'right' или 'both'
    """
    b, c, h, w = input_tensor.shape

    eye = (eye or 'left').lower()
    shift_amount = int(w * separation)

    x_coords = torch.linspace(
        0.0, 1.0, w,
        device=input_tensor.device,
        dtype=input_tensor.dtype
    ).view(1, 1, 1, w)
    depth_map = torch.abs(x_coords - convergence).expand(b, c, h, w)
    alpha = depth_map.clamp(0.0, 1.0)

    if eye == 'left':
        shifted = torch.roll(input_tensor, shifts=shift_amount, dims=3)
        stereo_adjusted = input_tensor * (1.0 - alpha) + shifted * alpha
    elif eye == 'right':
        shifted = torch.roll(input_tensor, shifts=-shift_amount, dims=3)
        stereo_adjusted = input_tensor * (1.0 - alpha) + shifted * alpha
    else:
        # 'both' — симметричный режим
        shifted_left = torch.roll(input_tensor, shifts=shift_amount, dims=3)
        shifted_right = torch.roll(input_tensor, shifts=-shift_amount, dims=3)
        shifted_avg = 0.5 * (shifted_left + shifted_right)
        stereo_adjusted = input_tensor * (1.0 - alpha) + shifted_avg * alpha

    padded = F.pad(stereo_adjusted, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    return padded

def compute_hex_padding_x(input_tensor, pad_l, pad_r):
    """Гексагональный паддинг (из v2.0)"""
    b, c, h, w = input_tensor.shape
    odd_mask = get_or_create_mask(h, w, input_tensor.device).expand(b, c, h, w)

    shift = w // 2
    input_shifted = torch.roll(input_tensor, shifts=shift, dims=3)
    source = torch.where(odd_mask, input_shifted, input_tensor)

    left_pad = source[:, :, :, -pad_l:]
    right_pad = source[:, :, :, :pad_r]

    return torch.cat([left_pad, input_tensor, right_pad], dim=3)

# ========================================================================
# ГЛАВНАЯ ФУНКЦИЯ ПАДДИНГА
# ========================================================================

def custom_padding_forward(input_tensor, weight, bias, stride, padding, dilation, groups, params):
    """
    Универсальная функция паддинга с поддержкой всех режимов v3.1
    
    ИСПРАВЛЕНО:
    - ✅ Blend mode работает правильно (input-space)
    - ✅ Multi-resolution работает правильно (output-space)
    - ✅ Правильная последовательность операций
    - ✅ Нет неиспользуемых переменных
    - ✅ Унифицированная логика для всех режимов
    """
    try:
        current_step = getattr(shared.state, 'sampling_step', 0)
        if not (params['start_step'] <= current_step <= params['end_step']):
            return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

        # Проверка "Disable Advanced Tiling during hires pass"
        script = params.get('script_ref', None)
        if params.get('tiling_disable_hr', False) and script is not None:
            if getattr(script, 'tiling_enable_hr', False) and getattr(script, 'tiling_is_hires', False):
                return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

        # Вычисляем требуемый padding
        k_h, k_w = weight.shape[2], weight.shape[3]
        d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        if isinstance(padding, str):
            if padding == 'same':
                req_pad_h = ((k_h - 1) * d_h) // 2
                req_pad_w = ((k_w - 1) * d_w) // 2
            elif padding == 'valid':
                req_pad_h = req_pad_w = 0
            else:
                req_pad_h = req_pad_w = 1
        elif isinstance(padding, int):
            req_pad_h = req_pad_w = padding
        elif isinstance(padding, (tuple, list)):
            req_pad_h, req_pad_w = padding[0], padding[1]
        else:
            req_pad_h = req_pad_w = 0

        if req_pad_h == 0 and req_pad_w == 0:
            return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

        mode_x = params.get('mode_x', MODE_OFF)
        mode_y = params.get('mode_y', MODE_OFF)

        # ====== НОВОЕ: ПРОВЕРКА ZOOM SYSTEM ======
        if params.get('use_zoom', False):
            try:
                # Импорт внутри функции, чтобы избежать циклических ссылок
                from advanced_zoom_extension import apply_unified_zoom, validate_zoom_params

                # Сбор всех параметров (V2.0 + V3.0)
                raw_zoom_params = {
                    'zoom_factor': params.get('zoom_factor', 0.0),
                    'zoom_mode': params.get('zoom_mode', 'outpaint_zoom'),
                    'blend_mode': params.get('blend_mode_type', 'circular_reflect'),
                    'convergence_point': params.get('zoom_convergence', 0.5),
                    'convergence_y': params.get('convergence_y', 0.5),
                    'depth_power': params.get('zoom_depth_power', 1.0),
                    
                    # 🔧 ИЗМЕНЕНИЕ 7: Используем новые параметры панорамирования Zoom
                    'x_pan': params.get('zoom_pan_x', 0.0),  # Было x_pan
                    'y_pan': params.get('zoom_pan_y', 0.0),  # Было y_pan
                    
                    # V3.0 параметры
                    'noise_strength': params.get('noise_strength', 1.0),
                    'interp_mode': params.get('interp_mode', 'bilinear'),
                    'zoom_in_fade': params.get('zoom_in_fade', True),
                    'variance_correction': params.get('variance_correction', True),
                    'auto_clamp_pan': params.get('auto_clamp_pan', True),
                    'adaptive_noise_scale': params.get('adaptive_noise_scale', True),
                    'debug_mode': params.get('debug_mode', False),
                    
                    # Legacy
                    'blend_falloff': params.get('blend_falloff', 'smoothstep'),
                    'blend_sharpness': params.get('blend_sharpness', 1.0),
                    'blend_width': params.get('blend_width', None),
                    'zoom_fade_to_black': params.get('zoom_fade_to_black', False),
                    'zoom_fade_strength': params.get('zoom_fade_strength', 0.15),
                }

                # Валидация
                zoom_params = validate_zoom_params(raw_zoom_params)
                
                # 🔧 ДОБАВЛЕНИЕ ДЛЯ SPIRAL ZOOM:
                # Передаем параметры спирали через extra_params или напрямую в словарь
                zoom_params['extra_params'] = {
                    'spiral_rotation': float(params.get('spiral_rotation', 0.0)),
                    'spiral_direction': 1.0 if params.get('spiral_direction', 0) == 0 else -1.0,
                }
                
                # Доп. параметры для специфических режимов
                extra_params = {
                    'compute_polar_padding': compute_polar_padding if 'compute_polar_padding' in globals() else None,
                    'compute_anisotropic_padding': compute_anisotropic_padding if 'compute_anisotropic_padding' in globals() else None,
                    'aniso_angle': params.get('anisotropic_angle', 45),
                    'aniso_angle2': params.get('anisotropic_angle2', None),
                    'aniso_mix': params.get('anisotropic_angle_mix', 1.0),
                    
                    # --- ИСПРАВЛЕНИЕ: Теперь параметры спирали ТОЧНО попадут в функцию ---
                    'spiral_rotation': float(params.get('spiral_rotation', 0.0)),
                    'spiral_direction': 1.0 if params.get('spiral_direction', 0) == 0 else -1.0,
                    
                    # --- ИСПРАВЛЕНИЕ: Передаем флаг Safety Mode ---
                    'force_original_size': params.get('zoom_force_original_size', True)
                }
                
                # Применение Unified Zoom
                x = apply_unified_zoom(
                    input_tensor,
                    pad_h=req_pad_h,
                    pad_w=req_pad_w,
                    
                    # Распаковка всех валидированных параметров
                    zoom_factor=zoom_params['zoom_factor'],
                    zoom_mode=zoom_params['zoom_mode'],
                    blend_mode=zoom_params['blend_mode'],
                    convergence_point=zoom_params['convergence_point'],
                    convergence_y=zoom_params['convergence_y'],
                    depth_power=zoom_params['depth_power'],
                    pan_x=zoom_params['pan_x'],
                    pan_y=zoom_params['pan_y'],
                    
                    noise_strength=zoom_params['noise_strength'],
                    interp_mode=zoom_params['interp_mode'],
                    zoom_in_fade=zoom_params['zoom_in_fade'],
                    variance_correction=zoom_params['variance_correction'],
                    auto_clamp_pan=zoom_params['auto_clamp_pan'],
                    adaptive_noise_scale=zoom_params['adaptive_noise_scale'],
                    debug=zoom_params['debug'],
                    
                    fade_to_black=zoom_params['fade_to_black'],
                    fade_strength=zoom_params['fade_strength'],
                    
                    extra_params=extra_params # <--- Теперь тут есть всё что нужно!
                )
                
                # Возврат с нулевым паддингом (так как зум уже сделал паддинг)
                return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
                
            except Exception as e:
                print(f"⚠ Zoom Error: {e}")
                # Fallback к обычному поведению при ошибке
                return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

        # ═══════════════════════════════════════════════════════════════════
        # СПЕЦИАЛЬНЫЕ РЕЖИМЫ (Cubemap, Panorama) - используют свои функции
        # ═══════════════════════════════════════════════════════════════════
        
        # ────────── CUBEMAP ──────────
        if mode_x == MODE_CUBEMAP or mode_y == MODE_CUBEMAP:
            cubemap_engine = params.get('cubemap_engine', 'A (Fast)')
            cubemap_pad_mode = params.get('cubemap_pad_mode', 'replicate')
            seam_w = int(params.get('cubemap_blend_width', 0) or 0)
            seam_s = float(params.get('cubemap_blend_strength', 0.0) or 0.0)

            if str(cubemap_engine).startswith('C'):
                # Engine C: grid_sample cubemap
                yaw = float(params.get('cubemap_yaw', 0.0) or 0.0)
                pitch = float(params.get('cubemap_pitch', 0.0) or 0.0)
                roll = float(params.get('cubemap_roll', 0.0) or 0.0)
                coord_mode = params.get('cubemap_coord_mode', 'Cartesian (Face UV)')
                twist_deg = float(params.get('cubemap_twist_deg', 0.0) or 0.0)
                polar_scale = float(params.get('cubemap_polar_scale', 1.0) or 1.0)
                polar_power = float(params.get('cubemap_polar_power', 1.0) or 1.0)
                swirl_deg = float(params.get('cubemap_swirl_deg', 0.0) or 0.0)
                swirl_power = float(params.get('cubemap_swirl_power', 1.0) or 1.0)
                grid_interp = params.get('cubemap_grid_interp', 'bilinear')
                grid_padding = params.get('cubemap_grid_padding', 'border')
                cache_angle_quant = float(params.get('cubemap_cache_angle_quant', 0.5) or 0.5)
                geoaa_samples = int(params.get('cubemap_geoaa_samples', 1) or 1)
                geoaa_radius = float(params.get('cubemap_geoaa_radius_px', 0.0) or 0.0)
                antipode_strength = float(params.get('cubemap_antipode_strength', 0.0) or 0.0)

                out_cube = conv2d_cubemap_gridsample(
                    input_tensor, weight, bias,
                    stride, dilation, groups,
                    req_pad_h, req_pad_w,
                    yaw=yaw, pitch=pitch, roll=roll,
                    coord_mode=coord_mode, twist_deg=twist_deg,
                    polar_scale=polar_scale, polar_power=polar_power,
                    swirl_deg=swirl_deg, swirl_power=swirl_power,
                    grid_interp=grid_interp, grid_padding=grid_padding,
                    cache_angle_quant=cache_angle_quant,
                    geoaa_samples=geoaa_samples, geoaa_radius_px=geoaa_radius,
                    antipode_strength=antipode_strength
                )
            else:
                # Engine A/B: batched cubemap
                out_cube = conv2d_cubemap_batched(
                    input_tensor, weight, bias,
                    stride, dilation, groups,
                    req_pad_h, req_pad_w,
                    pad_mode=cubemap_pad_mode,
                    engine=cubemap_engine,
                    seam_width=seam_w,
                    seam_strength=seam_s
                )

            # ✅ ИСПРАВЛЕНО: Multi-resolution для Cubemap (output-space)
            if params.get('multires_enabled', False):
                multires_params = validate_multires_params(params)
                
                # Создаем простую версию (output space)
                x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                                  mode='replicate')
                out_default = F.conv2d(x_default, weight, bias, stride, 0, dilation, groups)
                
                # Применяем смешивание к OUTPUTS
                out_cube = apply_multires_blend(
                    tensor_simple=out_default,
                    tensor_advanced=out_cube,  # ← ИСПРАВЛЕНО: используем out_cube!
                    current_step=current_step,
                    start_step=params['start_step'],
                    end_step=params['end_step'],
                    strategy=multires_params['strategy'],
                    transition_start=multires_params['transition_start'],
                    transition_end=multires_params['transition_end'],
                    sharpness=multires_params['sharpness'],
                    enabled=True
                )

            return out_cube
        
        # ────────── PANORAMA ──────────
        elif mode_x == MODE_PANORAMA or mode_y == MODE_PANORAMA:
            pano_engine = params.get('panorama_engine', 'A (Legacy)')
            pano_engine_str = str(pano_engine)

            # Engine C: true spherical mapping via grid_sample
            if pano_engine_str.startswith('C'):
                yaw = float(params.get('panorama_yaw', 0.0) or 0.0)
                pitch = float(params.get('panorama_pitch', 0.0) or 0.0)
                roll = float(params.get('panorama_roll', 0.0) or 0.0)
                coord_mode = params.get('panorama_coord_mode', 'Cartesian (lon/lat)')
                polar_scale = float(params.get('panorama_polar_scale', 1.0) or 1.0)
                polar_power = float(params.get('panorama_polar_power', 1.0) or 1.0)
                twist_deg = float(params.get('panorama_twist_deg', 0.0) or 0.0)
                twist_power = float(params.get('panorama_twist_power', 1.0) or 1.0)
                swirl_deg = float(params.get('panorama_swirl_deg', 0.0) or 0.0)
                swirl_power = float(params.get('panorama_swirl_power', 1.0) or 1.0)
                pole_ease_power = float(params.get('panorama_pole_ease_power', 1.0) or 1.0)
                grid_interp = params.get('panorama_grid_interp', 'bilinear')
                grid_padding = params.get('panorama_grid_padding', 'border')
                cache_angle_quant = float(params.get('panorama_cache_angle_quant', 0.5) or 0.5)
                geoaa_samples = int(params.get('panorama_geoaa_samples', 1) or 1)
                geoaa_radius = float(params.get('panorama_geoaa_radius_px', 0.0) or 0.0)
                antipode_strength = float(params.get('panorama_antipode_strength', 0.0) or 0.0)
                pole_blur_strength = float(params.get('panorama_pole_blur_strength', 0.0) or 0.0)
                pole_blur_radius = int(params.get('panorama_pole_blur_radius', 0) or 0)
                pole_blur_power = float(params.get('panorama_pole_blur_power', 1.0) or 1.0)

                out_pano = conv2d_panorama_gridsample(
                    input_tensor, weight, bias,
                    stride, dilation, groups,
                    req_pad_h, req_pad_w,
                    yaw=yaw, pitch=pitch, roll=roll,
                    coord_mode=coord_mode,
                    polar_scale=polar_scale, polar_power=polar_power,
                    twist_deg=twist_deg, twist_power=twist_power,
                    swirl_deg=swirl_deg, swirl_power=swirl_power,
                    pole_ease_power=pole_ease_power,
                    grid_interp=grid_interp, grid_padding=grid_padding,
                    cache_angle_quant=cache_angle_quant,
                    geoaa_samples=geoaa_samples, geoaa_radius_px=geoaa_radius,
                    antipode_strength=antipode_strength,
                    pole_blur_strength=pole_blur_strength, pole_blur_radius=pole_blur_radius,
                    pole_blur_power=pole_blur_power
                )

                # ✅ УЖЕ ПРАВИЛЬНО: Multi-resolution для Panorama C (output-space)
                if params.get('multires_enabled', False):
                    multires_params = validate_multires_params(params)
                    
                    x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                                      mode='replicate')
                    out_default = F.conv2d(x_default, weight, bias, stride, 0, dilation, groups)
                    
                    out_pano = apply_multires_blend(
                        tensor_simple=out_default,
                        tensor_advanced=out_pano,
                        current_step=current_step,
                        start_step=params['start_step'],
                        end_step=params['end_step'],
                        strategy=multires_params['strategy'],
                        transition_start=multires_params['transition_start'],
                        transition_end=multires_params['transition_end'],
                        sharpness=multires_params['sharpness'],
                        enabled=True
                    )

                return out_pano

            # Engine B: fast pole-correct padding
            elif pano_engine_str.startswith('B'):
                x = input_tensor
                anti_s = float(params.get('panorama_antipode_strength', 0.0) or 0.0)
                anti_s = float(max(0.0, min(anti_s, 1.0)))
                if anti_s > 0.0:
                    anti = torch.roll(torch.flip(x, dims=[2]), shifts=x.shape[-1] // 2, dims=3)
                    x = x * (1.0 - anti_s) + anti * anti_s

                x = compute_polar_padding(x, req_pad_h, req_pad_w)

                pole_blur_strength = float(params.get('panorama_pole_blur_strength', 0.0) or 0.0)
                pole_blur_radius = int(params.get('panorama_pole_blur_radius', 0) or 0)
                pole_blur_power = float(params.get('panorama_pole_blur_power', 1.0) or 1.0)
                x = _apply_pole_blur_smoothing(x, strength=pole_blur_strength, 
                                               radius=pole_blur_radius, power=pole_blur_power)
                
                # ✅ ИСПРАВЛЕНО: Применяем multires правильно
                out_pano = F.conv2d(x, weight, bias, stride, 0, dilation, groups)
                
                if params.get('multires_enabled', False):
                    multires_params = validate_multires_params(params)
                    
                    x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                                      mode='replicate')
                    out_default = F.conv2d(x_default, weight, bias, stride, 0, dilation, groups)
                    
                    out_pano = apply_multires_blend(
                        tensor_simple=out_default,
                        tensor_advanced=out_pano,
                        current_step=current_step,
                        start_step=params['start_step'],
                        end_step=params['end_step'],
                        strategy=multires_params['strategy'],
                        transition_start=multires_params['transition_start'],
                        transition_end=multires_params['transition_end'],
                        sharpness=multires_params['sharpness'],
                        enabled=True
                    )
                
                return out_pano

            # Engine A (legacy): simple circular wrap
            else:
                x = input_tensor
                x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='circular')
                x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')
                
                out_pano = F.conv2d(x, weight, bias, stride, 0, dilation, groups)
                
                if params.get('multires_enabled', False):
                    multires_params = validate_multires_params(params)
                    
                    x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                                      mode='replicate')
                    out_default = F.conv2d(x_default, weight, bias, stride, 0, dilation, groups)
                    
                    out_pano = apply_multires_blend(
                        tensor_simple=out_default,
                        tensor_advanced=out_pano,
                        current_step=current_step,
                        start_step=params['start_step'],
                        end_step=params['end_step'],
                        strategy=multires_params['strategy'],
                        transition_start=multires_params['transition_start'],
                        transition_end=multires_params['transition_end'],
                        sharpness=multires_params['sharpness'],
                        enabled=True
                    )
                
                return out_pano

        # ═══════════════════════════════════════════════════════════════════
        # ОБЫЧНЫЕ РЕЖИМЫ (Circular, Mirror, Hexagonal, Voronoi, и т.д.)
        # ═══════════════════════════════════════════════════════════════════
        
        x = input_tensor
        
        # ────────── NEW ENHANCED MODES ──────────
        if mode_x == MODE_VORONOI or mode_y == MODE_VORONOI:
            voronoi_cells = int(params.get('voronoi_cells', 8) or 8)
            voronoi_seed = int(params.get('voronoi_seed', 42) or 42)
            x = compute_voronoi_padding(x, req_pad_h, req_pad_w, voronoi_cells, voronoi_seed)
        
        elif mode_x == MODE_PERLIN or mode_y == MODE_PERLIN:
            perlin_strength = float(params.get('perlin_strength', 0.3) or 0.3)
            perlin_scale = float(params.get('perlin_scale', 10.0) or 10.0)
            x = compute_perlin_padding(x, req_pad_h, req_pad_w, perlin_strength, perlin_scale)
        
        elif mode_x == MODE_FRACTAL or mode_y == MODE_FRACTAL:
            fractal_iterations = int(params.get('fractal_iterations', 2) or 2)
            fractal_scale = float(params.get('fractal_scale', 0.6) or 0.6)
            x = compute_fractal_padding(x, req_pad_h, req_pad_w, fractal_iterations, fractal_scale)
        
        elif mode_x == MODE_ADAPTIVE or mode_y == MODE_ADAPTIVE:
            adaptive_threshold = float(params.get('adaptive_threshold', 0.1) or 0.1)
            x = compute_adaptive_padding(x, req_pad_h, req_pad_w, adaptive_threshold)
        
        # ────────── STANDARD MODES (раздельно по осям) ──────────
        else:
            # ✅ ИСПРАВЛЕНО: Blend mode применяется ВМЕСТО обычного padding
            blend_enabled = params.get('blend_enabled', False)
            
            if blend_enabled and (req_pad_h > 0 or req_pad_w > 0):
                # Blend mode ЗАМЕНЯЕТ обычный padding
                blend_params = validate_blend_params(params)
                
                # Определяем какой advanced режим использовать
                if mode_x == MODE_CIRCULAR or mode_y == MODE_CIRCULAR:
                    mode_advanced_str = 'circular'
                elif mode_x == MODE_MIRROR or mode_y == MODE_MIRROR:
                    mode_advanced_str = 'reflect'
                else:
                    mode_advanced_str = 'circular'  # fallback
                
                # Применяем улучшенный blend mode
                x = compute_advanced_blend_padding(
                    input_tensor,  # ← Исходный тензор БЕЗ padding
                    pad_h=req_pad_h,
                    pad_w=req_pad_w,
                    mode_simple='replicate',
                    mode_advanced=mode_advanced_str,
                    blend_strength=blend_params['strength'],
                    blend_width=blend_params['width'],
                    falloff_curve=blend_params['falloff'],
                    edge_sharpness=blend_params['sharpness']
                )
            
            else:
                # Обычный padding без blend
                # Ось Y
                if mode_y == MODE_CIRCULAR:
                    x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')
                elif mode_y == MODE_MIRROR:
                    x = _safe_pad4d(x, (0, 0, req_pad_h, req_pad_h), mode='reflect')
                elif mode_y == MODE_HEXAGONAL:
                    x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')
                else:
                    x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='constant', value=0)

                # Ось X
                if mode_x == MODE_CIRCULAR:
                    x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='circular')
                elif mode_x == MODE_MIRROR:
                    x = _safe_pad4d(x, (req_pad_w, req_pad_w, 0, 0), mode='reflect')
                elif mode_x == MODE_HEXAGONAL:
                    x = compute_hex_padding_x(x, req_pad_w, req_pad_w)
                else:
                    x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='constant', value=0)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONVOLUTION + MULTI-RESOLUTION (унифицированно для всех режимов)
        # ═══════════════════════════════════════════════════════════════════
        
        # x теперь содержит padded тензор (с blend или без)
        out_advanced = F.conv2d(x, weight, bias, stride, 0, dilation, groups)
        
        # ✅ ИСПРАВЛЕНО: Multi-resolution для обычных режимов
        if params.get('multires_enabled', False):
            multires_params = validate_multires_params(params)
            
            # Создаем простую версию (output space)
            x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                              mode='replicate')
            out_default = F.conv2d(x_default, weight, bias, stride, 0, dilation, groups)
            
            # Смешиваем outputs
            out_final = apply_multires_blend(
                tensor_simple=out_default,
                tensor_advanced=out_advanced,  # ← ИСПРАВЛЕНО: правильная переменная!
                current_step=current_step,
                start_step=params['start_step'],
                end_step=params['end_step'],
                strategy=multires_params['strategy'],
                transition_start=multires_params['transition_start'],
                transition_end=multires_params['transition_end'],
                sharpness=multires_params['sharpness'],
                enabled=True
            )
            
            return out_final  # ← ИСПРАВЛЕНО: возвращаем с multires!
        else:
            return out_advanced  # ← Без multires

    except Exception as e:
        print(f"Advanced Tiling v3.1 Error: {e}")
        import traceback
        traceback.print_exc()
        return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

# ========================================================================
# РЕГИСТРАЦИЯ KOHAKU SAMPLER
# ========================================================================


# ========================================================================
# STEREOSCOPIC 3D (postprocess helper)
# ========================================================================

def _pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> torch float tensor (1,C,H,W) in [0,1] on CPU."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t


def _float_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """torch float tensor (1,C,H,W) in [0,1] -> PIL RGB."""
    t = t.detach().clamp(0.0, 1.0)
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _stereo_warp_tensor(t: torch.Tensor,
                        eye: str,
                        engine: str,
                        separation: float,
                        convergence: float,
                        depth_power: float = 1.0,
                        pole_power: float = 1.0) -> torch.Tensor:
    """
    Apply a lightweight stereo warp to an RGB tensor (1,3,H,W) in equirectangular space.
    - Engine A: legacy roll+blend
    - Engine B: spherical warp via grid_sample with longitude wrap and pole attenuation
    """
    eye = str(eye or "left").lower()
    sign = 1.0 if eye == "left" else -1.0

    separation = float(max(0.0, min(float(separation), 0.5)))
    convergence = float(max(0.0, min(float(convergence), 1.0)))
    depth_power = float(max(0.25, min(float(depth_power), 6.0)))
    pole_power = float(max(0.25, min(float(pole_power), 6.0)))

    B, C, H, W = t.shape
    if W <= 1 or H <= 1 or separation <= 0.0:
        return t

    # base u,v in [0,1]
    u = torch.linspace(0.0, 1.0, steps=W, dtype=torch.float32).view(1, 1, 1, W).expand(B, 1, H, W)
    v = torch.linspace(0.0, 1.0, steps=H, dtype=torch.float32).view(1, 1, H, 1).expand(B, 1, H, W)

    depth = torch.abs(u - convergence)
    depth = torch.pow(torch.clamp(depth, 0.0, 1.0), depth_power)

    if str(engine).startswith("A"):
        # Legacy roll+blend
        shift_px = int(round(W * separation))
        if shift_px == 0:
            return t
        rolled = torch.roll(t, shifts=int(round(sign * shift_px)), dims=3)
        alpha = depth.expand_as(t)
        return t * (1.0 - alpha) + rolled * alpha

    # Engine B: wrap-safe grid_sample warp
    lat = (0.5 - v) * math.pi
    attn = torch.pow(torch.cos(lat).clamp(0.0, 1.0), pole_power)

    shift_u = sign * separation * depth * attn
    u2 = torch.remainder(u + shift_u, 1.0)

    # grid_sample normalized coords
    x = u2 * 2.0 - 1.0
    y = v * 2.0 - 1.0
    grid = torch.cat([x, y], dim=1).permute(0, 2, 3, 1).contiguous()  # (B,H,W,2)

    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _make_stereo_outputs(img: Image.Image, settings: dict):
    """Returns (left,right) PIL images for the given input."""
    t = _pil_to_float_tensor(img)
    engine = settings.get("engine", "B (Spherical warp)")
    separation = settings.get("separation", 0.065)
    convergence = settings.get("convergence", 0.5)
    depth_power = settings.get("depth_power", 1.0)
    pole_power = settings.get("pole_power", 1.0)

    left_t = _stereo_warp_tensor(t, "left", engine, separation, convergence, depth_power, pole_power)
    right_t = _stereo_warp_tensor(t, "right", engine, separation, convergence, depth_power, pole_power)

    return _float_tensor_to_pil(left_t), _float_tensor_to_pil(right_t)

def register_kohaku_sampler():
    """Регистрирует Kohaku_LoNyu_Yog сэмплер в WebUI"""
    global _SAMPLER_REGISTERED

    if _SAMPLER_REGISTERED:
        return

    if any(s.name == 'Kohaku_LoNyu_Yog' for s in sd_samplers.all_samplers):
        _SAMPLER_REGISTERED = True
        return

    if not hasattr(k_diffusion.sampling, 'sample_kohaku_lonyu_yog'):
        setattr(k_diffusion.sampling, 'sample_kohaku_lonyu_yog', sample_kohaku_lonyu_yog)

    sampler_data = sd_samplers_common.SamplerData(
        name='Kohaku_LoNyu_Yog',
        constructor=lambda model: sd_samplers_kdiffusion.KDiffusionSampler('sample_kohaku_lonyu_yog', model),
        aliases=['kohaku', 'lonyu'],
        options={'second_order': True}
    )

    sd_samplers.all_samplers.append(sampler_data)
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}

    _SAMPLER_REGISTERED = True
    print("✓ Kohaku_LoNyu_Yog sampler registered successfully!")

# ========================================================================
# КЛАСС СКРИПТА (Advanced Tiling + Latent Mirroring)
# ========================================================================

class AdvancedTilingScriptV3(scripts.Script):
    def title(self):
        return "Advanced Tiling v3.0 PRO (Kohaku + Stereo + Aniso + Latent Mirror)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🚀 Advanced Tiling v3.0 PRO Edition", open=False):
            with gr.Row():
                enabled = gr.Checkbox(label="Enable Tiling", value=False)
                use_kohaku = gr.Checkbox(
                    label="Use Kohaku_LoNyu_Yog Sampler",
                    value=False,
                    info="Geometric second-order method for better seamless quality"
                )

            with gr.Tabs():
                with gr.Tab("🎨 Basic Modes"):
                    with gr.Row():
                        mode_x = gr.Dropdown(
                            label="Mode X",
                            choices=[MODE_OFF, MODE_CIRCULAR, MODE_MIRROR, MODE_HEXAGONAL,
                                    MODE_VORONOI, MODE_PERLIN, MODE_FRACTAL, MODE_ADAPTIVE],
                            value=MODE_CIRCULAR,
                            info="NEW: Voronoi/Perlin/Fractal/Adaptive modes available!"
                        )
                        mode_y = gr.Dropdown(
                            label="Mode Y",
                            choices=[MODE_OFF, MODE_CIRCULAR, MODE_MIRROR, MODE_HEXAGONAL,
                                    MODE_VORONOI, MODE_PERLIN, MODE_FRACTAL, MODE_ADAPTIVE],
                            value=MODE_CIRCULAR,
                            info="NEW: Voronoi/Perlin/Fractal/Adaptive modes available!"
                        )

                    with gr.Row():
                        multires = gr.Checkbox(
                            label="Multi-Resolution Mode", 
                            value=False,
                            info="Gradually transition from simple to advanced tiling"
                        )
                        # 🚀 ADVANCED ZOOM & PROXIMITY SYSTEM
                    with gr.Row():
                        use_zoom = gr.Checkbox(label="🔍 Advanced Zoom System (NEW!)", value=False)
                        use_blend = gr.Checkbox(label="Advanced Blend Mode", value=False)
                    
                    with gr.Accordion("🔍 Advanced Zoom Settings", open=False):
                        with gr.Row():
                            zoom_factor = gr.Slider(-5.0, 5.0, step=0.1, label="🎯 Zoom Factor", value=0.0)
                        with gr.Row():
                            zoom_mode = gr.Dropdown(label="Zoom Mode", choices=["outpaint_zoom", "blend_transition", "convergence_shift", "grid_warp", "hybrid", "spiral_zoom" ], value="outpaint_zoom")
                            blend_mode_type = gr.Dropdown(label="Blend Mode Type", choices=["circular_reflect", "circular_constant", "reflect_constant", "polar_circular", "mirror_circular", "aniso_circular", "custom", "gradient_radial", "noise_blend" ], value="circular_reflect")
                        with gr.Group():
                            gr.Markdown("### ⚡ Zoom V3.0 Quality Settings")
                            with gr.Row():
                                noise_strength = gr.Slider(minimum=0.5, maximum=2.0, step=0.05, value=1.0, 
                                    label="Noise Strength", info="1.0 = Standard. Higher = Stronger generation for outpainting.")
                                interp_mode = gr.Dropdown(choices=["bilinear", "bicubic", "nearest"], value="bilinear", 
                                    label="Interpolation", info="Method to stretch the image.")
                            
                            with gr.Row():
                                zoom_in_fade = gr.Checkbox(label="Zoom In Fade", value=True, info="Smoothes edges when zooming IN.")
                                variance_correction = gr.Checkbox(label="Variance Correction", value=True, info="Fixes gray colors on seams.")
                            
                            with gr.Row():
                                auto_clamp_pan = gr.Checkbox(label="Auto Clamp Pan", value=True, info="Prevents panning image off-screen.")
                                adaptive_noise_scale = gr.Checkbox(label="Adaptive Noise", value=True, info="Scales noise based on zoom level.")
                                debug_mode = gr.Checkbox(label="Debug Mode", value=False)
                        
                        with gr.Row():
                            zoom_convergence = gr.Slider(0.0, 1.0, step=0.05, label="Convergence Point", value=0.5)
                            zoom_depth_power = gr.Slider(0.25, 4.0, step=0.05, label="Depth Curve Power", value=1.0)

                        # --- НОВОЕ: Переключатель Безопасного Режима ---
                        with gr.Row():
                            zoom_force_original_size = gr.Checkbox(
                                label="🛡️ Force Original Size (Safety Mode)", 
                                value=True, 
                                info="Если включено: размер картинки всегда 512x512 (SD не падает). Если выключить: холст будет расширяться (Infinite Zoom)."
                            )


                        with gr.Accordion("🎯 Zoom Positioning", open=False):
                            gr.Markdown("""
                            **Точное позиционирование для Zoom:**
                            - Независимое от Latent Mirroring
                            - Контроль положения оригинала при zoom
                            """)
                            
                            with gr.Row():
                                zoom_pan_x = gr.Slider(
                                    minimum=-1.0, maximum=1.0, step=0.01,
                                    label='Zoom X Offset (Pan)', 
                                    value=0.0,
                                    info='Смещение по горизонтали: -1=влево, 0=центр, +1=вправо'
                                )
                                zoom_pan_y = gr.Slider(
                                    minimum=-1.0, maximum=1.0, step=0.01,
                                    label='Zoom Y Offset (Pan)', 
                                    value=0.0,
                                    info='Смещение по вертикали: -1=вверх, 0=центр, +1=вниз'
                                )
                            
                            # Быстрые пресеты
                            gr.Markdown("**Быстрые пресеты:**")
                            with gr.Row():
                                btn_center = gr.Button("📍 Center", size="sm")
                                btn_top = gr.Button("⬆️ Top", size="sm")
                                btn_bottom = gr.Button("⬇️ Bottom", size="sm")
                                btn_left = gr.Button("⬅️ Left", size="sm")
                                btn_right = gr.Button("➡️ Right", size="sm")
                            
                            # Привязка кнопок
                            btn_center.click(lambda: (0.0, 0.0), outputs=[zoom_pan_x, zoom_pan_y])
                            btn_top.click(lambda: (0.0, -0.5), outputs=[zoom_pan_x, zoom_pan_y])
                            btn_bottom.click(lambda: (0.0, 0.5), outputs=[zoom_pan_x, zoom_pan_y])
                            btn_left.click(lambda: (-0.5, 0.0), outputs=[zoom_pan_x, zoom_pan_y])
                            btn_right.click(lambda: (0.5, 0.0), outputs=[zoom_pan_x, zoom_pan_y])


                        with gr.Accordion("🌀 Spiral Zoom Settings", open=False):
                            gr.Markdown("""
                            **Параметры спирального зума:**
                            - Работает только с Zoom Mode = spiral_zoom
                            - Rotation Strength: сила вращения
                            - Direction: направление вращения
                            """)
                            
                            with gr.Row():
                                spiral_rotation = gr.Slider(
                                    0.0, 2.0, step=0.05,
                                    label="Rotation Strength",
                                    value=0.5,
                                    info="0 = нет вращения, 1 = средне, 2 = максимум"
                                )
                                
                                spiral_direction = gr.Radio(
                                    label="Rotation Direction",
                                    choices=["Clockwise", "Counter-clockwise"],
                                    value="Clockwise",
                                    type="index"  # 0 = Clockwise, 1 = Counter-clockwise
                                )
                            
                        with gr.Row():
                            zoom_fade_to_black = gr.Checkbox(
                                label="Enable Fade to Black (for Zoom)",
                                value=True,
                                info="Затемняет края для создания zoom эффекта"
                            )
    
                            zoom_fade_strength = gr.Slider(
                                minimum=0.0, maximum=0.5, step=0.01, value=0.15,
                                label="Fade Strength",
                                info="Размер зоны затемнения (0.1 = 10%, 0.2 = 20%)"
                            )

                    # СТАРЫЕ BLEND НАСТРОЙКИ
                    with gr.Row():
                        blend_strength = gr.Slider(0.0, 1.0, step=0.05, label="Blend Strength (legacy)", value=0.5)

                    # ═══ NEW: ADVANCED BLEND SETTINGS ═══
                    with gr.Accordion("🎨 Advanced Blend Settings", open=False):
                        gr.Markdown("""
                        **Blend Mode улучшен!** Теперь это настоящий слайдер приближения/отдаления:
                        - `0.0` = далеко (простой padding)
                        - `0.5` = средняя дистанция (смешивание)
                        - `1.0` = близко (полный advanced tiling)
                        """)
                        
                        with gr.Row():
                            blend_falloff = gr.Dropdown(
                                label="Falloff Curve",
                                choices=["linear", "smoothstep", "cosine", "perceptual"],
                                value="smoothstep",
                                info="Shape of the transition gradient"
                            )
                            blend_sharpness = gr.Slider(
                                0.1, 5.0, step=0.1,
                                label="Edge Sharpness",
                                value=1.0,
                                info="<1 = softer, >1 = sharper transitions"
                            )
                        
                        blend_width = gr.Slider(
                            0, 128, step=8,
                            label="Blend Width (pixels, 0=auto)",
                            value=0,
                            info="Width of transition zone"
                        )
                        

                    # ═══ NEW: ADVANCED MULTI-RESOLUTION SETTINGS ═══
                    with gr.Accordion("🔬 Advanced Multi-Resolution Settings", open=False):
                        gr.Markdown("""
                        **Multi-Resolution улучшен!** Гибкие стратегии и параметры:
                        - **Cosine**: плавная (рекомендуется)
                        - **Exponential**: быстрый старт
                        - **Sigmoid**: S-кривая
                        """)
                        
                        multires_strategy = gr.Dropdown(
                            label="Interpolation Strategy",
                            choices=["linear", "cosine", "exponential", "sigmoid", "early_boost", "late_smooth"],
                            value="cosine",
                            info="How to blend simple → advanced"
                        )
                        
                        with gr.Row():
                            multires_start = gr.Slider(
                                0.0, 1.0, step=0.05,
                                label="Transition Start",
                                value=0.0,
                                info="When to begin transition (0.0 = from start)"
                            )
                            multires_end = gr.Slider(
                                0.0, 1.0, step=0.05,
                                label="Transition End",
                                value=0.3,
                                info="When to finish transition (0.3 = at 30%)"
                            )
                        
                        multires_sharpness = gr.Slider(
                            0.1, 5.0, step=0.1,
                            label="Transition Sharpness",
                            value=1.0,
                            info="Speed of transition (<1 = slower, >1 = faster)"
                        )

                    # NEW ENHANCED MODES Parameters
                    with gr.Accordion("✨ NEW Enhanced Modes Parameters", open=False):
                        gr.Markdown("**Parameters for Voronoi, Perlin, Fractal, and Adaptive modes**")
                        
                        with gr.Row():
                            voronoi_cells = gr.Slider(
                                label="Voronoi: Cells",
                                minimum=4, maximum=32, value=8, step=1,
                                info="Number of organic cells for Voronoi mode"
                            )
                            voronoi_seed = gr.Slider(
                                label="Voronoi: Seed",
                                minimum=0, maximum=9999, value=42, step=1,
                                info="Random seed for pattern reproducibility"
                            )
                        
                        with gr.Row():
                            perlin_strength = gr.Slider(
                                label="Perlin: Distortion Strength",
                                minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                                info="How much Perlin noise distorts tiling"
                            )
                            perlin_scale = gr.Slider(
                                label="Perlin: Noise Scale",
                                minimum=1.0, maximum=50.0, value=10.0, step=1.0,
                                info="Scale of Perlin noise pattern"
                            )
                        
                        with gr.Row():
                            fractal_iterations = gr.Slider(
                                label="Fractal: Iterations",
                                minimum=1, maximum=4, value=2, step=1,
                                info="Number of fractal recursion levels"
                            )
                            fractal_scale = gr.Slider(
                                label="Fractal: Scale Factor",
                                minimum=0.3, maximum=0.9, value=0.6, step=0.05,
                                info="Size reduction per iteration"
                            )
                        
                        adaptive_threshold = gr.Slider(
                            label="Adaptive: Edge Threshold",
                            minimum=0.0, maximum=1.0, value=0.1, step=0.05,
                            info="Edge detection threshold for auto mode selection"
                        )

                with gr.Tab("🌐 Advanced Modes"):
                    gr.Markdown("**Специальные режимы для сложных топологий**")

                    with gr.Row():
                        use_panorama = gr.Checkbox(label="Panorama 360°", value=False)
                        use_polar = gr.Checkbox(
                            label="Polar (Sphere Correct)",
                            value=False,
                            info="Correct pole transitions for equirectangular"
                        )


                    with gr.Accordion("🧭 Panorama Live (panorama-tools integrated)", open=False):
                        gr.Markdown("**Live equirectangular panorama processing during generation**")

                        with gr.Row():
                            panorama_engine = gr.Dropdown(
                                label="Panorama Engine",
                                choices=["A (Legacy)", "B (Fast Pole-Correct)", "C (Live 3D GridSample)"],
                                value="C (Live 3D GridSample)",
                                info="A = legacy circular wrap. B = fast pole-correct padding (+ optional antipode + pole blur). C = true spherical mapping via grid_sample (best seams, slower)."
                            )
                            panorama_coord_mode = gr.Dropdown(
                                label="Engine C: Coord Mode",
                                choices=["Cartesian (lon/lat)", "Polar (radial warp around poles)"],
                                value="Cartesian (lon/lat)",
                                info="Advanced: Cartesian uses standard lon/lat. Polar re-parameterizes latitude near poles."
                            )

                        with gr.Row():
                            panorama_yaw = gr.Slider(-180, 180, step=1, label="Engine C: Yaw (°)", value=0)
                            panorama_pitch = gr.Slider(-180, 180, step=1, label="Engine C: Pitch (°)", value=0)
                            panorama_roll = gr.Slider(-180, 180, step=1, label="Engine C: Roll (°)", value=0)

                        with gr.Row():
                            panorama_twist_deg = gr.Slider(-180, 180, step=1, label="Engine C: Twist (°)", value=0,
                                                          info="Latitude-dependent longitude twist (local roll feel).")
                            panorama_twist_power = gr.Slider(0.25, 4.0, step=0.01, label="Engine C: Twist Power", value=1.0)

                            panorama_swirl_deg = gr.Slider(-180, 180, step=1, label="Engine C: Swirl (°)", value=0,
                                                          info="Extra rotation bias toward poles (0=off).")
                            panorama_swirl_power = gr.Slider(0.25, 4.0, step=0.01, label="Engine C: Swirl Power", value=1.0)

                        with gr.Row():
                            panorama_polar_scale = gr.Slider(0.25, 2.0, step=0.01, label="Engine C: Polar Scale", value=1.0)
                            panorama_polar_power = gr.Slider(0.25, 4.0, step=0.01, label="Engine C: Polar Power", value=1.0)
                            panorama_pole_ease_power = gr.Slider(0.25, 4.0, step=0.01, label="Pole Easing Power", value=1.0,
                                                                 info="Eases latitude magnitude near poles (1=off).")

                        with gr.Row():
                            panorama_antipode_strength = gr.Slider(0.0, 1.0, step=0.05, label="Antipode Mix (anti-seam)", value=0.0,
                                                                  info="Kohaku-inspired mixing with the antipode point (0=off).")
                            panorama_pole_blur_strength = gr.Slider(0.0, 1.0, step=0.05, label="Pole Blur Strength", value=0.0)
                            panorama_pole_blur_radius = gr.Slider(0, 16, step=1, label="Pole Blur Radius (px)", value=0)

                        with gr.Row():
                            panorama_pole_blur_power = gr.Slider(0.25, 4.0, step=0.01, label="Pole Blur Mask Power", value=1.0)
                            panorama_geoaa_samples = gr.Slider(1, 4, step=1, label="Engine C: Geo AA Samples", value=1)
                            panorama_geoaa_radius = gr.Slider(0.0, 2.0, step=0.25, label="Engine C: Geo AA Radius (px)", value=0.0)

                        with gr.Row():
                            panorama_grid_interp = gr.Dropdown(label="Engine C: grid_sample Interp",
                                                               choices=["bilinear", "nearest"],
                                                               value="bilinear")
                            panorama_grid_padding = gr.Dropdown(label="Engine C: grid_sample Padding",
                                                                choices=["border", "reflection", "zeros"],
                                                                value="border")
                            panorama_cache_angle_quant = gr.Slider(0.1, 5.0, step=0.1, label="Engine C: Cache Quant (°)", value=0.5)

                    with gr.Row():
                        use_cubemap = gr.Checkbox(
                            label="Cubemap (3D)",
                            value=False,
                            info="3x2 cubemap net (S/E/N over B/T/W). Faces must be square: width/3 == height/2."
                        )
                        cubemap_engine = gr.Dropdown(
                            label="Cubemap Engine",
                            choices=["A (Fast)", "B (Advanced)", "C (3D GridSample)"],
                            value="A (Fast)",
                            info="A = fastest batched cubemap padding. B = seam-aware padding blend (reduces seams). C = true 3D grid_sample mapping (yaw/pitch/roll, slower but best seams)."
                        )
                        cubemap_resolution_mode = gr.Dropdown(
                            label="Cubemap Input Size",
                            choices=["Already 3x2 (Full image)", "Face size (auto scale to 3x2)"],
                            value="Already 3x2 (Full image)",
                            info="If you choose 'Face size', your base width/height will be interpreted as a single face."
                        )

                    with gr.Row():
                        cubemap_pad_mode = gr.Dropdown(
                            label="Cubemap Pad Mode",
                            choices=["replicate", "reflect", "circular"],
                            value="replicate",
                            info="Padding mode used when assembling neighbor strips."
                        )
                        cubemap_blend_width = gr.Slider(
                            0, 16, step=1,
                            label="Cubemap Seam Width (px)",
                            value=2,
                            info="Only for Engine B: how many pixels inside padding are blended."
                        )
                        cubemap_blend_strength = gr.Slider(
                            0.0, 1.0, step=0.05,
                            label="Cubemap Seam Strength",
                            value=0.75,
                            info="Only for Engine B: 0 = Engine A behavior, 1 = full seam blending."
                        )



                    with gr.Row():
                        cubemap_yaw = gr.Slider(
                            -180, 180, step=1,
                            label="Engine C: Yaw (°)",
                            value=0,
                            info="Only for Engine C: rotation around Y axis."
                        )
                        cubemap_pitch = gr.Slider(
                            -180, 180, step=1,
                            label="Engine C: Pitch (°)",
                            value=0,
                            info="Only for Engine C: rotation around X axis."
                        )
                        cubemap_roll = gr.Slider(
                            -180, 180, step=1,
                            label="Engine C: Roll (°)",
                            value=0,
                            info="Only for Engine C: rotation around Z axis."
                        )

                    # --- Engine C advanced coordinate controls ---
                    with gr.Row():
                        cubemap_coord_mode = gr.Dropdown(
                            choices=["Cartesian (Face UV)", "Polar (Radial Warp)"],
                            value="Cartesian (Face UV)",
                            label="Engine C: Coord Mode",
                            info="Advanced: how face UV is interpreted before cubemap projection."
                        )
                        cubemap_twist_deg = gr.Slider(
                            -180, 180, step=1,
                            label="Engine C: Face Twist (°)",
                            value=0,
                            info="Advanced: rotate face UV around face normal before projection."
                        )

                    with gr.Row():
                        cubemap_polar_scale = gr.Slider(
                            0.25, 2.0, step=0.01,
                            label="Engine C: Polar Radius Scale",
                            value=1.0,
                            info="Only for Polar mode: r' = (r*scale)^power."
                        )
                        cubemap_polar_power = gr.Slider(
                            0.25, 4.0, step=0.01,
                            label="Engine C: Polar Power",
                            value=1.0,
                            info="Only for Polar mode: r' = (r*scale)^power."
                        )

                    with gr.Row():
                        cubemap_swirl_deg = gr.Slider(
                            -180, 180, step=1,
                            label="Engine C: Swirl (°)",
                            value=0,
                            info="Extra angle: adds radius-dependent twist (0=off)."
                        )
                        cubemap_swirl_power = gr.Slider(
                            0.25, 4.0, step=0.01,
                            label="Engine C: Swirl Power",
                            value=1.0,
                            info="Swirl curve: twist ∝ r^power."
                        )

                    with gr.Row():
                        cubemap_grid_interp = gr.Dropdown(
                            choices=["bilinear", "nearest"],
                            value="bilinear",
                            label="Engine C: grid_sample Interp",
                            info="Bilinear=best quality. Nearest=faster/sharper."
                        )
                        cubemap_grid_padding = gr.Dropdown(
                            choices=["border", "reflection", "zeros"],
                            value="border",
                            label="Engine C: grid_sample Padding",
                            info="border recommended for seams."
                        )
                        cubemap_cache_angle_quant = gr.Slider(
                            0.1, 5.0, step=0.1,
                            label="Engine C: Cache Quant (°)",
                            value=0.5,
                            info="Cache step for yaw/pitch/roll/twist/swirl. Lower=more precise, slower."
                        )

                    with gr.Row():
                        cubemap_geoaa_samples = gr.Slider(
                            1, 4, step=1,
                            label="Engine C: Geo AA Samples",
                            value=1,
                            info="Only for Engine C: multi-sampling (1=off). Higher = smoother but slower."
                        )
                        cubemap_geoaa_radius = gr.Slider(
                            0.0, 2.0, step=0.25,
                            label="Engine C: Geo AA Radius (px)",
                            value=0.0,
                            info="Only for Engine C: jitter radius in atlas pixels."
                        )
                        cubemap_antipode_strength = gr.Slider(
                            0.0, 1.0, step=0.05,
                            label="Engine C: Antipode Mix",
                            value=0.0,
                            info="Only for Engine C: Kohaku-inspired antipode blending on the sphere."
                        )

                    with gr.Row():
                        use_anisotropic = gr.Checkbox(
                            label="Anisotropic (Directional)",
                            value=False,
                            info="Different behavior along diagonals"
                        )
                        aniso_angle = gr.Slider(
                            0, 360, step=15,
                            label="Anisotropic Angle",
                            value=45,
                            info="Direction of fibers/texture (degrees)"
                        )

                        aniso_angle2 = gr.Slider(
                            0, 360, step=15,
                            label="Second Angle (optional)",
                            value=0,
                            info="Advanced: enable by setting Angle Mix < 1.0"
                        )
                        aniso_angle_mix = gr.Slider(
                            0.0, 1.0, step=0.05,
                            label="Angle Mix (Angle1 → Angle2)",
                            value=1.0,
                            info="1.0 = only Angle1, 0.0 = only Angle2."
                        )


                with gr.Tab("🎭 Stereoscopic 3D"):
                    gr.Markdown("**Генерация стереопар для 3D контента**")

                    with gr.Row():
                        stereo_enabled = gr.Checkbox(label="Enable Stereoscopic Mode", value=False)
                        stereo_eye = gr.Radio(
                            label="Eye",
                            choices=["left", "right", "both"],
                            value="left",
                            info="Which eye view to generate"
                        )


                    with gr.Row():
                        stereo_engine = gr.Dropdown(
                            label="Stereo Engine",
                            choices=["A (Legacy shift)", "B (Spherical warp)"],
                            value="B (Spherical warp)",
                            info="B = smoother spherical warp (recommended). A = old-style shift/blend."
                        )
                        stereo_output = gr.Dropdown(
                            label="Output",
                            choices=["Side-by-Side", "Left only", "Right only", "Both separate"],
                            value="Side-by-Side",
                            info="Default: create a single side-by-side image. You can also output only one eye or two separate images."
                        )
                        stereo_in_model = gr.Checkbox(
                            label="Apply stereo during denoise (experimental)",
                            value=False,
                            info="If enabled, stereo shift affects UNet/conv padding during generation. If disabled, stereo is applied as a postprocess (recommended)."
                        )

                    with gr.Row():
                        stereo_depth_power = gr.Slider(
                            0.25, 4.0, step=0.05,
                            label="Depth Curve Power",
                            value=1.0,
                            info="Higher values push parallax more toward edges."
                        )
                        stereo_pole_power = gr.Slider(
                            0.25, 4.0, step=0.05,
                            label="Pole Attenuation Power",
                            value=1.0,
                            info="Scales stereo shift by cos(latitude)^power to reduce pole stretching."
                        )

                    with gr.Row():
                        stereo_separation = gr.Slider(
                            0.0, 0.15, step=0.005,
                            label="IPD (Inter-Pupillary Distance)",
                            value=0.065,
                            info="Eye separation as fraction of width"
                        )
                        stereo_convergence = gr.Slider(
                            0.0, 1.0, step=0.05,
                            label="Convergence Point",
                            value=0.5,
                            info="Depth at which eyes converge"
                        )

                    gr.Markdown("""
                    **💡 Совет для стерео:**
                    - Генерируйте сначала левый глаз
                    - Затем правый с теми же параметрами
                    - Используйте Side-by-Side или Anaglyph компоновку
                    """)

                with gr.Tab("🪞 Latent Mirroring"):
                    # Главный переключатель латентного зеркалирования
                    enable_mirroring = gr.Checkbox(
                        label="Enable Latent Mirroring",
                        value=False
                    )

                    with gr.Group():
                        mirror_mode = gr.Radio(
                            label='Latent Mirror mode',
                            choices=['None', 'Alternate Steps', 'Blend Average'],
                            value='None',
                            type="index"
                        )
                        mirror_style = gr.Radio(
                            label='Latent Mirror style',
                            choices=[
                                'Horizontal Mirroring',
                                'Vertical Mirroring',
                                'Horizontal+Vertical Mirroring',
                                '90 Degree Rotation',
                                '180 Degree Rotation',
                                'Roll Channels',
                                'None'
                            ],
                            value='Horizontal Mirroring',
                            type="index"
                        )

                        with gr.Row():
                            x_pan = gr.Slider(
                                minimum=-1.0, maximum=1.0, step=0.01,
                                label='X panning', value=0.0
                            )
                            y_pan = gr.Slider(
                                minimum=-1.0, maximum=1.0, step=0.01,
                                label='Y panning', value=0.0
                            )

                        mirroring_max_step_fraction = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01,
                            label='Maximum steps fraction to mirror at',
                            value=0.25
                        )

                        # Режим интерпретации шага для mirroring:
                        # 0 = Max fraction (оригинал), 1 = Custom range (Start/End)
                        mirror_step_mode = gr.Radio(
                            label="Mirroring Step Mode",
                            choices=["Max fraction (original)", "Custom range"],
                            value="Max fraction (original)",
                            type="index"
                        )
                        mirror_start_frac = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01,
                            label="Mirror Start (fraction of total steps)",
                            value=0.0
                        )
                        mirror_end_frac = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01,
                            label="Mirror End (fraction of total steps)",
                            value=0.25
                         )
                        
                        if not is_img2img:
                            disable_hr = gr.Checkbox(label='Disable during hires pass', value=False)
                        else:
                            disable_hr = gr.State(False)

            with gr.Row():
                start_step = gr.Slider(0, 150, step=1, label="Start Step", value=0)
                end_step = gr.Slider(0, 150, step=1, label="End Step", value=150)

            # Режим интерпретации Start/End Step: абсолютные шаги или доли от p.steps
            with gr.Row():
                step_mode = gr.Radio(
                    label="Tiling Step Mode",
                    choices=["Absolute Steps", "Fraction of total steps"],
                    value="Absolute Steps",
                    type="index"
                )
                step_start_frac = gr.Slider(
                    0.0, 1.0, step=0.01,
                    label="Start (fraction of total steps)",
                    value=0.0
                )
                step_end_frac = gr.Slider(
                    0.0, 1.0, step=0.01,
                    label="End (fraction of total steps)",
                    value=1.0
                )

            with gr.Row():
                patch_vae = gr.Checkbox(
                    label="Patch VAE Decoder",
                    value=True,
                    info="Fix seams in final pixel decode"
                )

            with gr.Row():
                tiling_disable_hr = gr.Checkbox(
                    label="Disable Advanced Tiling during hires pass",
                    value=False,
                    info=(
                        "When hires fix is enabled, apply Advanced Tiling only on "
                        "the first (base) sampling pass"
                    )
                )

            # Preview Tool
            with gr.Accordion("🔍 Preview & Info", open=False):
                with gr.Tabs():
                    with gr.Tab("Visual Preview"):
                        preview_btn = gr.Button("Generate Preview", variant="primary")
                        preview_img = gr.Image(label="Preview Grid (2x2)")

                        preview_btn.click(
                            fn=self.generate_preview,
                            inputs=[mode_x, mode_y, use_anisotropic, aniso_angle,
                                    stereo_enabled, stereo_eye],
                            outputs=preview_img
                        )

                    with gr.Tab("Kohaku Info"):
                        gr.Markdown("""
                        ## 🔬 Kohaku_LoNyu_Yog Принцип работы
                        (описание опущено ради компактности)
                        """)

        # для Latent Mirroring
        self.run_callback = False

        return [enabled, mode_x, mode_y,
                start_step, end_step,
                step_mode, step_start_frac, step_end_frac,
                # --- ИСПРАВЛЕНИЕ: Добавлены новые параметры multires ---
                multires, multires_strategy, multires_start, multires_end, multires_sharpness,
                
                voronoi_cells, voronoi_seed,
                perlin_strength, perlin_scale,
                fractal_iterations, fractal_scale,
                adaptive_threshold,
                use_panorama, use_polar,
                panorama_engine, panorama_coord_mode,
                panorama_yaw, panorama_pitch, panorama_roll,
                panorama_twist_deg, panorama_twist_power, panorama_swirl_deg, panorama_swirl_power,
                panorama_polar_scale, panorama_polar_power, panorama_pole_ease_power,
                panorama_antipode_strength, panorama_pole_blur_strength, panorama_pole_blur_radius, panorama_pole_blur_power,
                panorama_geoaa_samples, panorama_geoaa_radius,
                panorama_grid_interp, panorama_grid_padding, panorama_cache_angle_quant,
                use_cubemap, cubemap_engine, cubemap_resolution_mode, cubemap_pad_mode, cubemap_blend_width, cubemap_blend_strength,
                cubemap_yaw, cubemap_pitch, cubemap_roll, cubemap_coord_mode, cubemap_twist_deg, cubemap_polar_scale, cubemap_polar_power, cubemap_swirl_deg, cubemap_swirl_power, cubemap_grid_interp, cubemap_grid_padding, cubemap_cache_angle_quant, cubemap_geoaa_samples, cubemap_geoaa_radius, cubemap_antipode_strength,
                
                # --- ИСПРАВЛЕНИЕ: Добавлены новые параметры blend ---
                use_blend, blend_strength, blend_falloff, blend_sharpness, blend_width,

                # --- ZOOM PARAMETERS ---
                use_zoom, zoom_factor, zoom_mode,noise_strength, interp_mode, zoom_in_fade,
                variance_correction, auto_clamp_pan, adaptive_noise_scale, debug_mode,
                blend_mode_type,
                zoom_convergence, zoom_depth_power,
                zoom_pan_x, zoom_pan_y,           # 🆕 ДОБАВЛЕНО
                spiral_rotation, spiral_direction, # 🆕 ДОБАВЛЕНО
                use_anisotropic, aniso_angle, aniso_angle2, aniso_angle_mix,
                stereo_enabled, stereo_eye, stereo_engine, stereo_output, stereo_in_model, stereo_separation, stereo_convergence, stereo_depth_power, stereo_pole_power,
                patch_vae, tiling_disable_hr, use_kohaku,
                enable_mirroring,
                mirror_mode, mirror_style, x_pan, y_pan,
                mirroring_max_step_fraction,
                mirror_step_mode, mirror_start_frac, mirror_end_frac,
                disable_hr]

    # ====== LATENT MIRRORING CALLBACK ======

    def denoise_callback(self, params):
        # --- Детекция hires-прохода для Advanced Tiling ---
        tiling_is_hires = getattr(self, "tiling_is_hires", False)
        if getattr(self, "tiling_enable_hr", False) and params.total_sampling_steps > 0:
            if params.sampling_step >= params.total_sampling_steps - 2:
                # Переключаем флаг между base/hires
                self.tiling_is_hires = not tiling_is_hires

        # --- Оригинальная логика Latent Mirroring ---
        is_hires = self.is_hires

        # indices start at -1; params.sampling_step = max(0, real_sampling_step)
        if params.sampling_step >= params.total_sampling_steps - 2:
            self.is_hires = not is_hires and self.enable_hr

        if not self.run_callback or is_hires:
            return

        cur_step = params.sampling_step
        total_steps = max(params.total_sampling_steps, 1)

        # Режим интерпретации шагов mirroring:
        # 0 = Max fraction (оригинал),
        # 1 = Custom range [mirror_start_frac, mirror_end_frac]
        mirror_step_mode = getattr(self, "mirror_step_mode", 0)

        if mirror_step_mode == 0:
            # Оригинальное поведение: от начала до max_fraction * total_steps
            if cur_step >= total_steps * self.mirroring_max_step_fraction:
                return
        else:
            # Новый режим: собственный диапазон Start/End в долях
            start_step = int(total_steps * getattr(self, "mirror_start_frac", 0.0))
            end_step = int(total_steps * getattr(self, "mirror_end_frac", 1.0))
            if start_step > end_step:
                start_step, end_step = end_step, start_step
            if not (start_step <= cur_step <= end_step):
                return

        try:
            if self.mirror_mode == 1:
                if self.mirror_style == 0:
                    params.x[:, :, :, :] = torch.flip(params.x, [3])
                elif self.mirror_style == 1:
                    params.x[:, :, :, :] = torch.flip(params.x, [2])
                elif self.mirror_style == 2:
                    params.x[:, :, :, :] = torch.flip(params.x, [3, 2])
                elif self.mirror_style == 3:
                    params.x[:, :, :, :] = torch.rot90(params.x, dims=[2, 3])
                elif self.mirror_style == 4:
                    params.x[:, :, :, :] = torch.rot90(
                        torch.rot90(params.x, dims=[2, 3]), dims=[2, 3]
                    )
                elif self.mirror_style == 5:
                    params.x[:, :, :, :] = torch.roll(params.x, shifts=1, dims=[1])

            elif self.mirror_mode == 2:
                if self.mirror_style == 0:
                    params.x[:, :, :, :] = (torch.flip(params.x, [3]) + params.x) / 2
                elif self.mirror_style == 1:
                    params.x[:, :, :, :] = (torch.flip(params.x, [2]) + params.x) / 2
                elif self.mirror_style == 2:
                    params.x[:, :, :, :] = (torch.flip(params.x, [2, 3]) + params.x) / 2
                elif self.mirror_style == 3:
                    params.x[:, :, :, :] = (torch.rot90(params.x, dims=[2, 3]) + params.x) / 2
                elif self.mirror_style == 4:
                    params.x[:, :, :, :] = (
                        torch.rot90(torch.rot90(params.x, dims=[2, 3]), dims=[2, 3]) + params.x
                    ) / 2
                elif self.mirror_style == 5:
                    params.x[:, :, :, :] = (torch.roll(params.x, shifts=1, dims=[1]) + params.x) / 2
        except RuntimeError as e:
            if self.mirror_style in (3, 4):
                raise RuntimeError('90 Degree Rotation requires a square image.') from e
            else:
                raise RuntimeError('Error transforming image for latent mirroring.') from e

        if self.x_pan != 0:
            params.x[:, :, :, :] = torch.roll(
                params.x,
                shifts=int(params.x.size()[3] * self.x_pan),
                dims=[3]
            )
        if self.y_pan != 0:
            params.x[:, :, :, :] = torch.roll(
                params.x,
                shifts=int(params.x.size()[2] * self.y_pan),
                dims=[2]
            )

    # ====== PREVIEW ======

    def generate_preview(self, mx, my, use_aniso, aniso_angle, stereo, eye):
        """Генерация preview с учетом новых режимов"""
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                col = (i + j) % 255
                if i < 5 or i > h - 5 or j < 5 or j > w - 5:
                    img[i, j] = [255, 50, 50]
                else:
                    img[i, j] = [col, col // 2, 255 - col]

        def get_tile(r, c):
            tile = img.copy()

            if use_aniso:
                angle_rad = np.radians(aniso_angle)
                for ii in range(h):
                    for jj in range(w):
                        dist = abs((jj - w / 2) * np.cos(angle_rad) +
                                   (ii - h / 2) * np.sin(angle_rad))
                        tile[ii, jj] = tile[ii, jj] * (0.5 + 0.5 * np.sin(dist * 0.1))

            elif stereo:
                shift = 10 if eye == 'left' else -10
                tile = np.roll(tile, shift, axis=1)

            else:
                if mx == MODE_MIRROR and c % 2 != 0:
                    tile = np.fliplr(tile)
                if my == MODE_MIRROR and r % 2 != 0:
                    tile = np.flipud(tile)
                if mx == MODE_HEXAGONAL and r % 2 != 0:
                    tile = np.roll(tile, w // 2, axis=1)

            return tile.astype(np.uint8)

        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        for r in range(2):
            for c in range(2):
                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = get_tile(r, c)

        return canvas

    # ====== PROCESS (TILING + MIRRORING) ======

    def process(self, p,
                enabled, mode_x, mode_y,
                start_step, end_step,
                step_mode, step_start_frac, step_end_frac,
                multires, multires_strategy, multires_start, multires_end, multires_sharpness,
                voronoi_cells, voronoi_seed,
                perlin_strength, perlin_scale,
                fractal_iterations, fractal_scale,
                adaptive_threshold,
                use_panorama, use_polar,
                panorama_engine, panorama_coord_mode,
                panorama_yaw, panorama_pitch, panorama_roll,
                panorama_twist_deg, panorama_twist_power, panorama_swirl_deg, panorama_swirl_power,
                panorama_polar_scale, panorama_polar_power, panorama_pole_ease_power,
                panorama_antipode_strength, panorama_pole_blur_strength, panorama_pole_blur_radius, panorama_pole_blur_power,
                panorama_geoaa_samples, panorama_geoaa_radius,
                panorama_grid_interp, panorama_grid_padding, panorama_cache_angle_quant,
                use_cubemap, cubemap_engine, cubemap_resolution_mode, cubemap_pad_mode, cubemap_blend_width, cubemap_blend_strength,
                cubemap_yaw, cubemap_pitch, cubemap_roll, cubemap_coord_mode, cubemap_twist_deg, cubemap_polar_scale, cubemap_polar_power, cubemap_swirl_deg, cubemap_swirl_power, cubemap_grid_interp, cubemap_grid_padding, cubemap_cache_angle_quant, cubemap_geoaa_samples, cubemap_geoaa_radius, cubemap_antipode_strength,
                use_blend, blend_strength, blend_falloff, blend_sharpness, blend_width,
                # --- ZOOM PARAMETERS ---
                use_zoom, zoom_factor, zoom_mode,noise_strength, interp_mode, zoom_in_fade,
                variance_correction, auto_clamp_pan, adaptive_noise_scale, debug_mode,
                blend_mode_type,
                zoom_convergence, zoom_depth_power,
                zoom_pan_x, zoom_pan_y,           # 🆕 ДОБАВЛЕНО
                spiral_rotation, spiral_direction, # 🆕 ДОБАВЛЕНО
                use_anisotropic, aniso_angle, aniso_angle2, aniso_angle_mix,
                stereo_enabled, stereo_eye, stereo_engine, stereo_output, stereo_in_model, stereo_separation, stereo_convergence, stereo_depth_power, stereo_pole_power,
                patch_vae, tiling_disable_hr, use_kohaku,
                enable_mirroring,
                mirror_mode, mirror_style, x_pan, y_pan,
                mirroring_max_step_fraction,
                mirror_step_mode, mirror_start_frac, mirror_end_frac,
                disable_hr):

        # -------- Latent Mirroring часть --------
        self.mirror_mode = mirror_mode
        self.mirror_style = mirror_style
        self.mirroring_max_step_fraction = mirroring_max_step_fraction
        self.x_pan = x_pan
        self.y_pan = y_pan
        self.mirror_step_mode = mirror_step_mode
        self.mirror_start_frac = mirror_start_frac
        self.mirror_end_frac = mirror_end_frac

        # Mirroring включается, только если:
        # - чекбокс Enable Latent Mirroring включён
        # - и есть хотя бы какой-то эффект (mode или pan)
        need_mirroring = enable_mirroring and (mirror_mode != 0 or x_pan != 0 or y_pan != 0)

        # denoise_callback нужен либо для mirroring,
        # либо для Advanced Tiling с отключением на hires-проходе
        need_denoise_cb = need_mirroring or (tiling_disable_hr and getattr(p, "enable_hr", False))

        # Экспорт параметров только если Latent Mirroring реально активен
        if need_mirroring:
            if mirror_mode != 0:
                p.extra_generation_params["Mirror Mode"] = mirror_mode
                p.extra_generation_params["Mirror Style"] = mirror_style
                if mirror_step_mode == 0:
                    # Оригинальный режим — до max_fraction
                    p.extra_generation_params["Mirroring Max Step Fraction"] = mirroring_max_step_fraction
                else:
                    # Новый режим — диапазон Start/End в долях
                    p.extra_generation_params["Mirror Start Fraction"] = mirror_start_frac
                    p.extra_generation_params["Mirror End Fraction"] = mirror_end_frac
            if x_pan != 0:
                p.extra_generation_params["X Pan"] = x_pan
            if y_pan != 0:
                p.extra_generation_params["Y Pan"] = y_pan

        if need_denoise_cb and not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoise_callback)
            self.callbacks_added = True

        # run_callback управляет только Latent Mirroring
        self.run_callback = need_mirroring

        # hires-логика для Latent Mirroring (оригинальное поведение)
        self.enable_hr = getattr(p, 'enable_hr', False) and not disable_hr
        self.is_hires = False

        # hires-логика для Advanced Tiling (независимо от mirroring)
        self.tiling_enable_hr = getattr(p, 'enable_hr', False)
        self.tiling_is_hires = False

        # -------- Advanced Tiling часть --------
        if not enabled:
            return


        # -------- Обработка режимов Start/End Step для Tiling --------
        # step_mode: 0 = Absolute Steps, 1 = Fraction of total steps
        if step_mode == 1:
            total_steps = max(getattr(p, "steps", 0), 1)
            # Преобразуем доли в реальные шаги
            start_step = int(total_steps * step_start_frac)
            end_step = int(total_steps * step_end_frac)

        # Валидация диапазона
        if start_step > end_step:
            start_step, end_step = end_step, start_step

        # Kohaku
        if use_kohaku:
            register_kohaku_sampler()
            if p.sampler_name != 'Kohaku_LoNyu_Yog':
                print(f"Switching sampler from {p.sampler_name} to Kohaku_LoNyu_Yog")
                p.sampler_name = 'Kohaku_LoNyu_Yog'


                
        # Export panorama / stereo / cubemap settings to metadata (inside process)
        try:
            if getattr(p, "extra_generation_params", None) is None:
                p.extra_generation_params = {}
        except Exception:
            pass

        if use_panorama:
            try:
                p.extra_generation_params["Panorama Engine"] = str(panorama_engine)
                if str(panorama_engine).startswith(("B", "C")):
                    p.extra_generation_params["Panorama Antipode Mix"] = float(panorama_antipode_strength)
                    p.extra_generation_params["Panorama Pole Blur"] = float(panorama_pole_blur_strength)
                    p.extra_generation_params["Panorama Pole Blur Radius"] = int(panorama_pole_blur_radius)
                    p.extra_generation_params["Panorama Pole Easing Power"] = float(panorama_pole_ease_power)
                if str(panorama_engine).startswith("C"):
                    p.extra_generation_params["Panorama Yaw"] = float(panorama_yaw)
                    p.extra_generation_params["Panorama Pitch"] = float(panorama_pitch)
                    p.extra_generation_params["Panorama Roll"] = float(panorama_roll)
                    p.extra_generation_params["Panorama Coord Mode"] = str(panorama_coord_mode)
                    p.extra_generation_params["Panorama Twist"] = float(panorama_twist_deg)
                    p.extra_generation_params["Panorama Twist Power"] = float(panorama_twist_power)
                    p.extra_generation_params["Panorama Swirl"] = float(panorama_swirl_deg)
                    p.extra_generation_params["Panorama Swirl Power"] = float(panorama_swirl_power)
                    p.extra_generation_params["Panorama Polar Scale"] = float(panorama_polar_scale)
                    p.extra_generation_params["Panorama Polar Power"] = float(panorama_polar_power)
                    p.extra_generation_params["Panorama GeoAA Samples"] = int(panorama_geoaa_samples)
                    p.extra_generation_params["Panorama GeoAA Radius"] = float(panorama_geoaa_radius)
                    p.extra_generation_params["Panorama grid_sample Interp"] = str(panorama_grid_interp)
                    p.extra_generation_params["Panorama grid_sample Padding"] = str(panorama_grid_padding)
                    p.extra_generation_params["Panorama Cache Quant"] = float(panorama_cache_angle_quant)
            except Exception:
                pass

        if stereo_enabled:
            try:
                p.extra_generation_params["Stereoscopic 3D"] = "Enabled"
                p.extra_generation_params["Stereo Engine"] = str(stereo_engine)
                p.extra_generation_params["Stereo Output"] = str(stereo_output)
                p.extra_generation_params["Stereo In-Model"] = bool(stereo_in_model)
                p.extra_generation_params["Stereo Separation"] = float(stereo_separation)
                p.extra_generation_params["Stereo Convergence"] = float(stereo_convergence)
                p.extra_generation_params["Stereo Depth Power"] = float(stereo_depth_power)
                p.extra_generation_params["Stereo Pole Power"] = float(stereo_pole_power)
            except Exception:
                pass

        if use_cubemap:
            try:
                p.extra_generation_params["Cubemap Engine"] = str(cubemap_engine)
                p.extra_generation_params["Cubemap Pad Mode"] = str(cubemap_pad_mode)
                if str(cubemap_engine).startswith("B"):
                    p.extra_generation_params["Cubemap Seam Width"] = int(cubemap_blend_width)
                    p.extra_generation_params["Cubemap Seam Strength"] = float(cubemap_blend_strength)
                if str(cubemap_engine).startswith("C"):
                    p.extra_generation_params["Cubemap Yaw"] = float(cubemap_yaw)
                    p.extra_generation_params["Cubemap Pitch"] = float(cubemap_pitch)
                    p.extra_generation_params["Cubemap Roll"] = float(cubemap_roll)
                    p.extra_generation_params["Cubemap Coord Mode"] = str(cubemap_coord_mode)
                    p.extra_generation_params["Cubemap Face Twist"] = float(cubemap_twist_deg)
                    p.extra_generation_params["Cubemap Polar Scale"] = float(cubemap_polar_scale)
                    p.extra_generation_params["Cubemap Polar Power"] = float(cubemap_polar_power)
                    p.extra_generation_params["Cubemap Swirl"] = float(cubemap_swirl_deg)
                    p.extra_generation_params["Cubemap Swirl Power"] = float(cubemap_swirl_power)
                    p.extra_generation_params["Cubemap grid_sample Interp"] = str(cubemap_grid_interp)
                    p.extra_generation_params["Cubemap grid_sample Padding"] = str(cubemap_grid_padding)
                    p.extra_generation_params["Cubemap Cache Quant"] = float(cubemap_cache_angle_quant)
                    p.extra_generation_params["Cubemap GeoAA Samples"] = int(cubemap_geoaa_samples)
                    p.extra_generation_params["Cubemap GeoAA Radius"] = float(cubemap_geoaa_radius)
                    p.extra_generation_params["Cubemap Antipode Mix"] = float(cubemap_antipode_strength)
            except Exception:
                pass

        # Cubemap resolution helper (optional)
        if use_cubemap and isinstance(cubemap_resolution_mode, str) and cubemap_resolution_mode.startswith("Face size"):
            # Interpret current width/height as a single face, expand to 3x2 net
            try:
                p.width = int(p.width) * 3
                p.height = int(p.height) * 2
            except Exception:
                pass

        # Спецрежимы (приоритет: Cubemap > Polar > Panorama > Anisotropic)
        if use_cubemap:
            mode_x = mode_y = MODE_CUBEMAP
        elif use_polar:
            mode_x = mode_y = MODE_POLAR
        elif use_panorama:
            mode_x = mode_y = MODE_PANORAMA
        elif use_anisotropic:
            mode_x = mode_y = MODE_ANISOTROPIC

        params = {
            'mode_x': mode_x,
            'mode_y': mode_y,
            'start_step': start_step,
            'end_step': end_step,
            'multires_enabled': multires,
            'multires_strategy': multires_strategy,  # NEW
            'multires_start': multires_start,  # NEW
            'multires_end': multires_end,  # NEW
            'multires_sharpness': multires_sharpness,  # NEW
            # NEW ENHANCED MODES parameters
            'voronoi_cells': voronoi_cells,
            'voronoi_seed': voronoi_seed,
            'perlin_strength': perlin_strength,
            'perlin_scale': perlin_scale,
            'fractal_iterations': fractal_iterations,
            'fractal_scale': fractal_scale,
            'adaptive_threshold': adaptive_threshold,
            'panorama_engine': panorama_engine,
            'panorama_yaw': panorama_yaw,
            'panorama_pitch': panorama_pitch,
            'panorama_roll': panorama_roll,
            'panorama_coord_mode': panorama_coord_mode,
            'panorama_polar_scale': panorama_polar_scale,
            'panorama_polar_power': panorama_polar_power,
            'panorama_twist_deg': panorama_twist_deg,
            'panorama_twist_power': panorama_twist_power,
            'panorama_swirl_deg': panorama_swirl_deg,
            'panorama_swirl_power': panorama_swirl_power,
            'panorama_pole_ease_power': panorama_pole_ease_power,
            'panorama_antipode_strength': panorama_antipode_strength,
            'panorama_pole_blur_strength': panorama_pole_blur_strength,
            'panorama_pole_blur_radius': panorama_pole_blur_radius,
            'panorama_pole_blur_power': panorama_pole_blur_power,
            'panorama_geoaa_samples': panorama_geoaa_samples,
            'panorama_geoaa_radius_px': panorama_geoaa_radius,
            'panorama_grid_interp': panorama_grid_interp,
            'panorama_grid_padding': panorama_grid_padding,
            'panorama_cache_angle_quant': panorama_cache_angle_quant,
            'blend_enabled': use_blend,
            'blend_strength': blend_strength,
            'blend_falloff': blend_falloff,  # NEW
            'blend_sharpness': blend_sharpness,  # NEW
            'blend_width': blend_width if blend_width > 0 else None,  # NEW
            'use_zoom': use_zoom,
            'zoom_factor': zoom_factor,
            'zoom_mode': zoom_mode,
            'blend_mode_type': blend_mode_type,
            'noise_strength': noise_strength,
            'zoom_pan_x': zoom_pan_x,           # <--- ВСТАВИТЬ ЭТО
            'zoom_pan_y': zoom_pan_y,           # <--- ВСТАВИТЬ ЭТО
            'spiral_rotation': spiral_rotation,   # <--- ВСТАВИТЬ ЭТО
            'spiral_direction': spiral_direction, # <--- ВСТАВИТЬ ЭТО
            'interp_mode': interp_mode,
            'zoom_in_fade': zoom_in_fade,
            'variance_correction': variance_correction,
            'auto_clamp_pan': auto_clamp_pan,
            'adaptive_noise_scale': adaptive_noise_scale,
            'debug_mode': debug_mode,
            'zoom_convergence': zoom_convergence,
            'zoom_depth_power': zoom_depth_power,
            'zoom_force_original_size': zoom_force_original_size,
            'cubemap_engine': cubemap_engine,
            'cubemap_pad_mode': cubemap_pad_mode,
            'cubemap_blend_width': cubemap_blend_width,
            'cubemap_blend_strength': cubemap_blend_strength,
            'cubemap_yaw': cubemap_yaw,
            'cubemap_pitch': cubemap_pitch,
            'cubemap_roll': cubemap_roll,
            'cubemap_coord_mode': cubemap_coord_mode,
            'cubemap_twist_deg': cubemap_twist_deg,
            'cubemap_polar_scale': cubemap_polar_scale,
            'cubemap_polar_power': cubemap_polar_power,
            'cubemap_swirl_deg': cubemap_swirl_deg,
            'cubemap_swirl_power': cubemap_swirl_power,
            'cubemap_grid_interp': cubemap_grid_interp,
            'cubemap_grid_padding': cubemap_grid_padding,
            'cubemap_cache_angle_quant': cubemap_cache_angle_quant,
            'cubemap_geoaa_samples': cubemap_geoaa_samples,
            'cubemap_geoaa_radius_px': cubemap_geoaa_radius,
            'cubemap_antipode_strength': cubemap_antipode_strength,
            'anisotropic_angle': aniso_angle,
            'anisotropic_angle2': (aniso_angle2 if (use_anisotropic and aniso_angle_mix < 1.0) else None),
            'anisotropic_angle_mix': aniso_angle_mix,
            'stereo_enabled': stereo_enabled,
            'stereo_eye': stereo_eye,
            'stereo_engine': stereo_engine,
            'stereo_output': stereo_output,
            'stereo_in_model': stereo_in_model,
            'stereo_separation': stereo_separation,
            'stereo_convergence': stereo_convergence,
            'stereo_depth_power': stereo_depth_power,
            'stereo_pole_power': stereo_pole_power,
            'tiling_disable_hr': tiling_disable_hr,
            'script_ref': self,
        }

        
        # Save stereo settings for postprocess
        try:
            self._stereo_settings = {
                'enabled': bool(stereo_enabled),
                'eye': str(stereo_eye),
                'engine': str(stereo_engine),
                'output': str(stereo_output),
                'separation': float(stereo_separation),
                'convergence': float(stereo_convergence),
                'depth_power': float(stereo_depth_power),
                'pole_power': float(stereo_pole_power),
            }
        except Exception:
            self._stereo_settings = {'enabled': False}

# Патчинг UNet
        unet = self.get_unet(p)
        if unet:
            self.restore_original(unet)
            count_unet = self.patch_conv_layers(unet, params)
            print(f"✓ Patched {count_unet} UNet Conv2d layers")

        # Патчинг VAE
        if patch_vae and hasattr(p.sd_model, 'first_stage_model'):
            vae = p.sd_model.first_stage_model
            count_vae = self.patch_conv_layers(vae, params)
            print(f"✓ Patched {count_vae} VAE Conv2d layers")

        mode_str = f"{mode_x}/{mode_y}"
        if stereo_enabled:
            mode_str += f" + Stereo({stereo_eye})"
        if use_blend:
            mode_str += " + Blend"
        if use_kohaku:
            mode_str += " + Kohaku"
        if enable_mirroring and (mirror_mode != 0 or x_pan != 0 or y_pan != 0):
            mode_str += " + LatentMirror"

        print(f"✓ Advanced Tiling v3.0: Mode={mode_str}, Steps={start_step}-{end_step}")

    # ====== ПАТЧИНГ СЛОЁВ ======

    def patch_conv_layers(self, module, params):
        """Патчинг с исправленным замыканием"""
        count = 0
        for layer in module.modules():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.kernel_size == (1, 1) or layer.kernel_size == 1:
                    continue

                if layer not in _ORIGINAL_METHODS_CACHE:
                    _ORIGINAL_METHODS_CACHE[layer] = layer._conv_forward

                def make_patched(mod):
                    def patched(input, weight, bias):
                        return custom_padding_forward(
                            input, weight, bias,
                            mod.stride, mod.padding, mod.dilation, mod.groups,
                            params
                        )
                    return patched

                layer._conv_forward = make_patched(layer)
                count += 1

        return count

    def get_unet(self, p):
        """Универсальная детекция UNet"""
        if hasattr(p.sd_model, 'forge_objects') and hasattr(p.sd_model.forge_objects, 'unet'):
            return p.sd_model.forge_objects.unet
        elif hasattr(p.sd_model, 'model') and hasattr(p.sd_model.model, 'diffusion_model'):
            return p.sd_model.model.diffusion_model
        return p.sd_model

    def postprocess(self, p, processed, *args):
        """Восстановление"""
        unet = self.get_unet(p)
        if unet:
            self.restore_original(unet)

        if hasattr(p.sd_model, 'first_stage_model'):
            self.restore_original(p.sd_model.first_stage_model)

        
        # Stereo postprocess output formatting (Side-by-Side by default)
        try:
            st = getattr(self, "_stereo_settings", None)
            if st and st.get("enabled", False):
                out_mode = str(st.get("output", "Side-by-Side"))
                imgs = list(getattr(processed, "images", []) or [])
                infotexts = getattr(processed, "infotexts", None)

                new_imgs = []
                new_infos = [] if infotexts is not None else None

                for idx_img, im in enumerate(imgs):
                    left, right = _make_stereo_outputs(im, st)

                    if out_mode == "Side-by-Side":
                        sbs = Image.new("RGB", (left.width + right.width, left.height))
                        sbs.paste(left, (0, 0))
                        sbs.paste(right, (left.width, 0))
                        new_imgs.append(sbs)
                        if new_infos is not None:
                            new_infos.append(infotexts[idx_img] if idx_img < len(infotexts) else "")
                    elif out_mode == "Left only":
                        new_imgs.append(left)
                        if new_infos is not None:
                            new_infos.append(infotexts[idx_img] if idx_img < len(infotexts) else "")
                    elif out_mode == "Right only":
                        new_imgs.append(right)
                        if new_infos is not None:
                            new_infos.append(infotexts[idx_img] if idx_img < len(infotexts) else "")
                    else:
                        # Both separate
                        new_imgs.extend([left, right])
                        if new_infos is not None:
                            base = infotexts[idx_img] if idx_img < len(infotexts) else ""
                            new_infos.extend([base + " | Stereo: left", base + " | Stereo: right"])

                processed.images = new_imgs
                if new_infos is not None:
                    processed.infotexts = new_infos
        except Exception as e:
            print(f"[Advanced Asymmetric Tiling] Stereo postprocess error: {e}")

        self._stereo_settings = None
        # Отключаем латентный миррор до следующего запуска
        self.run_callback = False
        self.tiling_is_hires = False

    def restore_original(self, module):
        """Восстановление оригинальных методов"""
        if not _ORIGINAL_METHODS_CACHE:
            return

        restored = 0
        for layer in list(_ORIGINAL_METHODS_CACHE.keys()):
            if hasattr(layer, '_conv_forward'):
                layer._conv_forward = _ORIGINAL_METHODS_CACHE[layer]
                restored += 1

        _ORIGINAL_METHODS_CACHE.clear()
        _MASK_CACHE.clear()
        _CUBEMAP_GRID_CACHE.clear()
        _PANO_GRID_CACHE.clear()

        if restored > 0:
            print(f"✓ Restored {restored} layers to original state")

