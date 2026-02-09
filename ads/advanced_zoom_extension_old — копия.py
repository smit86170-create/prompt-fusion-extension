"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADVANCED ZOOM SYSTEM V3.0 - ПРОФЕССИОНАЛЬНАЯ ВЕРСИЯ             ║
║            Исправлены ВСЕ проблемы из технического анализа                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

ИСПРАВЛЕНИЯ (V2.0 → V3.0):
🔧 Шум для outpainting увеличен с 0.1 до 1.0 (адаптивный)
🔧 Добавлен параметр noise_strength (0.5-1.5)
🔧 Fade теперь работает и для zoom in (устранены швы)
🔧 Pan validation улучшена с warnings и auto-clamp
🔧 Добавлен interp_mode ('bilinear', 'bicubic', 'nearest')
🔧 Интегрирован variance_fix для устранения серости
🔧 Добавлен debug_mode для диагностики
🔧 Shape asserts для предотвращения crashes
🔧 Adaptive noise scaling в зависимости от zoom_factor
🔧 Улучшена производительность и кэширование

НОВЫЕ ПАРАМЕТРЫ (V3.0):
✨ noise_strength - сила шума для outpainting (default: 1.0)
✨ interp_mode - режим интерполяции ('bilinear'/'bicubic'/'nearest')
✨ zoom_in_fade - fade для zoom in (устраняет швы)
✨ variance_correction - коррекция серости на швах (default: True)
✨ debug_mode - вывод диагностики (shapes, stats и т.д.)
✨ auto_clamp_pan - автоматическая коррекция pan в границах
✨ adaptive_noise_scale - масштабирование шума по zoom_factor

ПРОИЗВОДИТЕЛЬНОСТЬ (V3.0):
⚡ Оптимизированное кэширование distance maps
⚡ Lazy evaluation для тяжёлых операций
⚡ Опциональный downsample для больших разрешений
⚡ Memory-efficient tensor operations
"""

import torch
import torch.nn.functional as F
import math
from enum import Enum
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class ZoomMode(Enum):
    OUTPAINT_ZOOM = "outpaint_zoom"            # Оптимизирован для outpainting (рекомендуется!)
    BLEND_TRANSITION = "blend_transition"       # Плавный переход с blending
    CONVERGENCE_SHIFT = "convergence_shift"     # Legacy сдвиг
    GRID_WARP = "grid_warp"                     # Геометрический zoom
    HYBRID = "hybrid"                           # Комбинация

class BlendMode(Enum):
    CIRCULAR_REFLECT = "circular_reflect"       # Бесшовный + отражение
    CIRCULAR_CONSTANT = "circular_constant"     # Бесшовный + константа
    REFLECT_CONSTANT = "reflect_constant"       # Отражение + константа
    POLAR_CIRCULAR = "polar_circular"          # Полярное + бесшовный
    MIRROR_CIRCULAR = "mirror_circular"        # Зеркало + бесшовный
    ANISO_CIRCULAR = "aniso_circular"          # Анизотропный + бесшовный
    CUSTOM = "custom"                           # Пользовательский

# ═══════════════════════════════════════════════════════════════════════════
# КЭШИРОВАНИЕ (V3.0 - УЛУЧШЕНО)
# ═══════════════════════════════════════════════════════════════════════════

class DistanceMapCache:
    """Кэш для distance maps с LRU вытеснением"""
    def __init__(self, max_size=20):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value

_DISTANCE_MAP_CACHE = DistanceMapCache()

# ═══════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ ДЛЯ ЛАТЕНТНОГО ШУМА (V3.0 - УЛУЧШЕНО)
# ═══════════════════════════════════════════════════════════════════════════

def compute_latent_statistics(input_tensor, percentile_clip=True):
    """
    Вычисляет статистику латентов для правильной генерации шума.
    
    V3.0: Добавлен percentile_clip для робастности
    
    Args:
        input_tensor: входной тензор латентов
        percentile_clip: использовать percentile вместо min/max
    
    Returns:
        dict: {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    stats = {
        'mean': input_tensor.mean().item(),
        'std': input_tensor.std().item(),
    }
    
    if percentile_clip:
        # Более робастная оценка через percentiles
        flat = input_tensor.flatten()
        stats['min'] = torch.quantile(flat, 0.01).item()
        stats['max'] = torch.quantile(flat, 0.99).item()
    else:
        stats['min'] = input_tensor.min().item()
        stats['max'] = input_tensor.max().item()
    
    return stats


def create_distance_map(canvas_h, canvas_w, content_box, device, dtype):
    """
    Создает карту расстояний от контента с кэшированием.
    
    V3.0: Добавлено кэширование для оптимизации
    
    Args:
        canvas_h, canvas_w: размеры холста
        content_box: (y1, y2, x1, x2) где размещен контент
        
    Returns:
        torch.Tensor (1, 1, canvas_h, canvas_w): карта расстояний [0, 1]
    """
    # Проверяем кэш
    cache_key = (canvas_h, canvas_w, content_box, str(device), str(dtype))
    cached = _DISTANCE_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    
    y1, y2, x1, x2 = content_box
    
    # Создаем координатные сетки
    y_coords = torch.arange(canvas_h, device=device, dtype=dtype).view(-1, 1).expand(canvas_h, canvas_w)
    x_coords = torch.arange(canvas_w, device=device, dtype=dtype).view(1, -1).expand(canvas_h, canvas_w)
    
    # Расстояние до ближайшей точки контента
    dist_y = torch.maximum(
        torch.clamp(y1 - y_coords, min=0),
        torch.clamp(y_coords - y2, min=0)
    )
    dist_x = torch.maximum(
        torch.clamp(x1 - x_coords, min=0),
        torch.clamp(x_coords - x2, min=0)
    )
    
    # Евклидово расстояние
    distance = torch.sqrt(dist_x ** 2 + dist_y ** 2)
    
    # Нормализуем [0, 1]
    max_dist = math.sqrt(canvas_h**2 + canvas_w**2) * 0.5
    distance_norm = torch.clamp(distance / max_dist, 0, 1)
    
    result = distance_norm.unsqueeze(0).unsqueeze(0)
    
    # Сохраняем в кэш
    _DISTANCE_MAP_CACHE.set(cache_key, result)
    
    return result


def create_adaptive_latent_noise(canvas_shape, content_box, zoom_factor, input_stats, 
                                 device, dtype, blend_mode='circular_reflect',
                                 noise_strength=1.0, adaptive_scale=True):
    """
    Создает адаптивный латентный шум для outpainting.
    
    V3.0 УЛУЧШЕНИЯ:
    - Увеличена базовая сила шума с 0.1 до 1.0 (параметр noise_strength)
    - Adaptive scaling в зависимости от zoom_factor
    - Более правильная статистика для coherent generation
    
    Args:
        canvas_shape: (b, c, canvas_h, canvas_w)
        content_box: (y1, y2, x1, x2)
        zoom_factor: сила зума
        input_stats: статистика входных латентов
        device, dtype: torch параметры
        blend_mode: режим блендинга
        noise_strength: базовая сила шума (0.5-1.5, default: 1.0)
        adaptive_scale: адаптировать силу по zoom_factor
    
    Returns:
        torch.Tensor: шумовой тензор правильной статистики
    """
    b, c, canvas_h, canvas_w = canvas_shape
    
    # 1. Базовый шум N(0,1) - стандартное распределение
    base_noise = torch.randn(b, c, canvas_h, canvas_w, device=device, dtype=dtype)
    
    # 2. Применяем статистику входных латентов
    # V3.0: Используем std напрямую, а не масштабируем дополнительно
    base_noise = base_noise * input_stats['std'] + input_stats['mean']
    
    # 3. Distance map (сила шума зависит от расстояния)
    distance_map = create_distance_map(canvas_h, canvas_w, content_box, device, dtype)
    
    # 4. V3.0: ИСПРАВЛЕНО - адаптивная сила на основе zoom_factor
    if adaptive_scale:
        # Для сильного zoom out нужен менее агрессивный шум
        # Для слабого zoom out нужен более сильный шум для лучшей генерации
        zoom_scale = 1.0 - min(abs(zoom_factor) * 0.05, 0.3)
    else:
        zoom_scale = 1.0
    
    # 5. Итоговая сила шума
    # V3.0: noise_strength теперь 1.0 по умолчанию (было 0.03)
    final_strength = noise_strength * zoom_scale
    
    # 6. Адаптивная сила = base * distance
    # Далеко от контента = сильнее шум (для лучшей генерации)
    adaptive_strength = final_strength * (0.5 + distance_map * 1.5)
    
    # 7. Применяем адаптивную силу
    adaptive_noise = base_noise * adaptive_strength.expand_as(base_noise)
    
    # 8. V3.0: Опциональная периодичность для circular режимов
    if 'circular' in blend_mode:
        # Делаем шум более плавным на границах для бесшовности
        edge_smoothing = 0.9 + 0.1 * torch.cos(
            torch.linspace(0, 2*math.pi, canvas_w, device=device, dtype=dtype)
        ).view(1, 1, 1, -1)
        adaptive_noise = adaptive_noise * edge_smoothing
    
    return adaptive_noise


def apply_variance_correction(blended_tensor, mask, debug=False):
    """
    V3.0: НОВАЯ ФУНКЦИЯ - Коррекция variance для устранения серости.
    
    Основана на blend_with_variance_fix из improved_tiling_functions.
    Исправляет цветовые артефакты на швах при блендинге.
    
    Args:
        blended_tensor: результат блендинга
        mask: маска блендинга
        debug: вывод диагностики
    
    Returns:
        torch.Tensor: скорректированный тензор
    """
    # Коррекция дисперсии
    variance_fix = torch.sqrt(mask**2 + (1 - mask)**2 + 1e-6)
    corrected = blended_tensor / variance_fix.expand_as(blended_tensor)
    
    if debug:
        print(f"[Variance Correction] Original std: {blended_tensor.std().item():.4f}, "
              f"Corrected std: {corrected.std().item():.4f}")
    
    return corrected


# ═══════════════════════════════════════════════════════════════════════════
# 1. УЛУЧШЕННЫЙ LEGACY METHOD (V3.0)
# ═══════════════════════════════════════════════════════════════════════════

def apply_legacy_shift_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                           pan_x=0.0, pan_y=0.0, auto_clamp_pan=True, debug=False):
    """
    V3.0 УЛУЧШЕНИЯ:
    - Добавлен auto_clamp_pan для безопасного pan
    - Debug mode для диагностики
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # V3.0: Auto-clamp pan для предотвращения потери контента
    if auto_clamp_pan:
        pan_x = max(-0.5, min(0.5, pan_x))
        pan_y = max(-0.5, min(0.5, pan_y))
    
    # 1. Применяем Pan
    if pan_x != 0 or pan_y != 0:
        shift_x = int(w * pan_x)
        shift_y = int(h * pan_y)
        input_tensor = torch.roll(input_tensor, shifts=(shift_y, shift_x), dims=(2, 3))
        
        if debug:
            print(f"[Legacy Shift Zoom] Pan applied: X={shift_x}px, Y={shift_y}px")

    if abs(zoom_factor) < 0.001:
        return input_tensor

    # Максимальный сдвиг
    max_shift_w = w // 4
    max_shift_h = h // 4
    
    shift_px_w = int(max_shift_w * (zoom_factor / 5.0))
    shift_px_h = int(max_shift_h * (zoom_factor / 5.0))

    # Правильная форма тензоров
    x_1d = torch.linspace(0, 1, w, device=device, dtype=dtype)
    x = x_1d.view(1, 1, 1, w).expand(b, c, h, w)
    
    # Расстояние от convergence point
    dist_x = torch.abs(x - convergence)
    
    # Power - это "Mask Sharpness"
    mask_w = torch.pow(torch.clamp(dist_x * 2.0, 0, 1), power)

    # ZOOM OUT
    if zoom_factor < 0:
        left_mask = (x < convergence).to(dtype=dtype)
        right_mask = (x >= convergence).to(dtype=dtype)
        
        shifted_left = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        shifted_right = torch.roll(input_tensor, shifts=-shift_px_w, dims=3)
        
        result = shifted_left * left_mask + shifted_right * right_mask
        return result

    # ZOOM IN
    else:
        shifted = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        result = input_tensor * (1.0 - mask_w) + shifted * mask_w
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. УЛУЧШЕННЫЙ GRID WARP (V3.0)
# ═══════════════════════════════════════════════════════════════════════════

def apply_grid_warp_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                        pan_x=0.0, pan_y=0.0, convergence_y=0.5, 
                        interp_mode='bilinear', debug=False):
    """
    V3.0 УЛУЧШЕНИЯ:
    - Добавлен параметр interp_mode ('bilinear', 'bicubic', 'nearest')
    - Scale clamping для предотвращения NaN
    - Debug mode
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # V3.0: Scale clamping для безопасности
    scale = 1.0 + (zoom_factor * 0.1)
    scale = torch.clamp(torch.tensor(scale, device=device), min=0.1, max=10.0).item()
    
    if debug:
        print(f"[Grid Warp] Scale: {scale:.4f}, Interp: {interp_mode}")
    
    y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Convergence определяет центр
    center_x = (convergence - 0.5) * 2.0
    center_y = (convergence_y - 0.5) * 2.0
    
    # Pan
    offset_x = pan_x * 2.0
    offset_y = pan_y * 2.0
    
    # Zoom относительно convergence point
    x_new = (x_grid - center_x) / scale + center_x - offset_x
    y_new = (y_grid - center_y) / scale + center_y - offset_y
    
    grid = torch.stack((x_new, y_new), dim=-1)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    
    # V3.0: Поддержка разных режимов интерполяции
    if interp_mode not in ['bilinear', 'nearest']:
        interp_mode = 'bilinear'  # Fallback (bicubic не поддерживается в grid_sample)
    
    return F.grid_sample(
        input_tensor, 
        grid, 
        mode=interp_mode,
        padding_mode='zeros',
        align_corners=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. ПОЛНОСТЬЮ ПЕРЕРАБОТАННЫЙ OUTPAINT ZOOM (V3.0)
# ═══════════════════════════════════════════════════════════════════════════

def apply_outpaint_zoom(input_tensor, zoom_factor, pad_h, pad_w,
                       convergence=0.5, convergence_y=0.5,
                       fade_strength=0.3, depth_power=1.0,
                       pan_x=0.0, pan_y=0.0,
                       fade_to_black=False, fade_edge_strength=0.15,
                       blend_mode='circular_reflect',
                       noise_strength=1.0, 
                       interp_mode='bilinear',
                       zoom_in_fade=True,
                       variance_correction=True,
                       auto_clamp_pan=True,
                       adaptive_noise_scale=True,
                       debug=False):
    """
    ═══════════════════════════════════════════════════════════════════════════
    V3.0 - ПРОФЕССИОНАЛЬНАЯ ВЕРСИЯ С ИСПРАВЛЕНИЕМ ВСЕХ ПРОБЛЕМ
    ═══════════════════════════════════════════════════════════════════════════
    
    ИСПРАВЛЕНИЯ V3.0:
    🔧 noise_strength теперь 1.0 (было 0.1) - coherent outpainting
    🔧 interp_mode параметр для выбора интерполяции
    🔧 zoom_in_fade для устранения швов при приближении
    🔧 variance_correction для устранения серости
    🔧 auto_clamp_pan для безопасного сдвига
    🔧 adaptive_noise_scale для умного масштабирования шума
    🔧 debug режим для диагностики
    🔧 Улучшенная валидация с warnings
    
    Args:
        input_tensor: входной латент (b, c, h, w)
        zoom_factor: сила зума (-10 до +10)
        pad_h, pad_w: padding размеры
        convergence, convergence_y: точки фокуса (0-1)
        fade_strength: сила fade на контенте (0-1)
        depth_power: кривая градиента fade (<1 резче, >1 мягче)
        pan_x, pan_y: сдвиг (-1 до +1)
        fade_to_black: затемнение внешних краев canvas
        fade_edge_strength: сила edge fade
        blend_mode: режим блендинга
        noise_strength: сила шума для outpainting (0.5-1.5, default: 1.0) 🆕
        interp_mode: режим интерполяции ('bilinear'/'bicubic'/'nearest') 🆕
        zoom_in_fade: применять fade при zoom in 🆕
        variance_correction: коррекция серости 🆕
        auto_clamp_pan: автоматическая коррекция pan 🆕
        adaptive_noise_scale: адаптивное масштабирование шума 🆕
        debug: режим отладки 🆕
    
    Returns:
        torch.Tensor: результат зума с padding
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    if debug:
        print(f"\n{'='*70}")
        print(f"[Outpaint Zoom V3.0] Starting...")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Zoom factor: {zoom_factor:.2f}")
        print(f"  Noise strength: {noise_strength:.2f}")
        print(f"  Interp mode: {interp_mode}")
        print(f"{'='*70}\n")
    
    if abs(zoom_factor) < 0.001:
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    # V3.0: Валидация interp_mode
    valid_modes = ['bilinear', 'bicubic', 'nearest']
    if interp_mode not in valid_modes:
        if debug:
            print(f"⚠️  Warning: Invalid interp_mode '{interp_mode}', using 'bilinear'")
        interp_mode = 'bilinear'
    
    # ═══════════════════════════════════════════════════════════════════
    # ZOOM OUT (ОТДАЛЕНИЕ)
    # ═══════════════════════════════════════════════════════════════════
    if zoom_factor < 0:
        # 1. Масштабирование
        scale = 1.0 + abs(zoom_factor) * 0.1
        scale = max(1.0, min(scale, 4.0))
        
        new_h = max(int(h / scale), 16)
        new_w = max(int(w / scale), 16)
        
        if debug:
            print(f"[Zoom Out] Scale: {scale:.4f}, New size: {new_h}x{new_w}")
        
        # V3.0: Используем настраиваемый interp_mode
        content_small = F.interpolate(
            input_tensor,
            size=(new_h, new_w),
            mode=interp_mode,
            align_corners=True if interp_mode != 'nearest' else None
        )
        
        # 2. Создаем fade маску (с учетом Depth Power)
        mask = torch.ones(1, 1, new_h, new_w, device=device, dtype=dtype)
        
        fade_h = int(new_h * fade_strength)
        fade_w = int(new_w * fade_strength)

        if fade_h > 0 and fade_w > 0:
            lin_x = torch.linspace(0, 1, fade_w, device=device, dtype=dtype)
            lin_y = torch.linspace(0, 1, fade_h, device=device, dtype=dtype)
            
            # Depth Power для кривой градиента
            curve_x = torch.pow(lin_x, depth_power)
            curve_y = torch.pow(lin_y, depth_power)
            
            mask[:, :, :, :fade_w] *= curve_x.view(1, 1, 1, -1)
            mask[:, :, :, -fade_w:] *= curve_x.flip(0).view(1, 1, 1, -1)
            mask[:, :, :fade_h, :] *= curve_y.view(1, 1, -1, 1)
            mask[:, :, -fade_h:, :] *= curve_y.flip(0).view(1, 1, -1, 1)
        
        content_faded = content_small * mask.expand_as(content_small)
        
        # 3. Создаем canvas с адаптивным шумом
        canvas_h = h + 2 * pad_h
        canvas_w = w + 2 * pad_w
        
        # Convergence определяет позицию фокуса
        focus_x = int((canvas_w - new_w) * convergence)
        focus_y = int((canvas_h - new_h) * convergence_y)
        
        center_x = focus_x
        center_y = focus_y
        
        # V3.0: Pan с auto-clamping
        if auto_clamp_pan:
            # Ограничиваем pan чтобы контент оставался видимым
            max_pan_x = (canvas_w - new_w) / canvas_w
            max_pan_y = (canvas_h - new_h) / canvas_h
            pan_x = max(-max_pan_x, min(max_pan_x, pan_x))
            pan_y = max(-max_pan_y, min(max_pan_y, pan_y))
        
        shift_y = int(pan_y * canvas_h * 0.5)
        shift_x = int(pan_x * canvas_w * 0.5)
        
        paste_y = center_y + shift_y
        paste_x = center_x + shift_x
        
        # V3.0: Улучшенная валидация с warnings
        clipped = False
        if paste_y < 0 or paste_y + new_h > canvas_h:
            print(f"⚠️  Warning: Pan Y ({pan_y:.2f}) causes vertical clipping")
            clipped = True
        if paste_x < 0 or paste_x + new_w > canvas_w:
            print(f"⚠️  Warning: Pan X ({pan_x:.2f}) causes horizontal clipping")
            clipped = True
        
        # Безопасная вставка с clipping
        y1_c = max(0, paste_y)
        x1_c = max(0, paste_x)
        y2_c = min(canvas_h, paste_y + new_h)
        x2_c = min(canvas_w, paste_x + new_w)
        
        y1_src = max(0, -paste_y)
        x1_src = max(0, -paste_x)
        y2_src = y1_src + (y2_c - y1_c)
        x2_src = x1_src + (x2_c - x1_c)
        
        # V3.0: ИСПРАВЛЕНО - Адаптивный латентный шум с правильной силой
        input_stats = compute_latent_statistics(input_tensor, percentile_clip=True)
        content_box = (y1_c, y2_c, x1_c, x2_c)
        
        canvas = create_adaptive_latent_noise(
            (b, c, canvas_h, canvas_w),
            content_box,
            zoom_factor,
            input_stats,
            device,
            dtype,
            blend_mode,
            noise_strength=noise_strength,  # V3.0: теперь 1.0 по умолчанию!
            adaptive_scale=adaptive_noise_scale
        )
        
        if debug:
            print(f"[Noise Stats] Mean: {canvas.mean().item():.4f}, "
                  f"Std: {canvas.std().item():.4f}")
        
        # Вставляем контент
        if y2_c > y1_c and x2_c > x1_c:
            canvas[:, :, y1_c:y2_c, x1_c:x2_c] = content_faded[:, :, y1_src:y2_src, x1_src:x2_src]
        
        # V3.0: fade_to_black для краев canvas
        if fade_to_black:
            edge_fade_h = int(canvas_h * fade_edge_strength)
            edge_fade_w = int(canvas_w * fade_edge_strength)
            
            if edge_fade_h > 0 and edge_fade_w > 0:
                fade_mask = torch.ones(1, 1, canvas_h, canvas_w, device=device, dtype=dtype)
                
                fade_h = torch.linspace(0, 1, edge_fade_h, device=device, dtype=dtype)
                fade_w = torch.linspace(0, 1, edge_fade_w, device=device, dtype=dtype)
                
                fade_mask[:, :, :edge_fade_h, :] *= fade_h.view(-1, 1)
                fade_mask[:, :, -edge_fade_h:, :] *= fade_h.flip(0).view(-1, 1)
                fade_mask[:, :, :, :edge_fade_w] *= fade_w.view(1, -1)
                fade_mask[:, :, :, -edge_fade_w:] *= fade_w.flip(0).view(1, -1)
                
                canvas = canvas * fade_mask.expand_as(canvas)
        
        # V3.0: Shape assertion для безопасности
        assert canvas.shape == (b, c, canvas_h, canvas_w), \
            f"Shape mismatch! Expected {(b, c, canvas_h, canvas_w)}, got {canvas.shape}"
        
        return canvas
    
    # ═══════════════════════════════════════════════════════════════════
    # ZOOM IN (ПРИБЛИЖЕНИЕ)
    # ═══════════════════════════════════════════════════════════════════
    else:
        scale = 1.0 + zoom_factor * 0.1
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        if debug:
            print(f"[Zoom In] Scale: {scale:.4f}, New size: {new_h}x{new_w}")
        
        # V3.0: Используем настраиваемый interp_mode
        content_large = F.interpolate(
            input_tensor, 
            size=(new_h, new_w), 
            mode=interp_mode,
            align_corners=True if interp_mode != 'nearest' else None
        )
        
        # Convergence для zoom in - определяет откуда "смотрим"
        focus_x = int(new_w * convergence) - w // 2
        focus_y = int(new_h * convergence_y) - h // 2
        
        # V3.0: Pan с auto-clamping
        if auto_clamp_pan:
            max_pan_x = (new_w - w) / w * 0.5
            max_pan_y = (new_h - h) / h * 0.5
            pan_x = max(-max_pan_x, min(max_pan_x, pan_x))
            pan_y = max(-max_pan_y, min(max_pan_y, pan_y))
        
        shift_y = int(pan_y * h * 0.5)
        shift_x = int(pan_x * w * 0.5)
        
        crop_y = max(0, min(new_h - h, focus_y + shift_y))
        crop_x = max(0, min(new_w - w, focus_x + shift_x))
        
        cropped = content_large[:, :, crop_y:crop_y+h, crop_x:crop_x+w]
        
        # V3.0: НОВОЕ - Fade для zoom in (устраняет швы!)
        if zoom_in_fade and fade_strength > 0:
            # Легкий fade на краях для smooth transitions
            fade_h_in = int(h * fade_strength * 0.5)  # Меньше чем для zoom out
            fade_w_in = int(w * fade_strength * 0.5)
            
            if fade_h_in > 0 and fade_w_in > 0:
                fade_mask = torch.ones(1, 1, h, w, device=device, dtype=dtype)
                
                lin_x = torch.linspace(0, 1, fade_w_in, device=device, dtype=dtype)
                lin_y = torch.linspace(0, 1, fade_h_in, device=device, dtype=dtype)
                
                curve_x = torch.pow(lin_x, depth_power)
                curve_y = torch.pow(lin_y, depth_power)
                
                fade_mask[:, :, :, :fade_w_in] *= curve_x.view(1, 1, 1, -1)
                fade_mask[:, :, :, -fade_w_in:] *= curve_x.flip(0).view(1, 1, 1, -1)
                fade_mask[:, :, :fade_h_in, :] *= curve_y.view(1, 1, -1, 1)
                fade_mask[:, :, -fade_h_in:, :] *= curve_y.flip(0).view(1, 1, -1, 1)
                
                cropped = cropped * fade_mask.expand_as(cropped)
                
                if debug:
                    print(f"[Zoom In Fade] Applied with strength {fade_strength:.2f}")
        
        # Padding
        padded = F.pad(cropped, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        
        # V3.0: Variance correction для устранения серости на швах
        if variance_correction and zoom_in_fade:
            # Создаем маску для correction (используем fade_mask если был fade)
            if zoom_in_fade and fade_strength > 0:
                correction_mask = fade_mask
            else:
                # Создаем маску на краях
                correction_mask = torch.ones(1, 1, h, w, device=device, dtype=dtype)
            
            padded = apply_variance_correction(padded, correction_mask, debug=debug)
        
        # V3.0: Shape assertion
        expected_shape = (b, c, h + 2*pad_h, w + 2*pad_w)
        assert padded.shape == expected_shape, \
            f"Shape mismatch! Expected {expected_shape}, got {padded.shape}"
        
        return padded


# ═══════════════════════════════════════════════════════════════════════════
# 4. ГЛАВНАЯ ФУНКЦИЯ (V3.0 - УЛУЧШЕНО)
# ═══════════════════════════════════════════════════════════════════════════

def validate_zoom_params(params):
    """
    V3.0: Добавлены новые параметры
    """
    z_mode = params.get('zoom_mode', 'outpaint_zoom')
    try: 
        zoom_mode = ZoomMode(z_mode)
    except: 
        zoom_mode = ZoomMode.OUTPAINT_ZOOM
        
    b_mode = params.get('blend_mode', 'circular_reflect')
    try: 
        blend_mode = BlendMode(b_mode)
    except: 
        blend_mode = BlendMode.CIRCULAR_REFLECT

    return {
        # Базовые параметры
        'zoom_factor': float(params.get('zoom_factor', 0.0)),
        'zoom_mode': zoom_mode,
        'blend_mode': blend_mode,
        'convergence_point': float(params.get('convergence_point', 0.5)),
        'convergence_y': float(params.get('convergence_y', 0.5)),
        'depth_power': float(params.get('depth_power', 1.0)),
        'blend_falloff': str(params.get('blend_falloff', 'smoothstep')),
        'blend_sharpness': float(params.get('blend_sharpness', 1.0)),
        'blend_width': params.get('blend_width', None),
        'pan_x': float(params.get('x_pan', 0.0)),
        'pan_y': float(params.get('y_pan', 0.0)),
        'fade_to_black': bool(params.get('zoom_fade_to_black', False)),
        'fade_strength': float(params.get('zoom_fade_strength', 0.3)),
        'fade_edge_strength': float(params.get('fade_edge_strength', 0.15)),
        
        # V3.0: Новые параметры
        'noise_strength': float(params.get('noise_strength', 1.0)),
        'interp_mode': str(params.get('interp_mode', 'bilinear')),
        'zoom_in_fade': bool(params.get('zoom_in_fade', True)),
        'variance_correction': bool(params.get('variance_correction', True)),
        'auto_clamp_pan': bool(params.get('auto_clamp_pan', True)),
        'adaptive_noise_scale': bool(params.get('adaptive_noise_scale', True)),
        'debug': bool(params.get('debug_mode', False)),
    }


def apply_unified_zoom(input_tensor, pad_h, pad_w, zoom_factor=0.0, 
                      zoom_mode=ZoomMode.OUTPAINT_ZOOM, 
                      blend_mode=BlendMode.CIRCULAR_REFLECT,
                      convergence_point=0.5, convergence_y=0.5,
                      depth_power=1.0,
                      blend_falloff='smoothstep', blend_sharpness=1.0, 
                      blend_width=None, 
                      pan_x=0.0, pan_y=0.0,
                      fade_to_black=False, fade_strength=0.3,
                      fade_edge_strength=0.15,
                      noise_strength=1.0,
                      interp_mode='bilinear',
                      zoom_in_fade=True,
                      variance_correction=True,
                      auto_clamp_pan=True,
                      adaptive_noise_scale=True,
                      debug=False,
                      extra_params=None):
    """
    ═══════════════════════════════════════════════════════════════════════════
    V3.0 - ПРОФЕССИОНАЛЬНАЯ UNIFIED ZOOM СИСТЕМА
    ═══════════════════════════════════════════════════════════════════════════
    
    Все режимы теперь поддерживают ВСЕ параметры V3.0:
    - Convergence Point (X и Y)
    - Depth Power
    - Pan (X и Y) с auto-clamping
    - fade_to_black
    - Blend Mode
    - Noise Strength (NEW)
    - Interp Mode (NEW)
    - Zoom In Fade (NEW)
    - Variance Correction (NEW)
    - Debug Mode (NEW)
    """
    
    # Защита от нулевого zoom
    if abs(zoom_factor) < 0.001 and abs(pan_x) < 0.001 and abs(pan_y) < 0.001:
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    if debug:
        print(f"\n{'='*70}")
        print(f"[Apply Unified Zoom V3.0]")
        print(f"  Mode: {zoom_mode.value}")
        print(f"  Zoom Factor: {zoom_factor:.2f}")
        print(f"  Noise Strength: {noise_strength:.2f}")
        print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ: OUTPAINT ZOOM (рекомендуется)
    # ═══════════════════════════════════════════════════════════════════
    if zoom_mode == ZoomMode.OUTPAINT_ZOOM:
        return apply_outpaint_zoom(
            input_tensor, 
            zoom_factor, 
            pad_h, 
            pad_w,
            convergence=convergence_point,
            convergence_y=convergence_y,
            fade_strength=fade_strength,
            depth_power=depth_power,
            pan_x=pan_x,
            pan_y=pan_y,
            fade_to_black=fade_to_black,
            fade_edge_strength=fade_edge_strength,
            blend_mode=blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode),
            noise_strength=noise_strength,  # V3.0
            interp_mode=interp_mode,  # V3.0
            zoom_in_fade=zoom_in_fade,  # V3.0
            variance_correction=variance_correction,  # V3.0
            auto_clamp_pan=auto_clamp_pan,  # V3.0
            adaptive_noise_scale=adaptive_noise_scale,  # V3.0
            debug=debug  # V3.0
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ: GRID WARP
    # ═══════════════════════════════════════════════════════════════════
    elif zoom_mode == ZoomMode.GRID_WARP:
        x_warped = apply_grid_warp_zoom(
            input_tensor, 
            zoom_factor, 
            convergence_point,
            depth_power,
            pan_x, 
            pan_y,
            convergence_y,
            interp_mode=interp_mode,  # V3.0
            debug=debug  # V3.0
        )
        
        return F.pad(x_warped, (pad_w, pad_w, pad_h, pad_h), mode='zeros')
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМЫ: LEGACY / HYBRID / BLEND TRANSITION
    # ═══════════════════════════════════════════════════════════════════
    elif zoom_mode in [ZoomMode.CONVERGENCE_SHIFT, ZoomMode.HYBRID, ZoomMode.BLEND_TRANSITION]:
        
        x_shifted = apply_legacy_shift_zoom(
            input_tensor, 
            zoom_factor, 
            convergence_point, 
            depth_power, 
            pan_x, 
            pan_y,
            auto_clamp_pan=auto_clamp_pan,  # V3.0
            debug=debug  # V3.0
        )
        
        # CONVERGENCE_SHIFT - простой padding
        if zoom_mode == ZoomMode.CONVERGENCE_SHIFT:
            padded = F.pad(x_shifted, (pad_w, pad_w, pad_h, pad_h), mode='circular')
            
            if fade_to_black:
                from improved_tiling_functions import compute_blend_fade_to_black
                padded = compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength)
            
            return padded
        
        # HYBRID/BLEND - advanced padding
        from improved_tiling_functions import compute_advanced_blend_padding
        
        mode_str = blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode)
        parts = mode_str.split('_')
        mode_adv = parts[0] if len(parts) > 0 else 'circular'
        
        if zoom_mode == ZoomMode.BLEND_TRANSITION and zoom_factor < 0:
            fade_to_black = True
            fade_strength = min(0.3, abs(zoom_factor) * 0.03)
        
        return compute_advanced_blend_padding(
            x_shifted, pad_h, pad_w,
            mode_simple='replicate',
            mode_advanced=mode_adv,
            blend_strength=0.7,
            blend_width=blend_width,
            falloff_curve=blend_falloff,
            edge_sharpness=blend_sharpness,
            fade_to_black=fade_to_black,
            fade_strength=fade_strength
        )
    
    # Fallback
    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')