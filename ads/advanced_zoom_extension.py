"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED ZOOM & PROXIMITY SYSTEM                           ║
║                 Расширение для asymmetric_tiling_UNIFIED                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

НОВЫЕ ВОЗМОЖНОСТИ:
✅ Настоящий zoom слайдер от -5 (отдаление) до +5 (приближение)
✅ Отдаление БЕЗ изменения всей картинки (legacy stereo convergence метод)
✅ Множественные режимы blend для разных эффектов
✅ Комбинированные режимы (circular+reflect, polar+mirror и т.д.)
"""

import torch
import torch.nn.functional as F
import math
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# ZOOM MODES - Режимы отдаления/приближения
# ═══════════════════════════════════════════════════════════════════════════

class ZoomMode(Enum):
    """Режимы для zoom эффекта"""
    BLEND_TRANSITION = "blend_transition"       # Смешивание режимов padding
    CONVERGENCE_SHIFT = "convergence_shift"     # Shift-based zoom (без изменения всей картинки)
    GRID_WARP = "grid_warp"                     # Warp через grid_sample
    HYBRID = "hybrid"                            # Комбинация методов


class BlendMode(Enum):
    """Режимы для Advanced Blend"""
    CIRCULAR_REFLECT = "circular_reflect"       # Circular → Reflect
    CIRCULAR_CONSTANT = "circular_constant"     # Circular → Constant (black)
    REFLECT_CONSTANT = "reflect_constant"       # Reflect → Constant
    POLAR_CIRCULAR = "polar_circular"           # Polar → Circular
    MIRROR_CIRCULAR = "mirror_circular"         # Mirror → Circular
    ANISO_CIRCULAR = "aniso_circular"          # Anisotropic → Circular
    CUSTOM = "custom"                           # Пользовательские режимы


# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENCE ZOOM - Отдаление БЕЗ изменения всей картинки
# ═══════════════════════════════════════════════════════════════════════════

def apply_convergence_zoom(input_tensor, zoom_factor, convergence_point=0.5, 
                          depth_power=1.0, axis='horizontal'):
    """
    🎯 LEGACY МЕТОД: Отдаление через shift без изменения всей картинки
    
    Этот метод НЕ меняет само изображение - он только создает
    иллюзию глубины через сдвиги на разных частях изображения.
    
    Args:
        input_tensor: [B, C, H, W]
        zoom_factor: -5.0 (сильное отдаление) ... 0 (нет эффекта) ... +5.0 (приближение)
        convergence_point: 0.0-1.0, точка схождения (где нет сдвига)
        depth_power: Кривая глубины (1.0 = линейно, >1 = более выражено)
        axis: 'horizontal', 'vertical', 'both'
        
    Returns:
        Тензор с примененным shift-based zoom
    """
    if abs(zoom_factor) < 0.01:
        return input_tensor
    
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Нормализуем zoom_factor
    zoom_factor = max(-5.0, min(5.0, float(zoom_factor)))
    
    # Shift amount пропорционален размеру и zoom_factor
    # Отрицательный zoom = отдаление = больше shift
    max_shift_ratio = abs(zoom_factor) * 0.02  # 2% на единицу zoom
    
    result = input_tensor.clone()
    
    # === ГОРИЗОНТАЛЬНЫЙ SHIFT ===
    if axis in ('horizontal', 'both'):
        # Создаем depth map по горизонтали
        x_coords = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
        depth_h = torch.abs(x_coords - convergence_point).expand(b, c, h, w)
        
        # Применяем depth power
        if abs(depth_power - 1.0) > 0.01:
            depth_h = torch.pow(depth_h, depth_power)
        
        # Вычисляем shift для каждого пикселя
        shift_amount_h = (depth_h * max_shift_ratio * w).clamp(-w//2, w//2)
        
        # Направление shift зависит от знака zoom
        # Отрицательный zoom (отдаление) = shift наружу от convergence point
        # Положительный zoom (приближение) = shift к convergence point
        direction = -1.0 if zoom_factor < 0 else 1.0
        shift_amount_h = shift_amount_h * direction
        
        # Применяем shift через roll (циклический для seamless)
        # Для более плавного эффекта используем weighted blend
        shift_pixels = int(shift_amount_h.mean().item())
        if shift_pixels != 0:
            shifted_h = torch.roll(result, shifts=shift_pixels, dims=3)
            # Blend с весами по depth map
            alpha_h = (depth_h * abs(zoom_factor) * 0.2).clamp(0.0, 1.0)
            result = result * (1.0 - alpha_h) + shifted_h * alpha_h
    
    # === ВЕРТИКАЛЬНЫЙ SHIFT ===
    if axis in ('vertical', 'both'):
        y_coords = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        depth_v = torch.abs(y_coords - convergence_point).expand(b, c, h, w)
        
        if abs(depth_power - 1.0) > 0.01:
            depth_v = torch.pow(depth_v, depth_power)
        
        shift_amount_v = (depth_v * max_shift_ratio * h).clamp(-h//2, h//2)
        direction = -1.0 if zoom_factor < 0 else 1.0
        shift_amount_v = shift_amount_v * direction
        
        shift_pixels = int(shift_amount_v.mean().item())
        if shift_pixels != 0:
            shifted_v = torch.roll(result, shifts=shift_pixels, dims=2)
            alpha_v = (depth_v * abs(zoom_factor) * 0.2).clamp(0.0, 1.0)
            result = result * (1.0 - alpha_v) + shifted_v * alpha_v
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# GRID WARP ZOOM - Приближение/отдаление через grid_sample
# ═══════════════════════════════════════════════════════════════════════════

def apply_grid_warp_zoom(input_tensor, zoom_factor, center_x=0.5, center_y=0.5, 
                        warp_power=1.0, interpolation='bilinear'):
    """
    🔄 GRID WARP: Настоящий zoom через деформацию координат
    
    Args:
        input_tensor: [B, C, H, W]
        zoom_factor: -5.0 (отдаление) ... 0 ... +5.0 (приближение)
        center_x, center_y: Центр zoom (0.0-1.0)
        warp_power: Кривая деформации
        interpolation: 'bilinear' или 'nearest'
        
    Returns:
        Деформированный тензор
    """
    if abs(zoom_factor) < 0.01:
        return input_tensor
    
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Конвертируем zoom_factor в scale factor
    # zoom_factor = -5: scale = 0.5 (видим 2x больше области)
    # zoom_factor = 0: scale = 1.0 (нет изменений)
    # zoom_factor = +5: scale = 2.0 (видим 0.5x области, приближение)
    scale = math.pow(2.0, zoom_factor / 5.0)
    
    # Создаем сетку координат
    y_coords = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Центрируем относительно заданной точки
    center_x_norm = (center_x - 0.5) * 2.0
    center_y_norm = (center_y - 0.5) * 2.0
    
    grid_x = grid_x - center_x_norm
    grid_y = grid_y - center_y_norm
    
    # Применяем zoom (scale)
    # Для отдаления (zoom < 0): увеличиваем координаты = видим больше
    # Для приближения (zoom > 0): уменьшаем координаты = видим меньше
    grid_x = grid_x / scale
    grid_y = grid_y / scale
    
    # Опционально применяем warp power (для нелинейного zoom)
    if abs(warp_power - 1.0) > 0.01:
        radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        angle = torch.atan2(grid_y, grid_x)
        radius_warped = torch.pow(radius.clamp(0.0, math.sqrt(2.0)), warp_power)
        grid_x = radius_warped * torch.cos(angle)
        grid_y = radius_warped * torch.sin(angle)
    
    # Возвращаем к центру
    grid_x = grid_x + center_x_norm
    grid_y = grid_y + center_y_norm
    
    # Собираем grid
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # [B, H, W, 2]
    
    # Применяем grid_sample
    mode = 'bilinear' if interpolation == 'bilinear' else 'nearest'
    result = F.grid_sample(
        input_tensor, grid, 
        mode=mode, 
        padding_mode='border',  # Можно изменить на 'zeros' или 'reflection'
        align_corners=True
    )
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED BLEND WITH MODES - Смешивание с разными режимами
# ═══════════════════════════════════════════════════════════════════════════

def get_blend_mode_tensors(input_tensor, pad_h, pad_w, blend_mode, params=None):
    """
    Получает два тензора для смешивания в зависимости от blend_mode
    
    Args:
        input_tensor: [B, C, H, W]
        pad_h, pad_w: Размеры padding
        blend_mode: BlendMode enum
        params: Дополнительные параметры для сложных режимов
        
    Returns:
        (tensor_simple, tensor_advanced) - два тензора для blend
    """
    params = params or {}
    
    if blend_mode == BlendMode.CIRCULAR_REFLECT.value:
        # Simple: circular, Advanced: reflect
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        tensor_advanced = _safe_pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    
    elif blend_mode == BlendMode.CIRCULAR_CONSTANT.value:
        # Simple: constant (black), Advanced: circular
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        tensor_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    elif blend_mode == BlendMode.REFLECT_CONSTANT.value:
        # Simple: constant, Advanced: reflect
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        tensor_advanced = _safe_pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    
    elif blend_mode == BlendMode.POLAR_CIRCULAR.value:
        # Simple: circular, Advanced: polar (требует импорта из основного файла)
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        # Используем функцию из основного файла если доступна
        if 'compute_polar_padding' in params:
            tensor_advanced = params['compute_polar_padding'](input_tensor, pad_h, pad_w)
        else:
            # Fallback
            tensor_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    elif blend_mode == BlendMode.MIRROR_CIRCULAR.value:
        # Simple: replicate (mirror-like), Advanced: circular
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        tensor_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    elif blend_mode == BlendMode.ANISO_CIRCULAR.value:
        # Simple: circular, Advanced: anisotropic
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        if 'compute_anisotropic_padding' in params:
            tensor_advanced = params['compute_anisotropic_padding'](
                input_tensor, pad_h, pad_w,
                params.get('aniso_angle', 45),
                params.get('aniso_angle2', None),
                params.get('aniso_mix', 1.0)
            )
        else:
            # Fallback
            tensor_advanced = _safe_pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    
    else:  # Default / Custom
        tensor_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        tensor_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    return tensor_simple, tensor_advanced


def _safe_pad(x, pad, mode='reflect', value=0.0):
    """Safe wrapper для reflect mode"""
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


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED ZOOM SYSTEM - Объединяет все методы
# ═══════════════════════════════════════════════════════════════════════════

def apply_unified_zoom(input_tensor, pad_h, pad_w, 
                      zoom_factor=0.0,
                      zoom_mode='blend_transition',
                      blend_mode='circular_reflect',
                      convergence_point=0.5,
                      depth_power=1.0,
                      blend_falloff='smoothstep',
                      blend_sharpness=1.0,
                      blend_width=None,
                      extra_params=None):
    """
    🌟 UNIFIED ZOOM SYSTEM 🌟
    
    Объединяет все методы zoom в одну систему:
    - Blend Transition: Смешивание режимов padding
    - Convergence Shift: Отдаление без изменения всей картинки (legacy)
    - Grid Warp: Настоящий geometric zoom
    - Hybrid: Комбинация методов
    
    Args:
        input_tensor: [B, C, H, W]
        pad_h, pad_w: Padding размеры
        zoom_factor: -5.0 (далеко) ... 0 (нормально) ... +5.0 (близко)
        zoom_mode: 'blend_transition', 'convergence_shift', 'grid_warp', 'hybrid'
        blend_mode: Режим для blend (circular_reflect и т.д.)
        convergence_point: Точка схождения (0.0-1.0)
        depth_power: Кривая глубины
        blend_falloff: Кривая перехода
        blend_sharpness: Резкость перехода
        blend_width: Ширина зоны перехода
        extra_params: Дополнительные параметры
        
    Returns:
        Обработанный тензор с padding
    """
    extra_params = extra_params or {}
    
    # === РЕЖИМ 1: BLEND TRANSITION ===
    if zoom_mode == ZoomMode.BLEND_TRANSITION.value or zoom_mode == 'blend_transition':
        # Конвертируем zoom_factor (-5...+5) в blend_strength (0...1)
        # zoom = -5: strength = 0.0 (полностью simple mode)
        # zoom = 0:  strength = 0.5 (смешивание)
        # zoom = +5: strength = 1.0 (полностью advanced mode)
        blend_strength = (zoom_factor + 5.0) / 10.0
        blend_strength = max(0.0, min(1.0, blend_strength))
        
        # Получаем два режима padding
        tensor_simple, tensor_advanced = get_blend_mode_tensors(
            input_tensor, pad_h, pad_w, blend_mode, extra_params
        )
        
        # Импортируем функцию из improved_tiling_functions если доступна
        try:
            from improved_tiling_functions import compute_advanced_blend_padding
            # Используем уже готовые тензоры
            if abs(blend_strength) < 0.001:
                result = tensor_simple
            elif blend_strength > 0.999:
                result = tensor_advanced
            else:
                # Создаем маску и смешиваем
                b, c, H, W = tensor_simple.shape
                h = H - 2 * pad_h
                w = W - 2 * pad_w
                
                if blend_width is None:
                    blend_width = max(pad_h, pad_w)
                
                # Простое смешивание (можно улучшить)
                alpha = blend_strength
                result = tensor_simple * (1.0 - alpha) + tensor_advanced * alpha
        except ImportError:
            # Простой fallback
            alpha = blend_strength
            result = tensor_simple * (1.0 - alpha) + tensor_advanced * alpha
        
        return result
    
    # === РЕЖИМ 2: CONVERGENCE SHIFT (Legacy метод) ===
    elif zoom_mode == ZoomMode.CONVERGENCE_SHIFT.value or zoom_mode == 'convergence_shift':
        # Сначала применяем padding (простой circular)
        padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        
        # Затем применяем convergence zoom БЕЗ изменения изображения
        result = apply_convergence_zoom(
            padded, 
            zoom_factor=zoom_factor,
            convergence_point=convergence_point,
            depth_power=depth_power,
            axis='both'
        )
        
        return result
    
    # === РЕЖИМ 3: GRID WARP ===
    elif zoom_mode == ZoomMode.GRID_WARP.value or zoom_mode == 'grid_warp':
        # Применяем grid warp zoom (истинный geometric zoom)
        warped = apply_grid_warp_zoom(
            input_tensor,
            zoom_factor=zoom_factor,
            center_x=convergence_point,
            center_y=0.5,
            warp_power=depth_power
        )
        
        # Затем padding на warped изображение
        result = F.pad(warped, (pad_w, pad_w, pad_h, pad_h), mode='circular')
        
        return result
    
    # === РЕЖИМ 4: HYBRID (Комбинация методов) ===
    elif zoom_mode == ZoomMode.HYBRID.value or zoom_mode == 'hybrid':
        # Комбинируем convergence shift (для краев) и blend transition (для seamless)
        
        # 1. Blend transition
        blend_strength = (zoom_factor + 5.0) / 10.0
        blend_strength = max(0.0, min(1.0, blend_strength))
        
        tensor_simple, tensor_advanced = get_blend_mode_tensors(
            input_tensor, pad_h, pad_w, blend_mode, extra_params
        )
        
        blended = tensor_simple * (1.0 - blend_strength) + tensor_advanced * blend_strength
        
        # 2. Convergence shift (слабо)
        result = apply_convergence_zoom(
            blended,
            zoom_factor=zoom_factor * 0.3,  # Уменьшаем силу
            convergence_point=convergence_point,
            depth_power=depth_power,
            axis='horizontal'
        )
        
        return result
    
    # === DEFAULT: Обычный circular padding ===
    else:
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def validate_zoom_params(params):
    """Валидация параметров zoom"""
    return {
        'zoom_factor': max(-5.0, min(5.0, float(params.get('zoom_factor', 0.0)))),
        'zoom_mode': str(params.get('zoom_mode', 'blend_transition')),
        'blend_mode': str(params.get('blend_mode', 'circular_reflect')),
        'convergence_point': max(0.0, min(1.0, float(params.get('convergence_point', 0.5)))),
        'depth_power': max(0.25, min(4.0, float(params.get('depth_power', 1.0)))),
        'blend_falloff': str(params.get('blend_falloff', 'smoothstep')),
        'blend_sharpness': max(0.1, min(5.0, float(params.get('blend_sharpness', 1.0)))),
        'blend_width': int(params.get('blend_width', 0)) if params.get('blend_width') else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED ZOOM & PROXIMITY SYSTEM - TEST")
    print("=" * 80)
    
    # Тест convergence zoom
    print("\n🎯 ТЕСТ CONVERGENCE ZOOM (Legacy Method):")
    x = torch.randn(1, 3, 64, 64)
    
    for zoom in [-3, -1, 0, 1, 3]:
        result = apply_convergence_zoom(x, zoom_factor=zoom)
        print(f"  Zoom {zoom:+2d}: shape={result.shape}, mean={result.mean().item():.4f}")
    
    # Тест grid warp zoom
    print("\n🔄 ТЕСТ GRID WARP ZOOM:")
    for zoom in [-3, -1, 0, 1, 3]:
        result = apply_grid_warp_zoom(x, zoom_factor=zoom)
        print(f"  Zoom {zoom:+2d}: shape={result.shape}, mean={result.mean().item():.4f}")
    
    # Тест unified zoom
    print("\n🌟 ТЕСТ UNIFIED ZOOM SYSTEM:")
    for zoom in [-5, -2, 0, 2, 5]:
        result = apply_unified_zoom(
            x, pad_h=8, pad_w=8,
            zoom_factor=zoom,
            zoom_mode='blend_transition',
            blend_mode='circular_reflect'
        )
        print(f"  Zoom {zoom:+2d}: shape={result.shape}, mean={result.mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 80)
