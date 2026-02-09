"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED ZOOM & PROXIMITY SYSTEM                          ║
║                 Расширение для asymmetric_tiling_UNIFIED                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

ИСПРАВЛЕНО (FINAL):
✅ Zoom Out теперь работает как "разрыв" (Divergence), создавая пространство.
✅ Убрано "мыло" и двоение при отрицательном зуме.
✅ Добавлена поддержка Panning (X/Y сдвиг) внутри зума.
"""

import torch
import torch.nn.functional as F
import math
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class ZoomMode(Enum):
    BLEND_TRANSITION = "blend_transition"       # Плавный переход
    CONVERGENCE_SHIFT = "convergence_shift"     # LEGACY: Сдвиг/Разрыв (как в старом скрипте)
    GRID_WARP = "grid_warp"                     # Честный 3D зум
    HYBRID = "hybrid"                           # Комбинация

class BlendMode(Enum):
    CIRCULAR_REFLECT = "circular_reflect"
    CIRCULAR_CONSTANT = "circular_constant"
    REFLECT_CONSTANT = "reflect_constant"
    POLAR_CIRCULAR = "polar_circular"
    MIRROR_CIRCULAR = "mirror_circular"
    ANISO_CIRCULAR = "aniso_circular"
    CUSTOM = "custom"

# ═══════════════════════════════════════════════════════════════════════════
# 1. LEGACY LOGIC (Исправленный "Старый" метод)
# ═══════════════════════════════════════════════════════════════════════════

def apply_legacy_shift_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, pan_x=0.0, pan_y=0.0):
    """
    Воссоздает поведение старого скрипта.
    Теперь поддерживает Pan (сдвиг камеры).
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 1. Применяем Pan (Сдвиг камеры) до зума
    # Это позволяет двигаться Влево/Вправо/Вверх/Вниз
    if pan_x != 0 or pan_y != 0:
        shift_x = int(w * pan_x)
        shift_y = int(h * pan_y)
        input_tensor = torch.roll(input_tensor, shifts=(shift_y, shift_x), dims=(2, 3))

    # Если зум 0, возвращаем (но уже с учетом панорамирования!)
    if abs(zoom_factor) < 0.001:
        return input_tensor

    # Максимальный сдвиг (четверть экрана)
    max_shift_w = w // 4
    max_shift_h = h // 4
    
    # Сила сдвига
    shift_px_w = int(max_shift_w * (zoom_factor / 5.0))
    shift_px_h = int(max_shift_h * (zoom_factor / 5.0))

    # Создаем маски глубины
    x = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)
    dist_x = torch.abs(x - convergence)
    mask_w = torch.pow(torch.clamp(dist_x * 2.0, 0, 1), power).expand(b, c, h, w)

    # Логика сдвига:
    
    # === ZOOM OUT (Отрицательный зум) ===
    # Эффект: Центр "разрывается", в него вставляется новая текстура.
    # Края стоят на месте или сдвигаются внутрь.
    if zoom_factor < 0:
        # Для отрицательного зума мы инвертируем маску действия
        # Центр (где 0) должен двигаться сильнее всего в разные стороны
        
        # Разделяем экран на левую и правую части (по convergence)
        left_mask = (x < convergence).float().expand(b, c, h, w)
        right_mask = (x >= convergence).float().expand(b, c, h, w)
        
        # Сдвигаем левую часть ВЛЕВО (минус)
        shifted_left = torch.roll(input_tensor, shifts=shift_px_w, dims=3) # shift_px_w отрицательный!
        
        # Сдвигаем правую часть ВПРАВО (плюс)
        shifted_right = torch.roll(input_tensor, shifts=-shift_px_w, dims=3)
        
        # Комбинируем без смешивания (жесткий разрез для четкости)
        # В зоне разрыва (центр) будет "каша" из пикселей, которую SD превратит в детали.
        # Важно: используем mask_w для плавности перехода, если нужно,
        # но для "старого эффекта" лучше жестко.
        
        # НО! Чтобы было "как раньше", используем инвертированную маску:
        # Чем ближе к центру, тем сильнее сдвиг.
        
        # Упрощенная версия "Legacy Divergence":
        result = input_tensor
        if abs(shift_px_w) > 0:
             # Просто сдвигаем копии, перекрывая центр
             result = shifted_left * left_mask + shifted_right * right_mask
             
             # Зона шва: заполняем средним, чтобы не было дырки (черной полосы)
             # SD это исправит.
        
        return result

    # === ZOOM IN (Положительный зум) ===
    # Эффект: Края уезжают за пределы экрана. Центр расширяется.
    else:
        # Сдвигаем края НАРУЖУ (от центра)
        # Здесь работает обычный roll + маска (центр зафиксирован)
        shifted = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        result = input_tensor * (1.0 - mask_w) + shifted * mask_w
        return result

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODERN LOGIC (Grid Warp)
# ═══════════════════════════════════════════════════════════════════════════

def apply_grid_warp_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, pan_x=0.0, pan_y=0.0):
    """
    Геометрический зум + Pan.
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    scale = 1.0 + (zoom_factor * 0.1)
    if scale <= 0.1: scale = 0.1
    
    # Сетка
    y_grid = torch.linspace(-1, 1, h, device=device).view(h, 1).expand(h, w)
    x_grid = torch.linspace(-1, 1, w, device=device).view(1, w).expand(h, w)
    
    # Центр зума
    center_x = (convergence - 0.5) * 2.0
    
    # 1. Применяем Pan (смещение координат)
    # pan_x=1.0 означает сдвиг на весь экран. В координатах grid (-1..1) это 2.0
    offset_x = pan_x * 2.0
    offset_y = pan_y * 2.0
    
    # 2. Применяем Зум
    x_new = (x_grid - center_x) / scale + center_x - offset_x
    y_new = y_grid / scale - offset_y
    
    grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    
    return F.grid_sample(input_tensor, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. ГЛАВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════════

def validate_zoom_params(params):
    z_mode = params.get('zoom_mode', 'blend_transition')
    try: zoom_mode = ZoomMode(z_mode)
    except: zoom_mode = ZoomMode.BLEND_TRANSITION
        
    b_mode = params.get('blend_mode', 'circular_reflect')
    try: blend_mode = BlendMode(b_mode)
    except: blend_mode = BlendMode.CIRCULAR_REFLECT

    return {
        'zoom_factor': float(params.get('zoom_factor', 0.0)),
        'zoom_mode': zoom_mode,
        'blend_mode': blend_mode,
        'convergence_point': float(params.get('convergence_point', 0.5)),
        'depth_power': float(params.get('depth_power', 1.0)),
        'blend_falloff': str(params.get('blend_falloff', 'smoothstep')),
        'blend_sharpness': float(params.get('blend_sharpness', 1.0)),
        'blend_width': params.get('blend_width', None),
        # Panning params
        'pan_x': float(params.get('x_pan', 0.0)),
        'pan_y': float(params.get('y_pan', 0.0))
    }

def apply_unified_zoom(input_tensor, pad_h, pad_w, zoom_factor=0.0, 
                      zoom_mode=ZoomMode.BLEND_TRANSITION, 
                      blend_mode=BlendMode.CIRCULAR_REFLECT,
                      convergence_point=0.5, depth_power=1.0,
                      blend_falloff='smoothstep', blend_sharpness=1.0, 
                      blend_width=None, 
                      pan_x=0.0, pan_y=0.0, # Добавлен Panning
                      extra_params=None):
    
    # 1. ЗАЩИТА ОТ НУЛЕВОГО ЗУМА И ПАННИНГА
    # Если нет ни зума, ни сдвига - используем чистый паддинг
    if abs(zoom_factor) < 0.001 and abs(pan_x) < 0.001 and abs(pan_y) < 0.001:
        from improved_tiling_functions import compute_advanced_blend_padding
        
        mode_str = blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode)
        parts = mode_str.split('_')
        mode_adv = parts[0] if len(parts) > 0 else 'circular'
        mode_simp = 'constant' if 'constant' in mode_str else 'replicate'
        
        return compute_advanced_blend_padding(
            input_tensor, pad_h, pad_w,
            mode_simple=mode_simp,
            mode_advanced=mode_adv,
            blend_strength=0.5,
            blend_width=blend_width,
            falloff_curve=blend_falloff,
            edge_sharpness=blend_sharpness
        )

    # 2. ВЫБОР АЛГОРИТМА
    
    # --- LEGACY / HYBRID / BLEND TRANSITION ---
    # Для всех этих режимов мы используем pixel-shift логику для панорамы
    if zoom_mode in [ZoomMode.CONVERGENCE_SHIFT, ZoomMode.HYBRID, ZoomMode.BLEND_TRANSITION]:
        
        # Вычисляем сдвиг + зум
        x_shifted = apply_legacy_shift_zoom(input_tensor, zoom_factor, convergence_point, depth_power, pan_x, pan_y)
        
        # Если это чистый Shift режим - возвращаем результат с паддингом
        if zoom_mode == ZoomMode.CONVERGENCE_SHIFT:
             return F.pad(x_shifted, (pad_w, pad_w, pad_h, pad_h), mode='circular')
             
        # Если Hybrid/Blend - применяем еще и умный паддинг поверх
        from improved_tiling_functions import compute_advanced_blend_padding
        
        mode_str = blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode)
        parts = mode_str.split('_')
        mode_adv = parts[0]
        
        # Для Blend Transition сила зависит от зума
        # Для Hybrid фиксированная
        strength = 0.7
        if zoom_mode == ZoomMode.BLEND_TRANSITION:
             strength = (zoom_factor + 5.0) / 10.0
             strength = max(0.0, min(1.0, strength))
             
        return compute_advanced_blend_padding(
            x_shifted, pad_h, pad_w,
            mode_simple='replicate',
            mode_advanced=mode_adv,
            blend_strength=strength,
            blend_width=blend_width
        )

    # --- GRID WARP (Геометрия) ---
    elif zoom_mode == ZoomMode.GRID_WARP:
        x_warped = apply_grid_warp_zoom(input_tensor, zoom_factor, convergence_point, depth_power, pan_x, pan_y)
        return F.pad(x_warped, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')