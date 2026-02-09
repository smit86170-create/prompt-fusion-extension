"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED ZOOM & PROXIMITY SYSTEM                          ║
║                 Расширение для asymmetric_tiling_UNIFIED                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

ИСПРАВЛЕНО (FINAL FIXED):
✅ Zoom Out теперь работает как в старой версии через FADE TO BLACK
✅ Убрано "мыло" и двоение при отрицательном зуме
✅ Добавлена поддержка Panning (X/Y сдвиг) внутри зума
✅ Legacy режим работает корректно с затемнением краев
"""

import torch
import torch.nn.functional as F
import math
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class ZoomMode(Enum):
    BLEND_TRANSITION = "blend_transition"       # Плавный переход с затемнением
    CONVERGENCE_SHIFT = "convergence_shift"     # LEGACY: Сдвиг/Разрыв (старый скрипт)
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

def apply_legacy_shift_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                           pan_x=0.0, pan_y=0.0):
    """
    Воссоздает поведение старого скрипта.
    Теперь поддерживает Pan (сдвиг камеры).
    
    ⚠️ ВАЖНО: Эта функция делает только PIXEL SHIFT!
    Для создания zoom эффекта ОБЯЗАТЕЛЬНО нужно применить 
    затемнение краев (fade to black) после паддинга!
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 1. Применяем Pan (Сдвиг камеры) до зума
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

    # === ZOOM OUT (Отрицательный зум) ===
    if zoom_factor < 0:
        # Разделяем экран на левую и правую части
        left_mask = (x < convergence).float().expand(b, c, h, w)
        right_mask = (x >= convergence).float().expand(b, c, h, w)
        
        # Сдвигаем левую часть ВЛЕВО
        shifted_left = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        
        # Сдвигаем правую часть ВПРАВО
        shifted_right = torch.roll(input_tensor, shifts=-shift_px_w, dims=3)
        
        # Комбинируем
        result = shifted_left * left_mask + shifted_right * right_mask
        return result

    # === ZOOM IN (Положительный зум) ===
    else:
        # Сдвигаем края НАРУЖУ (от центра)
        shifted = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        result = input_tensor * (1.0 - mask_w) + shifted * mask_w
        return result

# ═══════════════════════════════════════════════════════════════════════════
# 2. MODERN LOGIC (Grid Warp)
# ═══════════════════════════════════════════════════════════════════════════

def apply_grid_warp_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                        pan_x=0.0, pan_y=0.0):
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
    
    # 1. Применяем Pan
    offset_x = pan_x * 2.0
    offset_y = pan_y * 2.0
    
    # 2. Применяем Зум
    x_new = (x_grid - center_x) / scale + center_x - offset_x
    y_new = y_grid / scale - offset_y
    
    grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    
    return F.grid_sample(input_tensor, grid, mode='bilinear', 
                        padding_mode='reflection', align_corners=True)

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
        'pan_x': float(params.get('x_pan', 0.0)),
        'pan_y': float(params.get('y_pan', 0.0)),
        # ✅ НОВЫЕ ПАРАМЕТРЫ для fade to black
        'fade_to_black': bool(params.get('zoom_fade_to_black', True)),  # По умолчанию ВКЛ
        'fade_strength': float(params.get('zoom_fade_strength', 0.15))  # 15% зона затемнения
    }

def apply_unified_zoom(input_tensor, pad_h, pad_w, zoom_factor=0.0, 
                      zoom_mode=ZoomMode.BLEND_TRANSITION, 
                      blend_mode=BlendMode.CIRCULAR_REFLECT,
                      convergence_point=0.5, depth_power=1.0,
                      blend_falloff='smoothstep', blend_sharpness=1.0, 
                      blend_width=None, 
                      pan_x=0.0, pan_y=0.0,
                      fade_to_black=True, fade_strength=0.15,  # ✅ НОВЫЕ ПАРАМЕТРЫ
                      extra_params=None):
    """
    Унифицированная функция Zoom с поддержкой FADE TO BLACK.
    
    ✅ ИСПРАВЛЕНО:
    - Добавлен параметр fade_to_black для создания zoom эффекта
    - По умолчанию fade_to_black=True для всех legacy режимов
    - fade_strength=0.15 означает 15% зона затемнения на краях
    
    Как это работает:
    1. Применяется pixel shift (если нужен)
    2. Применяется паддинг (circular/reflect)
    3. ✅ КРАЯ ЗАТЕМНЯЮТСЯ через fade to black
    4. SD видит "пустоту" и заполняет её → эффект zoom!
    """
    
    # 1. ЗАЩИТА ОТ НУЛЕВОГО ЗУМА И ПАННИНГА
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
            edge_sharpness=blend_sharpness,
            fade_to_black=False  # Обычный режим БЕЗ затемнения
        )

    # 2. ВЫБОР АЛГОРИТМА
    
    # ═══════════════════════════════════════════════════════════════════
    # LEGACY / HYBRID / BLEND TRANSITION (С FADE TO BLACK!)
    # ═══════════════════════════════════════════════════════════════════
    if zoom_mode in [ZoomMode.CONVERGENCE_SHIFT, ZoomMode.HYBRID, ZoomMode.BLEND_TRANSITION]:
        
        # 1. Вычисляем сдвиг + зум
        x_shifted = apply_legacy_shift_zoom(input_tensor, zoom_factor, 
                                           convergence_point, depth_power, pan_x, pan_y)
        
        # 2. Если это чистый Shift режим
        if zoom_mode == ZoomMode.CONVERGENCE_SHIFT:
            # Простой circular padding
            padded = F.pad(x_shifted, (pad_w, pad_w, pad_h, pad_h), mode='circular')
            
            # ✅ КРИТИЧЕСКИ ВАЖНО: Затемняем края!
            if fade_to_black:
                from improved_tiling_functions import compute_blend_fade_to_black
                padded = compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength)
            
            return padded
             
        # 3. Если Hybrid/Blend - применяем умный паддинг с fade to black
        from improved_tiling_functions import compute_advanced_blend_padding
        
        mode_str = blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode)
        parts = mode_str.split('_')
        mode_adv = parts[0] if len(parts) > 0 else 'circular'
        
        # Для Blend Transition сила зависит от зума
        if zoom_mode == ZoomMode.BLEND_TRANSITION:
            # При отрицательном зуме (zoom out) ОБЯЗАТЕЛЬНО fade to black
            # При положительном зуме можно регулировать
            if zoom_factor < 0:
                fade_to_black = True
                # Чем сильнее zoom out, тем сильнее затемнение
                fade_strength = min(0.3, abs(zoom_factor) * 0.03)
        
        # ✅ ПРИМЕНЯЕМ С FADE TO BLACK!
        return compute_advanced_blend_padding(
            x_shifted, pad_h, pad_w,
            mode_simple='replicate',
            mode_advanced=mode_adv,
            blend_strength=0.7,
            blend_width=blend_width,
            falloff_curve=blend_falloff,
            edge_sharpness=blend_sharpness,
            # ✅ КРИТИЧЕСКИ ВАЖНО!
            fade_to_black=fade_to_black,
            fade_strength=fade_strength
        )

    # ═══════════════════════════════════════════════════════════════════
    # GRID WARP (Геометрия) - НЕ нужен fade to black
    # ═══════════════════════════════════════════════════════════════════
    elif zoom_mode == ZoomMode.GRID_WARP:
        x_warped = apply_grid_warp_zoom(input_tensor, zoom_factor, 
                                       convergence_point, depth_power, pan_x, pan_y)
        # Grid warp сам создает эффект zoom через геометрию
        # Fade to black не нужен
        return F.pad(x_warped, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # Fallback
    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')