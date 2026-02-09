"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED ZOOM & PROXIMITY SYSTEM                          ║
║                 ИСПРАВЛЕННАЯ ВЕРСИЯ - РЕАЛЬНЫЙ OUTPAINTING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ИСПРАВЛЕНО (НАСТОЯЩАЯ ПРОБЛЕМА):
✅ ИСПРАВЛЕНА ошибка Broadcasting (expanded size must match)
✅ Правильная форма тензоров для expand()
✅ Zoom работает В LATENT SPACE для outpainting эффекта
✅ Grid Warp оптимизирован для латентов
✅ Поддержка всех режимов zoom

ВАЖНО: Этот код делает НАСТОЯЩИЙ zoom с outpainting в latent space!
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
    CONVERGENCE_SHIFT = "convergence_shift"     # Сдвиг/Разрыв 
    GRID_WARP = "grid_warp"                     # Честный геометрический зум
    HYBRID = "hybrid"                           # Комбинация
    OUTPAINT_ZOOM = "outpaint_zoom"            # Специальный режим для outpainting

class BlendMode(Enum):
    CIRCULAR_REFLECT = "circular_reflect"
    CIRCULAR_CONSTANT = "circular_constant"
    REFLECT_CONSTANT = "reflect_constant"
    POLAR_CIRCULAR = "polar_circular"
    MIRROR_CIRCULAR = "mirror_circular"
    ANISO_CIRCULAR = "aniso_circular"
    CUSTOM = "custom"

# ═══════════════════════════════════════════════════════════════════════════
# 1. ИСПРАВЛЕННЫЙ LEGACY METHOD
# ═══════════════════════════════════════════════════════════════════════════

def apply_legacy_shift_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                           pan_x=0.0, pan_y=0.0):
    """
    ✅ ИСПРАВЛЕНО: Правильный Broadcasting для expand()
    
    ПРОБЛЕМА БЫЛА: x.view(1, 1, 1, w) не может expand до (b, c, h, w) 
                   если h != 1 (ошибка "expanded size must match")
    
    РЕШЕНИЕ: Создаем маски правильной формы с самого начала
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # 1. Применяем Pan (Сдвиг камеры) до зума
    if pan_x != 0 or pan_y != 0:
        shift_x = int(w * pan_x)
        shift_y = int(h * pan_y)
        input_tensor = torch.roll(input_tensor, shifts=(shift_y, shift_x), dims=(2, 3))

    # Если зум 0, возвращаем
    if abs(zoom_factor) < 0.001:
        return input_tensor

    # Максимальный сдвиг
    max_shift_w = w // 4
    max_shift_h = h // 4
    
    # Сила сдвига
    shift_px_w = int(max_shift_w * (zoom_factor / 5.0))
    shift_px_h = int(max_shift_h * (zoom_factor / 5.0))

    # ✅ ИСПРАВЛЕНО: Создаем x ПРАВИЛЬНОЙ формы с самого начала!
    # БЫЛО: x = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
    # СТАЛО: Создаем сразу полноразмерную сетку
    x_1d = torch.linspace(0, 1, w, device=device, dtype=dtype)
    x = x_1d.view(1, 1, 1, w).expand(b, c, h, w)  # ✅ Теперь это (b, c, h, w) сразу!
    
    # Вычисляем расстояние и маску
    dist_x = torch.abs(x - convergence)
    mask_w = torch.pow(torch.clamp(dist_x * 2.0, 0, 1), power)
    # Теперь mask_w уже (b, c, h, w), expand не нужен!

    # === ZOOM OUT (Отрицательный зум) ===
    if zoom_factor < 0:
        # ✅ ИСПРАВЛЕНО: Маски уже правильной формы
        left_mask = (x < convergence).to(dtype=dtype)
        right_mask = (x >= convergence).to(dtype=dtype)
        
        # Сдвигаем части
        shifted_left = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        shifted_right = torch.roll(input_tensor, shifts=-shift_px_w, dims=3)
        
        # Комбинируем
        result = shifted_left * left_mask + shifted_right * right_mask
        return result

    # === ZOOM IN (Положительный зум) ===
    else:
        shifted = torch.roll(input_tensor, shifts=shift_px_w, dims=3)
        result = input_tensor * (1.0 - mask_w) + shifted * mask_w
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. УЛУЧШЕННЫЙ GRID WARP (для латентов)
# ═══════════════════════════════════════════════════════════════════════════

def apply_grid_warp_zoom(input_tensor, zoom_factor, convergence=0.5, power=1.0, 
                        pan_x=0.0, pan_y=0.0):
    """
    ✅ ИСПРАВЛЕНО: Оптимизирован для работы в latent space
    
    Использует grid_sample с правильным padding для создания 
    outpainting эффекта (дорисовки краев нейросетью).
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Вычисляем масштаб
    scale = 1.0 + (zoom_factor * 0.1)
    scale = max(0.1, min(scale, 5.0))  # Ограничиваем разумными пределами
    
    # ✅ Создаем координатную сетку
    y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    
    # Используем meshgrid для создания полной сетки
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Центр зума (в нормализованных координатах [-1, 1])
    center_x = (convergence - 0.5) * 2.0
    center_y = 0.0  # Вертикальный центр по умолчанию
    
    # 1. Применяем Pan (сдвиг)
    offset_x = pan_x * 2.0
    offset_y = pan_y * 2.0
    
    # 2. Применяем Zoom относительно центра
    x_new = (x_grid - center_x) / scale + center_x - offset_x
    y_new = (y_grid - center_y) / scale + center_y - offset_y
    
    # 3. Создаем grid для grid_sample
    # Форма должна быть (B, H, W, 2), где последний dim = [x, y]
    grid = torch.stack((x_new, y_new), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)
    
    # ✅ ВАЖНО: padding_mode='zeros' для outpainting эффекта!
    # Области вне [-1,1] заполнятся нулями, что даст нейросети 
    # "пространство" для генерации нового контента
    return F.grid_sample(
        input_tensor, 
        grid, 
        mode='bilinear',
        padding_mode='zeros',  # ← Ключевой параметр для outpainting!
        align_corners=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. НОВЫЙ РЕЖИМ: OUTPAINT ZOOM (специально для дорисовки)
# ═══════════════════════════════════════════════════════════════════════════

def apply_outpaint_zoom(input_tensor, zoom_factor, pad_h, pad_w, 
                       convergence=0.5, fade_strength=0.3):
    """
    ✅ НОВЫЙ МЕТОД: Оптимизирован для создания outpainting эффекта
    
    Принцип работы:
    1. Если zoom < 0 (отдаление): создает области для дорисовки
    2. Использует мягкий градиент на краях
    3. Сохраняет латентную структуру для генерации
    
    Это дает нейросети "подсказку" где рисовать новый контент!
    """
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    if abs(zoom_factor) < 0.001:
        # Нет zoom - просто делаем padding
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    # Zoom OUT (отдаление) - основной режим для outpainting
    if zoom_factor < 0:
        # 1. Уменьшаем контент
        scale = 1.0 + abs(zoom_factor) * 0.1
        scale = max(1.0, min(scale, 3.0))
        
        # Размеры уменьшенного контента
        new_h = max(int(h / scale), h // 4)
        new_w = max(int(w / scale), w // 4)
        
        # Уменьшаем с помощью F.interpolate
        content_small = F.interpolate(
            input_tensor,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=True
        )
        
        # 2. Создаем градиентную маску для мягких краев
        fade_h = int(new_h * fade_strength)
        fade_w = int(new_w * fade_strength)
        
        # Создаем маску
        mask = torch.ones(1, 1, new_h, new_w, device=device, dtype=dtype)
        
        if fade_h > 0 and fade_w > 0:
            # Горизонтальные градиенты
            fade_x = torch.linspace(0, 1, fade_w, device=device, dtype=dtype)
            mask[:, :, :, :fade_w] *= fade_x.view(1, 1, 1, -1)
            mask[:, :, :, -fade_w:] *= fade_x.flip(0).view(1, 1, 1, -1)
            
            # Вертикальные градиенты
            fade_y = torch.linspace(0, 1, fade_h, device=device, dtype=dtype)
            mask[:, :, :fade_h, :] *= fade_y.view(1, 1, -1, 1)
            mask[:, :, -fade_h:, :] *= fade_y.flip(0).view(1, 1, -1, 1)
        
        # Применяем маску к контенту
        content_faded = content_small * mask.expand_as(content_small)
        
        # 3. Размещаем в центре нового (большего) холста
        canvas_h = h + 2 * pad_h
        canvas_w = w + 2 * pad_w
        
        # Создаем холст (заполненный слабым шумом для outpainting)
        # ✅ ВАЖНО: НЕ нулями, а слабым шумом! Это помогает нейросети
        canvas = torch.randn(b, c, canvas_h, canvas_w, device=device, dtype=dtype) * 0.1
        
        # Вычисляем позицию для вставки
        paste_y = (canvas_h - new_h) // 2
        paste_x = (canvas_w - new_w) // 2
        
        # Вставляем контент
        canvas[:, :, paste_y:paste_y+new_h, paste_x:paste_x+new_w] = content_faded
        
        return canvas
    
    # Zoom IN (приближение)
    else:
        # Увеличиваем контент и обрезаем
        scale = 1.0 + zoom_factor * 0.1
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Увеличиваем
        content_large = F.interpolate(
            input_tensor,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=True
        )
        
        # Обрезаем центральную часть
        crop_y = (new_h - h) // 2
        crop_x = (new_w - w) // 2
        
        cropped = content_large[:, :, crop_y:crop_y+h, crop_x:crop_x+w]
        
        # Добавляем padding
        return F.pad(cropped, (pad_w, pad_w, pad_h, pad_h), mode='circular')


# ═══════════════════════════════════════════════════════════════════════════
# 4. ГЛАВНАЯ ФУНКЦИЯ (ИСПРАВЛЕННАЯ)
# ═══════════════════════════════════════════════════════════════════════════

def validate_zoom_params(params):
    """Валидация параметров zoom"""
    z_mode = params.get('zoom_mode', 'outpaint_zoom')
    try: 
        zoom_mode = ZoomMode(z_mode)
    except: 
        zoom_mode = ZoomMode.OUTPAINT_ZOOM  # По умолчанию используем outpaint режим
        
    b_mode = params.get('blend_mode', 'circular_reflect')
    try: 
        blend_mode = BlendMode(b_mode)
    except: 
        blend_mode = BlendMode.CIRCULAR_REFLECT

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
        'fade_to_black': bool(params.get('zoom_fade_to_black', True)),
        'fade_strength': float(params.get('zoom_fade_strength', 0.3))
    }


def apply_unified_zoom(input_tensor, pad_h, pad_w, zoom_factor=0.0, 
                      zoom_mode=ZoomMode.OUTPAINT_ZOOM, 
                      blend_mode=BlendMode.CIRCULAR_REFLECT,
                      convergence_point=0.5, depth_power=1.0,
                      blend_falloff='smoothstep', blend_sharpness=1.0, 
                      blend_width=None, 
                      pan_x=0.0, pan_y=0.0,
                      fade_to_black=True, fade_strength=0.3,
                      extra_params=None):
    """
    ✅ ИСПРАВЛЕННАЯ унифицированная функция Zoom
    
    РАБОТАЕТ В LATENT SPACE ДЛЯ OUTPAINTING ЭФФЕКТА!
    
    Режимы:
    - OUTPAINT_ZOOM: Оптимизирован для дорисовки (рекомендуется!)
    - GRID_WARP: Геометрический zoom с grid_sample
    - BLEND_TRANSITION: Плавный переход с blending
    - CONVERGENCE_SHIFT: Legacy режим со сдвигом
    """
    
    # 1. Защита от нулевого zoom
    if abs(zoom_factor) < 0.001 and abs(pan_x) < 0.001 and abs(pan_y) < 0.001:
        # Просто делаем обычный padding
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ: OUTPAINT ZOOM (рекомендуется для дорисовки)
    # ═══════════════════════════════════════════════════════════════════
    if zoom_mode == ZoomMode.OUTPAINT_ZOOM:
        return apply_outpaint_zoom(
            input_tensor, 
            zoom_factor, 
            pad_h, 
            pad_w,
            convergence_point, 
            fade_strength
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ: GRID WARP (геометрический zoom)
    # ═══════════════════════════════════════════════════════════════════
    elif zoom_mode == ZoomMode.GRID_WARP:
        x_warped = apply_grid_warp_zoom(
            input_tensor, 
            zoom_factor, 
            convergence_point, 
            depth_power, 
            pan_x, 
            pan_y
        )
        
        # После warp добавляем padding
        return F.pad(x_warped, (pad_w, pad_w, pad_h, pad_h), mode='zeros')
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ: LEGACY / HYBRID / BLEND TRANSITION
    # ═══════════════════════════════════════════════════════════════════
    elif zoom_mode in [ZoomMode.CONVERGENCE_SHIFT, ZoomMode.HYBRID, ZoomMode.BLEND_TRANSITION]:
        
        # 1. Применяем shift zoom (теперь с исправленным broadcasting!)
        x_shifted = apply_legacy_shift_zoom(
            input_tensor, 
            zoom_factor, 
            convergence_point, 
            depth_power, 
            pan_x, 
            pan_y
        )
        
        # 2. Для CONVERGENCE_SHIFT - простой padding
        if zoom_mode == ZoomMode.CONVERGENCE_SHIFT:
            padded = F.pad(x_shifted, (pad_w, pad_w, pad_h, pad_h), mode='circular')
            
            # Опционально затемняем края
            if fade_to_black:
                from improved_tiling_functions import compute_blend_fade_to_black
                padded = compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength)
            
            return padded
        
        # 3. Для HYBRID/BLEND - умный padding
        from improved_tiling_functions import compute_advanced_blend_padding
        
        mode_str = blend_mode.value if isinstance(blend_mode, BlendMode) else str(blend_mode)
        parts = mode_str.split('_')
        mode_adv = parts[0] if len(parts) > 0 else 'circular'
        
        # Автоматическая настройка fade для zoom out
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
    
    # Fallback - просто circular padding
    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')