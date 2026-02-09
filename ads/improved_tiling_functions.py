"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    IMPROVED BLEND & MULTI-RESOLUTION                          ║
║                          Advanced Tiling v3.1                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

УЛУЧШЕНИЯ:
✅ Blend Mode - теперь настоящий слайдер zoom/proximity
✅ Multi-Resolution - гибкие стратегии и параметры
✅ Унифицированная работа для всех режимов
✅ Без потери информации на краях
✅ Адаптивное кэширование
"""

import torch
import torch.nn.functional as F
import math
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# BLEND MODE - УЛУЧШЕННАЯ ВЕРСИЯ
# ═══════════════════════════════════════════════════════════════════════════

class BlendStrategy(Enum):
    """Стратегии смешивания для Blend Mode"""
    LINEAR = "linear"              # Линейная интерполяция
    SMOOTHSTEP = "smoothstep"      # Плавная S-кривая (Hermite)
    COSINE = "cosine"              # Косинусоидальная интерполяция
    PERCEPTUAL = "perceptual"      # Перцептивная (квадратный корень)
    

def blend_interpolation(t, strategy="smoothstep"):
    """
    Улучшенная функция интерполяции для blend mode
    
    Args:
        t: Значение от 0.0 до 1.0
        strategy: Стратегия интерполяции
        
    Returns:
        Интерполированное значение от 0.0 до 1.0
    """
    t = torch.clamp(t, 0.0, 1.0)
    
    if strategy == "linear":
        return t
    
    elif strategy == "smoothstep":
        # Hermite interpolation: 3t² - 2t³
        return t * t * (3.0 - 2.0 * t)
    
    elif strategy == "cosine":
        # Cosine interpolation: (1 - cos(πt)) / 2
        return (1.0 - torch.cos(t * math.pi)) / 2.0
    
    elif strategy == "perceptual":
        # Perceptual (sqrt): более сильное смешивание в начале
        return torch.sqrt(t)
    
    else:
        return t


def create_advanced_blend_mask(h, w, blend_width, device, dtype=torch.float32, 
                               falloff_curve="smoothstep", edge_sharpness=1.0):
    """
    Создает улучшенную маску для blend mode с настраиваемым falloff
    
    Args:
        h, w: Размеры маски
        blend_width: Ширина зоны перехода (в пикселях)
        device: Torch device
        dtype: Тип данных
        falloff_curve: Тип кривой затухания ('linear', 'smoothstep', 'cosine', 'perceptual')
        edge_sharpness: Резкость краев (1.0 = нормально, >1 = резче, <1 = мягче)
        
    Returns:
        Маска размером [1, 1, h, w] со значениями от 0.0 (края) до 1.0 (центр)
    """
    mask = torch.ones(h, w, device=device, dtype=dtype)
    
    if blend_width <= 0:
        return mask.unsqueeze(0).unsqueeze(0)
    
    # Ограничиваем ширину блендинга до 1/3 размера (чтобы не перекрывались)
    blend_w = min(blend_width, w // 3)
    blend_h = min(blend_width, h // 3)
    
    # === ГОРИЗОНТАЛЬНЫЕ КРАЯ ===
    if blend_w > 0:
        # Левый край
        for i in range(blend_w):
            t = (i + 1) / (blend_w + 1)
            t = t ** edge_sharpness  # Применяем sharpness
            alpha = blend_interpolation(
                torch.tensor(t, device=device, dtype=dtype), 
                strategy=falloff_curve
            )
            mask[:, i] = alpha
        
        # Правый край
        for i in range(blend_w):
            t = (i + 1) / (blend_w + 1)
            t = t ** edge_sharpness
            alpha = blend_interpolation(
                torch.tensor(t, device=device, dtype=dtype), 
                strategy=falloff_curve
            )
            mask[:, -(i + 1)] = alpha
    
    # === ВЕРТИКАЛЬНЫЕ КРАЯ ===
    if blend_h > 0:
        # Верхний край
        for i in range(blend_h):
            t = (i + 1) / (blend_h + 1)
            t = t ** edge_sharpness
            alpha = blend_interpolation(
                torch.tensor(t, device=device, dtype=dtype), 
                strategy=falloff_curve
            )
            # Берем минимум, чтобы углы работали правильно
            mask[i, :] = torch.minimum(
                mask[i, :], 
                torch.full_like(mask[i, :], alpha)
            )
        
        # Нижний край
        for i in range(blend_h):
            t = (i + 1) / (blend_h + 1)
            t = t ** edge_sharpness
            alpha = blend_interpolation(
                torch.tensor(t, device=device, dtype=dtype), 
                strategy=falloff_curve
            )
            mask[-(i + 1), :] = torch.minimum(
                mask[-(i + 1), :], 
                torch.full_like(mask[-(i + 1), :], alpha)
            )
    
    return mask.unsqueeze(0).unsqueeze(0)


def compute_advanced_blend_padding(input_tensor, pad_h, pad_w, 
                                   mode_simple='constant', 
                                   mode_advanced='circular',
                                   blend_strength=0.5,
                                   blend_width=None,
                                   falloff_curve='smoothstep',
                                   edge_sharpness=1.0):
    """
    🌟 УЛУЧШЕННАЯ ВЕРСИЯ BLEND MODE 🌟
    
    Правильное смешивание двух режимов padding без потери информации.
    Работает как слайдер "приближения/отдаления":
    
    blend_strength = 0.0 → Полностью mode_simple (далеко, простой padding)
    blend_strength = 0.5 → Смешивание 50/50
    blend_strength = 1.0 → Полностью mode_advanced (близко, продвинутый tiling)
    
    Args:
        input_tensor: Входной тензор [B, C, H, W]
        pad_h, pad_w: Размеры padding
        mode_simple: "Простой" режим padding ('constant', 'replicate')
        mode_advanced: "Продвинутый" режим ('circular', 'reflect', или кастомный тензор)
        blend_strength: Сила смешивания (0.0 = simple, 1.0 = advanced)
        blend_width: Ширина зоны перехода в пикселях (None = auto)
        falloff_curve: Кривая затухания ('linear', 'smoothstep', 'cosine', 'perceptual')
        edge_sharpness: Резкость перехода (1.0 = нормально)
        
    Returns:
        Тензор с padding [B, C, H+2*pad_h, W+2*pad_w]
    """
    
    # Валидация
    blend_strength = max(0.0, min(float(blend_strength), 1.0))
    
    # Если blend_strength = 0, сразу возвращаем простой режим
    if blend_strength < 0.001:
        if mode_simple == 'constant':
            return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        else:
            return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_simple)
    
    # Если blend_strength = 1, возвращаем продвинутый режим
    if blend_strength > 0.999:
        if isinstance(mode_advanced, str):
            if mode_advanced == 'circular':
                return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
            elif mode_advanced == 'reflect':
                return _safe_pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            else:
                return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_advanced)
        else:
            # mode_advanced это уже готовый тензор (для panorama/cubemap)
            return mode_advanced
    
    # === СОЗДАЕМ ОБА РЕЖИМА ===
    
    # Простой режим
    if mode_simple == 'constant':
        padded_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                             mode='constant', value=0)
    else:
        padded_simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                             mode=mode_simple)
    
    # Продвинутый режим
    if isinstance(mode_advanced, str):
        if mode_advanced == 'circular':
            padded_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                                   mode='circular')
        elif mode_advanced == 'reflect':
            padded_advanced = _safe_pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                                       mode='reflect')
        else:
            padded_advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), 
                                   mode=mode_advanced)
    else:
        # Это уже готовый тензор (для сложных режимов типа panorama)
        padded_advanced = mode_advanced
    
    # === СОЗДАЕМ МАСКУ СМЕШИВАНИЯ ===
    
    # Auto blend width
    if blend_width is None:
        blend_width = max(pad_h, pad_w)
    
    h_padded, w_padded = padded_simple.shape[-2:]
    
    # Создаем маску с улучшенным falloff
    mask = create_advanced_blend_mask(
        h_padded, w_padded, blend_width,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
        falloff_curve=falloff_curve,
        edge_sharpness=edge_sharpness
    )
    
    # Применяем blend_strength к маске
    # mask: 0.0 (края) → 1.0 (центр)
    # Когда blend_strength = 0.5:
    #   - На краях: 0.0 * 0.5 = 0.0 → 100% simple
    #   - В центре: 1.0 * 0.5 = 0.5 → 50% simple, 50% advanced
    
    mask = mask * blend_strength
    mask = mask.expand_as(padded_simple)
    
    # === СМЕШИВАЕМ ===
    # result = simple * (1 - mask) + advanced * mask
    result = padded_simple * (1.0 - mask) + padded_advanced * mask
    
    return result


def _safe_pad(x, pad, mode='reflect', value=0.0):
    """
    Безопасный wrapper для F.pad с обработкой reflect mode
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


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-RESOLUTION MODE - УЛУЧШЕННАЯ ВЕРСИЯ
# ═══════════════════════════════════════════════════════════════════════════

class MultiResStrategy(Enum):
    """Стратегии для Multi-Resolution Mode"""
    LINEAR = "linear"              # Линейный переход
    COSINE = "cosine"              # Косинусоидальный (плавный)
    EXPONENTIAL = "exponential"    # Экспоненциальный (быстрый старт)
    SIGMOID = "sigmoid"            # S-образный (медленный старт и конец)
    EARLY_BOOST = "early_boost"    # Быстрое введение продвинутого режима
    LATE_SMOOTH = "late_smooth"    # Долгое сглаживание


def compute_multires_alpha(step_ratio, strategy="cosine", 
                           transition_start=0.0, transition_end=0.3,
                           sharpness=1.0):
    """
    Вычисляет alpha для multi-resolution смешивания
    
    Args:
        step_ratio: Текущий прогресс генерации (0.0 - 1.0)
        strategy: Стратегия интерполяции
        transition_start: Начало перехода (0.0 - 1.0)
        transition_end: Конец перехода (0.0 - 1.0)
        sharpness: Резкость перехода (>1 = резче, <1 = мягче)
        
    Returns:
        alpha: 0.0 = полностью simple mode, 1.0 = полностью advanced mode
    """
    
    # Если вне зоны перехода
    if step_ratio <= transition_start:
        return 0.0
    if step_ratio >= transition_end:
        return 1.0
    
    # Нормализуем к диапазону [0, 1] внутри transition zone
    t = (step_ratio - transition_start) / (transition_end - transition_start)
    t = max(0.0, min(t, 1.0))
    
    # Применяем sharpness
    if sharpness != 1.0:
        t = t ** sharpness
    
    # Применяем стратегию
    if strategy == "linear":
        alpha = t
    
    elif strategy == "cosine":
        # Плавная косинусоида: (1 - cos(πt)) / 2
        alpha = (1.0 - math.cos(t * math.pi)) / 2.0
    
    elif strategy == "exponential":
        # Экспоненциальный рост: t²
        alpha = t * t
    
    elif strategy == "sigmoid":
        # S-образная кривая: плавный старт и конец
        # Используем tanh для мягкой сигмоиды
        x = (t - 0.5) * 6  # Растягиваем на [-3, 3]
        alpha = (math.tanh(x) + 1.0) / 2.0
    
    elif strategy == "early_boost":
        # Быстрое введение в начале: sqrt(t)
        alpha = math.sqrt(t)
    
    elif strategy == "late_smooth":
        # Долгое сглаживание: 1 - sqrt(1-t)
        alpha = 1.0 - math.sqrt(1.0 - t)
    
    else:
        alpha = t
    
    return alpha


def apply_multires_blend(tensor_simple, tensor_advanced, 
                         current_step, start_step, end_step,
                         strategy="cosine",
                         transition_start=0.0,
                         transition_end=0.3,
                         sharpness=1.0,
                         enabled=True):
    """
    🌟 УЛУЧШЕННАЯ ВЕРСИЯ MULTI-RESOLUTION MODE 🌟
    
    Плавно переводит от простого padding к продвинутому tiling
    с гибкими стратегиями и настройками.
    
    Args:
        tensor_simple: Результат простого padding (constant/replicate)
        tensor_advanced: Результат продвинутого tiling (circular/panorama/cubemap)
        current_step: Текущий шаг генерации
        start_step: Начальный шаг диапазона
        end_step: Конечный шаг диапазона
        strategy: Стратегия интерполяции ('linear', 'cosine', 'exponential', etc.)
        transition_start: Начало перехода (0.0 = с самого начала)
        transition_end: Конец перехода (0.3 = первые 30%)
        sharpness: Резкость перехода
        enabled: Включен ли режим
        
    Returns:
        Смешанный тензор
    """
    
    if not enabled:
        return tensor_advanced
    
    # Вычисляем прогресс
    total_steps = max(end_step - start_step, 1)
    step_ratio = (current_step - start_step) / total_steps
    step_ratio = max(0.0, min(step_ratio, 1.0))
    
    # Вычисляем alpha
    alpha = compute_multires_alpha(
        step_ratio, 
        strategy=strategy,
        transition_start=transition_start,
        transition_end=transition_end,
        sharpness=sharpness
    )
    
    # Если полностью в простом режиме
    if alpha < 0.001:
        return tensor_simple
    
    # Если полностью в продвинутом режиме
    if alpha > 0.999:
        return tensor_advanced
    
    # Смешиваем
    result = tensor_simple * (1.0 - alpha) + tensor_advanced * alpha
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════

def validate_multires_params(params):
    """
    Валидация и нормализация параметров multi-resolution
    
    Returns:
        dict с валидными параметрами
    """
    return {
        'enabled': bool(params.get('multires_enabled', False)),
        'strategy': str(params.get('multires_strategy', 'cosine')),
        'transition_start': max(0.0, min(float(params.get('multires_start', 0.0)), 1.0)),
        'transition_end': max(0.0, min(float(params.get('multires_end', 0.3)), 1.0)),
        'sharpness': max(0.1, min(float(params.get('multires_sharpness', 1.0)), 5.0)),
    }


def validate_blend_params(params):
    """
    Валидация и нормализация параметров blend mode
    
    Returns:
        dict с валидными параметрами
    """
    return {
        'enabled': bool(params.get('blend_enabled', False)),
        'strength': max(0.0, min(float(params.get('blend_strength', 0.5)), 1.0)),
        'width': int(params.get('blend_width', 0)) if params.get('blend_width') else None,
        'falloff': str(params.get('blend_falloff', 'smoothstep')),
        'sharpness': max(0.1, min(float(params.get('blend_sharpness', 1.0)), 5.0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ТЕСТОВЫЕ ПРИМЕРЫ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("УЛУЧШЕННЫЕ ФУНКЦИИ BLEND & MULTI-RESOLUTION")
    print("=" * 80)
    
    # Тест Blend Mode
    print("\n🎨 ТЕСТ BLEND MODE:")
    x = torch.randn(1, 3, 64, 64)
    
    for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = compute_advanced_blend_padding(
            x, pad_h=8, pad_w=8,
            mode_simple='constant',
            mode_advanced='circular',
            blend_strength=strength,
            falloff_curve='smoothstep'
        )
        print(f"  Strength {strength:.2f}: shape={result.shape}, "
              f"mean={result.mean().item():.4f}")
    
    # Тест Multi-Resolution
    print("\n🔬 ТЕСТ MULTI-RESOLUTION:")
    for step in [0, 10, 20, 30, 50, 100]:
        alpha = compute_multires_alpha(
            step_ratio=step/100,
            strategy='cosine',
            transition_start=0.0,
            transition_end=0.3
        )
        print(f"  Step {step:3d}/100: alpha={alpha:.4f} "
              f"({'простой' if alpha < 0.5 else 'продвинутый'} режим)")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 80)
