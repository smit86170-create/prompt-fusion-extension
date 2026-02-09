import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

# =======================================================================
# 1. SMART CACHING (Оптимизация скорости ~15-20%)
# =======================================================================
class SmartMaskCache:
    def __init__(self, max_size=50):
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
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# Глобальный кэш
_MASK_CACHE = SmartMaskCache()

# =======================================================================
# 2. LATENT COLOR FIX (Исправление серости/блеклости на швах)
# =======================================================================
def blend_with_variance_fix(a, b, mask):
    """
    Математически корректирует смешивание латентных шумов.
    Обычное линейное смешивание (a*m + b*(1-m)) уменьшает дисперсию шума,
    из-за чего нейросеть видит "серые пятна". 
    Эта функция восстанавливает энергию сигнала.
    """
    # 1. Линейное смешивание
    blended = a * mask + b * (1 - mask)
    
    # 2. Коррекция дисперсии (Contrast preservation)
    # Формула восстановления: 1 / sqrt(m^2 + (1-m)^2)
    # Добавляем epsilon (1e-6) чтобы избежать деления на ноль
    variance_fix = torch.sqrt(mask**2 + (1 - mask)**2 + 1e-6)
    
    return blended / variance_fix

# =======================================================================
# 3. LEGACY FADE TO BLACK (Для ZOOM эффекта) ✅ КРИТИЧЕСКИ ВАЖНО!
# =======================================================================
def compute_blend_fade_to_black(padded, pad_h, pad_w, blend_strength=0.1):
    """
    ⚡ LEGACY РЕЖИМ ⚡
    
    Воссоздает поведение старого скрипта asymmetric_tiling_engineC.
    Затемняет края ПОВЕРХ уже примененного паддинга, создавая плавный
    переход к черному (0.0).
    
    ЭТО СОЗДАЕТ ЭФФЕКТ ZOOM OUT!
    
    Работает следующим образом:
    1. Получает тензор С УЖЕ ПРИМЕНЕННЫМ паддингом
    2. Вычисляет зоны затемнения на краях
    3. УМНОЖАЕТ края на маску (0.0 -> 1.0)
    4. Результат: плавный переход от контента к черному
    
    В латентном пространстве SD это интерпретируется как "пустота",
    которую нужно заполнить новым контентом → эффект отдаления камеры.
    
    Args:
        padded: Тензор С УЖЕ ПРИМЕНЕННЫМ паддингом (b, c, H, W)
        pad_h: Размер вертикального паддинга
        pad_w: Размер горизонтального паддинга
        blend_strength: Сила затемнения (0.0-1.0), обычно 0.05-0.2
    
    Returns:
        Тензор с затемненными краями
    """
    b, c, H, W = padded.shape

    # Вычисляем размер оригинального контента (до паддинга)
    h = max(H - 2 * pad_h, 0)
    w = max(W - 2 * pad_w, 0)

    if h <= 0 or w <= 0:
        return padded

    # Размер зоны затемнения (часть от размера контента)
    blend_w = min(int(w * blend_strength), w)
    blend_h = min(int(h * blend_strength), h)

    # Создаем копию чтобы не модифицировать in-place
    result = padded.clone()

    # ═══════════════════════════════════════════════════════════════════
    # ВЕРТИКАЛЬНЫЕ КРАЯ (верх и низ)
    # ═══════════════════════════════════════════════════════════════════
    if blend_h > 0 and pad_h > 0:
        # Верхний край: 0.0 -> 1.0 (затемнен -> полная яркость)
        top_mask = torch.linspace(0.0, 1.0, steps=blend_h, 
                                 device=padded.device, dtype=padded.dtype)
        top_mask = top_mask.view(1, 1, blend_h, 1).expand(b, c, blend_h, W)
        
        # Применяем маску (умножаем!)
        # Это затемняет верхний край контента
        result[:, :, pad_h:pad_h + blend_h, :] *= top_mask

        # Нижний край: 1.0 -> 0.0 (полная яркость -> затемнен)
        bottom_mask = torch.linspace(1.0, 0.0, steps=blend_h, 
                                    device=padded.device, dtype=padded.dtype)
        bottom_mask = bottom_mask.view(1, 1, blend_h, 1).expand(b, c, blend_h, W)
        
        result[:, :, pad_h + h - blend_h:pad_h + h, :] *= bottom_mask

    # ═══════════════════════════════════════════════════════════════════
    # ГОРИЗОНТАЛЬНЫЕ КРАЯ (лево и право)
    # ═══════════════════════════════════════════════════════════════════
    if blend_w > 0 and pad_w > 0:
        # Левый край: 0.0 -> 1.0
        left_mask = torch.linspace(0.0, 1.0, steps=blend_w, 
                                  device=padded.device, dtype=padded.dtype)
        left_mask = left_mask.view(1, 1, 1, blend_w).expand(b, c, H, blend_w)
        
        result[:, :, :, pad_w:pad_w + blend_w] *= left_mask

        # Правый край: 1.0 -> 0.0
        right_mask = torch.linspace(1.0, 0.0, steps=blend_w, 
                                   device=padded.device, dtype=padded.dtype)
        right_mask = right_mask.view(1, 1, 1, blend_w).expand(b, c, H, blend_w)
        
        result[:, :, :, pad_w + w - blend_w:pad_w + w] *= right_mask

    return result

# =======================================================================
# 4. ADVANCED BLEND MASK (Для современного режима)
# =======================================================================
def create_advanced_blend_mask(h, w, blend_width, device, dtype=torch.float32, 
                               falloff_curve="smoothstep", edge_sharpness=1.0):
    """
    Создает кэшированную маску для смешивания краев.
    Используется в режиме "смешивания двух паддингов" (НЕ для zoom).
    """
    if blend_width <= 0:
        return torch.ones(1, 1, h, w, device=device, dtype=dtype)

    # Ограничиваем ширину чтобы маски не схлопнулись
    blend_w = min(blend_width, w // 2)
    blend_h = min(blend_width, h // 2)
    
    mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    
    # Генерация линейного градиента
    def get_ramp(size):
        t = torch.linspace(0, 1, steps=size, device=device, dtype=dtype)
        # Применяем резкость
        if edge_sharpness != 1.0:
            t = torch.pow(t, edge_sharpness)
        
        # Применяем кривую
        if falloff_curve == 'smoothstep':
            return t * t * (3 - 2 * t)
        elif falloff_curve == 'cosine':
            return (1 - torch.cos(t * math.pi)) / 2
        elif falloff_curve == 'linear':
            return t
        return t

    # Заполняем вертикальные края
    if blend_w > 0:
        ramp = get_ramp(blend_w)
        mask[:, :, :, :blend_w] = torch.maximum(mask[:, :, :, :blend_w], 
                                                 ramp.flip(0).view(1,1,1,-1))
        mask[:, :, :, -blend_w:] = torch.maximum(mask[:, :, :, -blend_w:], 
                                                  ramp.view(1,1,1,-1))

    # Заполняем горизонтальные края
    if blend_h > 0:
        ramp = get_ramp(blend_h)
        mask[:, :, :blend_h, :] = torch.maximum(mask[:, :, :blend_h, :], 
                                                 ramp.flip(0).view(1,1,-1,1))
        mask[:, :, -blend_h:, :] = torch.maximum(mask[:, :, -blend_h:, :], 
                                                  ramp.view(1,1,-1,1))
        
    return mask

# =======================================================================
# 5. IMPROVED BLEND PADDING (Смешивание двух паддингов)
# =======================================================================
def compute_advanced_blend_padding(input_tensor, pad_h, pad_w, 
                                 mode_simple='replicate', mode_advanced='circular',
                                 blend_strength=0.5, blend_width=None, 
                                 falloff_curve='smoothstep', edge_sharpness=1.0,
                                 fade_to_black=False, fade_strength=0.1):
    """
    УЛУЧШЕННЫЙ РЕЖИМ ПАДДИНГА.
    
    Два режима работы:
    
    1. FADE TO BLACK (fade_to_black=True) - для Zoom эффекта:
       - Применяет один паддинг (mode_advanced)
       - ЗАТЕМНЯЕТ края, создавая эффект zoom out
       - Использует legacy функцию compute_blend_fade_to_black
       
    2. BLEND TWO PADDINGS (fade_to_black=False) - для качественных краев:
       - Создает два разных паддинга (simple и advanced)
       - Смешивает их через маску
       - Применяет variance fix для исправления цвета
       - НЕ создает zoom эффекта
    
    Args:
        input_tensor: Исходный тензор БЕЗ паддинга
        pad_h, pad_w: Размеры паддинга
        mode_simple: Режим для "простого" паддинга ('replicate', 'constant')
        mode_advanced: Режим для "продвинутого" паддинга ('circular', 'reflect')
        blend_strength: Сила смешивания (0.0-1.0)
        fade_to_black: Если True, использует legacy режим затемнения
        fade_strength: Сила затемнения для fade_to_black режима
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ 1: FADE TO BLACK (для Zoom)
    # ═══════════════════════════════════════════════════════════════════
    if fade_to_black:
        # Применяем ОДИН паддинг
        if isinstance(mode_advanced, str):
            if mode_advanced == 'reflect':
                # Безопасная проверка
                b, c, h, w = input_tensor.shape
                if pad_w < w and pad_h < h:
                    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                else:
                    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            else:  # circular
                padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_advanced)
        else:
            padded = mode_advanced  # Если это тензор
        
        # Затемняем края (LEGACY режим!)
        return compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength)
    
    # ═══════════════════════════════════════════════════════════════════
    # РЕЖИМ 2: BLEND TWO PADDINGS (стандартный)
    # ═══════════════════════════════════════════════════════════════════
    
    # Если выключено или нет силы
    if blend_strength <= 0.001:
        if mode_simple == 'constant':
            return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_simple)

    # Если 100% сила
    if blend_strength >= 0.999:
        if isinstance(mode_advanced, str):
            if mode_advanced == 'reflect':
                b, c, h, w = input_tensor.shape
                if pad_w < w and pad_h < h:
                    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                else:
                    return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            return F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_advanced)
        return mode_advanced

    # 1. Готовим слои
    # Simple Layer
    if mode_simple == 'constant':
        simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    else:
        simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_simple)
        
    # Advanced Layer
    if isinstance(mode_advanced, str):
        if mode_advanced == 'reflect':
            b, c, h, w = input_tensor.shape
            if pad_w < w and pad_h < h:
                advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            else:
                advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        else:  # circular
            advanced = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    else:
        advanced = mode_advanced

    # 2. Получаем маску из кэша
    if blend_width is None: 
        blend_width = max(pad_h, pad_w)
    
    b, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    cache_key = (h, w, pad_h, pad_w, blend_width, falloff_curve, edge_sharpness, str(device))
    mask = _MASK_CACHE.get(cache_key)
    
    if mask is None:
        H_pad, W_pad = simple.shape[2:]
        mask = create_advanced_blend_mask(H_pad, W_pad, blend_width, device, 
                                         input_tensor.dtype, falloff_curve, edge_sharpness)
        _MASK_CACHE.set(cache_key, mask)

    # 3. Смешиваем с коррекцией цвета (Variance Fix)
    final_mask = mask * blend_strength
    return blend_with_variance_fix(simple, advanced, final_mask)


# =======================================================================
# 6. MULTI-RES (Стабильная версия)
# =======================================================================
class BlendStrategy:
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"

class MultiResStrategy:
    def __init__(self, strategy_type=BlendStrategy.COSINE):
        self.strategy_type = strategy_type

    def get_factor(self, progress, sharpness=1.0):
        t = max(0.0, min(1.0, progress))
        if sharpness != 1.0:
            t = math.pow(t, sharpness)

        if self.strategy_type == BlendStrategy.LINEAR:
            return t
        elif self.strategy_type == BlendStrategy.COSINE:
            return (1.0 - math.cos(t * math.pi)) / 2.0
        elif self.strategy_type == BlendStrategy.EXPONENTIAL:
            return math.pow(t, 2)
        elif self.strategy_type == BlendStrategy.SIGMOID:
            if t <= 0: return 0.0
            if t >= 1: return 1.0
            return 1.0 / (1.0 + math.exp(-12.0 * (t - 0.5)))
        return t

def apply_multires_blend(tensor_simple, tensor_advanced, current_step, start_step, end_step, 
                        strategy="cosine", transition_start=0.0, transition_end=0.3, 
                        sharpness=1.0, enabled=False):
    if not enabled:
        return tensor_advanced

    total_steps = end_step - start_step
    if total_steps <= 0: 
        return tensor_advanced

    step_frac = (current_step - start_step) / total_steps
    step_frac = max(0.0, min(1.0, step_frac))
    
    if step_frac < transition_start:
        local_progress = 0.0
    elif step_frac > transition_end:
        local_progress = 1.0
    else:
        duration = transition_end - transition_start
        if duration <= 0:
            local_progress = 1.0
        else:
            local_progress = (step_frac - transition_start) / duration

    strat = MultiResStrategy(strategy)
    alpha = strat.get_factor(local_progress, sharpness)

    if alpha <= 0.001: return tensor_simple
    if alpha >= 0.999: return tensor_advanced

    # Здесь обычный lerp, так как это переход во времени, а не в пространстве
    return tensor_simple * (1.0 - alpha) + tensor_advanced * alpha

# Helpers for params validation
def validate_blend_params(params):
    return {
        'strength': float(params.get('blend_strength', 0.5)),
        'width': int(params.get('blend_width', 0)) if params.get('blend_width') else None,
        'falloff': params.get('blend_falloff', 'smoothstep'),
        'sharpness': float(params.get('blend_sharpness', 1.0)),
        'fade_to_black': bool(params.get('blend_fade_to_black', False)),
        'fade_strength': float(params.get('blend_fade_strength', 0.1))
    }

def validate_multires_params(params):
    return {
        'strategy': params.get('multires_strategy', 'cosine'),
        'transition_start': float(params.get('multires_start', 0.0)),
        'transition_end': float(params.get('multires_end', 0.3)),
        'sharpness': float(params.get('multires_sharpness', 1.0))
    }