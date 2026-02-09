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


def get_safe_epsilon(tensor_or_dtype):
    """
    Float16-safe epsilon - КРИТИЧНО для работы с half precision!
    
    Args:
        tensor_or_dtype: torch.Tensor или torch.dtype
    
    Returns:
        float: безопасный epsilon для данного типа
    """
    if isinstance(tensor_or_dtype, torch.Tensor):
        dtype = tensor_or_dtype.dtype
    else:
        dtype = tensor_or_dtype
    
    # Float16 минимальное значение ~6e-5, поэтому get_safe_epsilon(torch.float16) → 0 (underflow)
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3  # Безопасно для half precision
    elif dtype == torch.float32:
        return get_safe_epsilon(torch.float16)  # Стандартный epsilon
    else:
        return 1e-12  # Высокая точность для float64

def get_safe_epsilon(tensor_or_dtype):
    """Float16-safe epsilon"""
    if isinstance(tensor_or_dtype, torch.Tensor):
        dtype = tensor_or_dtype.dtype
    else:
        dtype = tensor_or_dtype
    
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    elif dtype == torch.float32:
        return get_safe_epsilon(torch.float16)
    else:
        return 1e-12

# =======================================================================
# 2. LATENT COLOR FIX (Исправление серости/блеклости на швах)
# =======================================================================
def blend_with_variance_fix(a, b, mask):
    """
    Математически корректирует смешивание латентных шумов.
    ✅ FLOAT16 FIX: Безопасный sqrt с адаптивным epsilon
    
    Args:
        a: Основной слой (активен там, где mask=1) -> Advanced/Circular
        b: Фоновый слой (активен там, где mask=0) -> Simple/Replicate
        mask: Маска смешивания [0,1]
    """
    # 1. Линейное смешивание
    blended = a * mask + b * (1 - mask)
    
    # 2. Коррекция дисперсии с адаптивным epsilon
    eps_val = get_safe_epsilon(mask.dtype)
    variance_fix = torch.sqrt(mask**2 + (1 - mask)**2 + eps_val)
    
    return blended / variance_fix

# =======================================================================
# 3. LEGACY FADE TO BLACK (Для ZOOM эффекта) ✅
# =======================================================================
def compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength=0.1):
    """
    ⚡ LEGACY РЕЖИМ для Zoom эффекта (V3.5 FIX) ⚡
    
    Теперь градиент покрывает ВЕСЬ паддинг и часть контента.
    Результат: Красивая виньетка от 0 (край) до 1 (центр).
    """
    b, c, H, W = padded.shape
    
    # Вычисляем размер контента (без паддинга)
    h_content = max(H - 2 * pad_h, 0)
    w_content = max(W - 2 * pad_w, 0)
    
    # Глубина смешивания внутри контента
    blend_in_h = int(h_content * fade_strength)
    blend_in_w = int(w_content * fade_strength)
    
    # Общая зона затемнения = Паддинг + Вход в контент
    total_fade_h = pad_h + blend_in_h
    total_fade_w = pad_w + blend_in_w
    
    result = padded.clone()
    
    # ═══════════════════════════════════════════════════════════════════
    # ВЕРТИКАЛЬНЫЕ КРАЯ
    # ═══════════════════════════════════════════════════════════════════
    if total_fade_h > 0:
        # Создаем градиент 0 -> 1
        fade = torch.linspace(0, 1, steps=total_fade_h, device=padded.device, dtype=padded.dtype)
        fade = fade.view(1, 1, -1, 1) # (1,1,H,1)
        
        # Верх (от 0 до total_fade_h)
        safe_h = min(total_fade_h, H)
        result[:, :, :safe_h, :] *= fade[:, :, :safe_h, :]
        
        # Низ (от H-total_fade_h до H) - используем перевернутый градиент
        result[:, :, -safe_h:, :] *= fade[:, :, :safe_h, :].flip(2)

    # ═══════════════════════════════════════════════════════════════════
    # ГОРИЗОНТАЛЬНЫЕ КРАЯ
    # ═══════════════════════════════════════════════════════════════════
    if total_fade_w > 0:
        # Создаем градиент 0 -> 1
        fade = torch.linspace(0, 1, steps=total_fade_w, device=padded.device, dtype=padded.dtype)
        fade = fade.view(1, 1, 1, -1) # (1,1,1,W)
        
        # Лево
        safe_w = min(total_fade_w, W)
        result[:, :, :, :safe_w] *= fade[:, :, :, :safe_w]
        
        # Право
        result[:, :, :, -safe_w:] *= fade[:, :, :, :safe_w].flip(3)

    return result

# =======================================================================
# 4. ADVANCED BLEND MASK (Для современного режима)
# =======================================================================
def create_advanced_blend_mask(h, w, blend_width, device, dtype=torch.float32, 
                               falloff_curve="smoothstep", edge_sharpness=1.0):
    """
    Создает кэшированную маску для смешивания краев.
    1.0 = на самом краю (где нужен Advanced Padding)
    0.0 = в центре (где контент или Simple Padding)
    """
    if blend_width <= 0:
        return torch.ones(1, 1, h, w, device=device, dtype=dtype)

    blend_w = min(blend_width, w // 2)
    blend_h = min(blend_width, h // 2)
    
    mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    
    # Генерация линейного градиента
    def get_ramp(size):
        t = torch.linspace(0, 1, steps=size, device=device, dtype=dtype)
        if edge_sharpness != 1.0:
            t = torch.pow(t, edge_sharpness)
        
        if falloff_curve == 'smoothstep':
            return t * t * (3 - 2 * t)
        elif falloff_curve == 'cosine':
            return (1 - torch.cos(t * math.pi)) / 2
        elif falloff_curve == 'linear':
            return t
        return t

    # Заполняем края
    if blend_w > 0:
        ramp = get_ramp(blend_w)
        # Левый край
        mask[:, :, :, :blend_w] = torch.maximum(mask[:, :, :, :blend_w], 
                                                 ramp.flip(0).view(1,1,1,-1))
        # Правый край
        mask[:, :, :, -blend_w:] = torch.maximum(mask[:, :, :, -blend_w:], 
                                                  ramp.view(1,1,1,-1))

    if blend_h > 0:
        ramp = get_ramp(blend_h)
        # Верхний край
        mask[:, :, :blend_h, :] = torch.maximum(mask[:, :, :blend_h, :], 
                                                 ramp.flip(0).view(1,1,-1,1))
        # Нижний край
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
    УЛУЧШЕННЫЙ РЕЖИМ ПАДДИНГА
    """
    
    # РЕЖИМ 1: FADE TO BLACK (Zoom)
    if fade_to_black:
        # Сначала делаем обычный паддинг
        if isinstance(mode_advanced, str):
            if mode_advanced == 'reflect':
                b, c, h, w = input_tensor.shape
                if pad_w < w and pad_h < h:
                    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                else:
                    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            else:  # circular
                padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_advanced)
        else:
            padded = mode_advanced
        
        # Затемняем края (теперь работает правильно!)
        return compute_blend_fade_to_black(padded, pad_h, pad_w, fade_strength)
    
    # РЕЖИМ 2: BLEND TWO PADDINGS (Tiling)
    
    # Если выключено
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
    if mode_simple == 'constant':
        simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    else:
        simple = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=mode_simple)
        
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
    dtype = input_tensor.dtype
    
    cache_key = (h, w, pad_h, pad_w, blend_width, falloff_curve, edge_sharpness, 
                 str(device), str(dtype))
    mask = _MASK_CACHE.get(cache_key)
    
    if mask is None:
        H_pad, W_pad = simple.shape[2:]
        mask = create_advanced_blend_mask(H_pad, W_pad, blend_width, device, 
                                         dtype, falloff_curve, edge_sharpness)
        _MASK_CACHE.set(cache_key, mask)

    # 3. Смешиваем
    final_mask = mask * blend_strength
    
    # advanced (mask=1 на краях) / simple (mask=0 в центре)
    return blend_with_variance_fix(advanced, simple, final_mask)


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

# =======================================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Возвращены для совместимости с тестами)
# =======================================================================

def create_circular_mask(h, w, center_x=0.5, center_y=0.5, radius=0.5, device='cpu', dtype=torch.float32):
    """
    Создает круговую маску (белый круг на черном фоне).
    ✅ FLOAT16 FIX: Безопасный sqrt
    """
    # ✅ FIX: Получаем безопасный epsilon
    eps_val = get_safe_epsilon(dtype)
    
    # Создаем сетку координат
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # Сдвигаем центр
    x = x - (center_x - 0.5) * 2
    y = y - (center_y - 0.5) * 2
    
    # ✅ FIX: Считаем расстояние от центра с защитой
    dist = torch.sqrt(x*x + y*y + eps_val)
    
    # Создаем мягкую маску (сглаживание краев 0.1)
    mask = 1.0 - torch.clamp((dist - (radius - 0.1)) / 0.2, 0, 1)
    
    # Добавляем размерность каналов и батча
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
        
    return mask

def create_fade_to_black_mask(h, w, strength=0.1, device='cpu', dtype=torch.float32):
    """
    Создает виньетку (затемнение к краям).
    ✅ FLOAT16 FIX: Безопасный sqrt
    """
    # ✅ FIX: Получаем безопасный epsilon
    eps_val = get_safe_epsilon(dtype)
    
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # ✅ FIX: sqrt с защитой
    dist = torch.sqrt(x*x + y*y + eps_val)
    
    # Нормализуем, чтобы углы были 1.0 (максимальное расстояние ~1.41)
    dist = dist / 1.4142
    
    # Инвертируем: центр белый (1), края черные (0)
    threshold = 1.0 - strength
    mask = 1.0 - torch.clamp((dist - threshold) / strength, 0, 1)
    
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
        
    return mask