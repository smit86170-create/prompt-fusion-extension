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

# Глобальное хранилище
_ORIGINAL_METHODS_CACHE = {}
_MASK_CACHE = {}
_SAMPLER_REGISTERED = False

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
# РАСШИРЕННЫЕ ФУНКЦИИ ПАДДИНГА
# ========================================================================

def get_or_create_mask(h, w, device):
    """Кэширование масок для оптимизации"""
    key = (h, w, str(device))
    if key not in _MASK_CACHE:
        row_indices = torch.arange(h, device=device).view(1, 1, h, 1)
        _MASK_CACHE[key] = (row_indices % 2 == 1)
    return _MASK_CACHE[key]

def compute_anisotropic_padding(input_tensor, pad_h, pad_w, angle_deg=45):
    """
    Анизотропный паддинг - разное поведение по диагоналям.
    Эмулирует направленные материалы (дерево, металл, волокна).
    """
    b, c, h, w = input_tensor.shape

    # Базовые паддинги: circular и reflect
    padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    padded_reflect = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

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

def compute_blend_padding(padded, pad_h, pad_w, blend_strength=0.1):
    """
    Мягкое смешивание краев поверх уже примененного паддинга.
    Работает с любым базовым режимом.
    """
    b, c, H, W = padded.shape

    h = max(H - 2 * pad_h, 0)
    w = max(W - 2 * pad_w, 0)

    if h <= 0 or w <= 0:
        return padded

    blend_w = min(int(w * blend_strength), w)
    blend_h = min(int(h * blend_strength), h)

    if blend_h > 0 and pad_h > 0:
        top_mask = torch.linspace(0.0, 1.0, steps=blend_h, device=padded.device, dtype=padded.dtype)
        top_mask = top_mask.view(1, 1, blend_h, 1).expand(b, c, blend_h, W)
        padded[:, :, pad_h:pad_h + blend_h, :] *= top_mask

        bottom_mask = torch.linspace(1.0, 0.0, steps=blend_h, device=padded.device, dtype=padded.dtype)
        bottom_mask = bottom_mask.view(1, 1, blend_h, 1).expand(b, c, blend_h, W)
        padded[:, :, pad_h + h - blend_h:pad_h + h, :] *= bottom_mask

    if blend_w > 0 and pad_w > 0:
        left_mask = torch.linspace(0.0, 1.0, steps=blend_w, device=padded.device, dtype=padded.dtype)
        left_mask = left_mask.view(1, 1, 1, blend_w).expand(b, c, H, blend_w)
        padded[:, :, :, pad_w:pad_w + blend_w] *= left_mask

        right_mask = torch.linspace(1.0, 0.0, steps=blend_w, device=padded.device, dtype=padded.dtype)
        right_mask = right_mask.view(1, 1, 1, blend_w).expand(b, c, H, blend_w)
        padded[:, :, :, pad_w + w - blend_w:pad_w + w] *= right_mask

    return padded

# ========================================================================
# ГЛАВНАЯ ФУНКЦИЯ ПАДДИНГА
# ========================================================================

def custom_padding_forward(input_tensor, weight, bias, stride, padding, dilation, groups, params):
    """
    Универсальная функция паддинга с поддержкой всех режимов v3.0
    """
    try:
        current_step = getattr(shared.state, 'sampling_step', 0)
        if not (params['start_step'] <= current_step <= params['end_step']):
            return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

        # Если включено "Disable Advanced Tiling during hires pass",
        # отключаем Advanced Tiling на hires-проходе
        script = params.get('script_ref', None)
        if params.get('tiling_disable_hr', False) and script is not None:
            # tiling_enable_hr: включён ли вообще hires.fix
            # tiling_is_hires: находимся ли сейчас на hires-проходе
            if getattr(script, 'tiling_enable_hr', False) and getattr(script, 'tiling_is_hires', False):
                return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

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

        x = input_tensor
        mode_x = params['mode_x']
        mode_y = params['mode_y']

        # ====== СПЕЦИАЛЬНЫЕ РЕЖИМЫ ======

        # Стереоскопия
        if params.get('stereo_enabled', False):
            eye = params.get('stereo_eye', 'left')
            convergence = params.get('stereo_convergence', 0.5)
            separation = params.get('stereo_separation', 0.065)
            x = compute_stereoscopic_padding(x, req_pad_h, req_pad_w, eye, convergence, separation)

        # Анизотропный
        elif mode_x == MODE_ANISOTROPIC or mode_y == MODE_ANISOTROPIC:
            angle = params.get('anisotropic_angle', 45)
            x = compute_anisotropic_padding(x, req_pad_h, req_pad_w, angle)

        # Полярный (сферический)
        elif mode_x == MODE_POLAR or mode_y == MODE_POLAR:
            x = compute_polar_padding(x, req_pad_h, req_pad_w)

        # Панорама
        elif mode_x == MODE_PANORAMA or mode_y == MODE_PANORAMA:
            x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')

        # Стандартные режимы (раздельно по осям)
        else:
            # Ось Y
            if mode_y == MODE_CIRCULAR:
                x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')
            elif mode_y == MODE_MIRROR:
                x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='reflect')
            elif mode_y == MODE_HEXAGONAL:
                x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='circular')
            else:
                x = F.pad(x, (0, 0, req_pad_h, req_pad_h), mode='constant', value=0)

            # Ось X
            if mode_x == MODE_CIRCULAR:
                x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='circular')
            elif mode_x == MODE_MIRROR:
                x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='reflect')
            elif mode_x == MODE_HEXAGONAL:
                x = compute_hex_padding_x(x, req_pad_w, req_pad_w)
            else:
                x = F.pad(x, (req_pad_w, req_pad_w, 0, 0), mode='constant', value=0)

        # Дополнительный мягкий бленд поверх выбранного режима
        if params.get('blend_enabled', False) and (req_pad_h > 0 or req_pad_w > 0):
            blend_str = params.get('blend_strength', 0.1)
            x = compute_blend_padding(x, req_pad_h, req_pad_w, blend_str)

        # Мультиразрешение
        if params.get('multires_enabled', False):
            step_ratio = current_step / max(params['end_step'], 1)
            if step_ratio < 0.3:
                x_default = F.pad(input_tensor, (req_pad_w, req_pad_w, req_pad_h, req_pad_h),
                                  mode='constant', value=0)
                alpha = step_ratio * 3
                x = x * alpha + x_default * (1 - alpha)

        return F.conv2d(x, weight, bias, stride, 0, dilation, groups)

    except Exception as e:
        print(f"Advanced Tiling v3 Error: {e}")
        return F.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)

# ========================================================================
# РЕГИСТРАЦИЯ KOHAKU SAMPLER
# ========================================================================

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
                            choices=[MODE_OFF, MODE_CIRCULAR, MODE_MIRROR, MODE_HEXAGONAL],
                            value=MODE_CIRCULAR
                        )
                        mode_y = gr.Dropdown(
                            label="Mode Y",
                            choices=[MODE_OFF, MODE_CIRCULAR, MODE_MIRROR, MODE_HEXAGONAL],
                            value=MODE_CIRCULAR
                        )

                    with gr.Row():
                        multires = gr.Checkbox(label="Multi-Resolution Mode", value=False)
                        use_blend = gr.Checkbox(label="Soft Blend Edges", value=False)
                        blend_strength = gr.Slider(0.0, 0.5, step=0.05, label="Blend Strength", value=0.1)

                with gr.Tab("🌐 Advanced Modes"):
                    gr.Markdown("**Специальные режимы для сложных топологий**")

                    with gr.Row():
                        use_panorama = gr.Checkbox(label="Panorama 360°", value=False)
                        use_polar = gr.Checkbox(
                            label="Polar (Sphere Correct)",
                            value=False,
                            info="Correct pole transitions for equirectangular"
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
                multires,
                use_panorama, use_polar, use_blend, blend_strength,
                use_anisotropic, aniso_angle,
                stereo_enabled, stereo_eye, stereo_separation, stereo_convergence,
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
                multires,
                use_panorama, use_polar, use_blend, blend_strength,
                use_anisotropic, aniso_angle,
                stereo_enabled, stereo_eye, stereo_separation, stereo_convergence,
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

        # Спецрежимы
        if use_panorama:
            mode_x = mode_y = MODE_PANORAMA
        if use_polar:
            mode_x = mode_y = MODE_POLAR
        if use_anisotropic:
            mode_x = mode_y = MODE_ANISOTROPIC

        params = {
            'mode_x': mode_x,
            'mode_y': mode_y,
            'start_step': start_step,
            'end_step': end_step,
            'multires_enabled': multires,
            'blend_enabled': use_blend,
            'blend_strength': blend_strength,
            'anisotropic_angle': aniso_angle,
            'stereo_enabled': stereo_enabled,
            'stereo_eye': stereo_eye,
            'stereo_separation': stereo_separation,
            'stereo_convergence': stereo_convergence,
            'tiling_disable_hr': tiling_disable_hr,
            'script_ref': self,
        }

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

        if restored > 0:
            print(f"✓ Restored {restored} layers to original state")