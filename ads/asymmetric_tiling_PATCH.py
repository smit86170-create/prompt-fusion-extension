#!/usr/bin/env python3
"""
ПАТЧ ДЛЯ ИСПРАВЛЕНИЯ ПРОБЛЕМЫ РАЗМЕРОВ ТЕНЗОРОВ В ADVANCED TILING v3.0
==========================================================================

Этот файл содержит патч для исправления ошибки:
RuntimeError: The size of tensor a (102) must match the size of tensor b (98)

ИСПОЛЬЗОВАНИЕ:
1. Поместите этот файл в ту же папку, что и asymmetric_tiling_UNIFIED__24_.py
2. Запустите: python asymmetric_tiling_PATCH.py
3. Файл будет автоматически пропатчен

РЕЗЕРВНАЯ КОПИЯ:
Оригинальный файл будет сохранен как asymmetric_tiling_UNIFIED__24_.py.backup
"""

import os
import sys
import shutil
from datetime import datetime

def create_backup(filepath):
    """Создает резервную копию файла"""
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Резервная копия создана: {backup_path}")
    return backup_path

def apply_patch(filepath):
    """Применяет патч к файлу"""
    if not os.path.exists(filepath):
        print(f"✗ Файл не найден: {filepath}")
        return False
    
    # Создаем резервную копию
    create_backup(filepath)
    
    # Читаем файл
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем, не был ли патч уже применен
    if 'ensure_output_size_match' in content:
        print("⚠ Патч уже был применен ранее!")
        return True
    
    # Находим позицию для вставки вспомогательных функций
    import_section_end = content.find('# ========================================================================')
    if import_section_end == -1:
        import_section_end = content.find('def custom_padding_forward')
    
    # Вспомогательные функции
    helper_functions = '''
# ========================================================================
# ИСПРАВЛЕНИЕ: Функции для обеспечения правильных размеров тензоров
# ========================================================================

def ensure_output_size_match(output_tensor, expected_h, expected_w):
    """
    Гарантирует, что выходной тензор имеет ожидаемые размеры.
    Если размеры не совпадают, применяет interpolation или padding/cropping.
    
    Args:
        output_tensor: Выходной тензор после свертки (B, C, H, W)
        expected_h: Ожидаемая высота
        expected_w: Ожидаемая ширина
    
    Returns:
        Тензор с правильными размерами
    """
    b, c, h, w = output_tensor.shape
    
    if h == expected_h and w == expected_w:
        return output_tensor
    
    # Вычисляем разницу
    diff_h = h - expected_h
    diff_w = w - expected_w
    
    # Если разница небольшая (±5 пикселей), используем padding/cropping
    if abs(diff_h) <= 5 and abs(diff_w) <= 5:
        if diff_h > 0 or diff_w > 0:
            # Нужно обрезать
            crop_h_start = diff_h // 2
            crop_w_start = diff_w // 2
            crop_h_end = crop_h_start + expected_h
            crop_w_end = crop_w_start + expected_w
            
            # Проверяем границы
            if crop_h_end <= h and crop_w_end <= w:
                return output_tensor[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            else:
                # Если не можем обрезать, используем interpolation
                return F.interpolate(output_tensor, size=(expected_h, expected_w), 
                                   mode='bilinear', align_corners=False)
        else:
            # Нужно добавить padding
            pad_h = (-diff_h) // 2
            pad_w = (-diff_w) // 2
            # Добавляем остаток справа/снизу если нечетное
            pad_h_extra = (-diff_h) % 2
            pad_w_extra = (-diff_w) % 2
            return F.pad(output_tensor, 
                        (pad_w, pad_w + pad_w_extra, pad_h, pad_h + pad_h_extra),
                        mode='replicate')
    else:
        # Если разница большая, используем interpolation
        print(f"[Advanced Tiling] Корректируем размер: {h}x{w} -> {expected_h}x{expected_w}")
        return F.interpolate(output_tensor, size=(expected_h, expected_w), 
                           mode='bilinear', align_corners=False)


def compute_expected_output_size(input_h, input_w, kernel_size, stride, padding, dilation):
    """
    Вычисляет ожидаемый размер выходного тензора после свертки.
    
    Formula: output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    """
    k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
    d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    
    if isinstance(padding, (tuple, list)):
        p_h, p_w = padding[0], padding[1]
    elif isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = 0, 0
    
    out_h = ((input_h + 2*p_h - d_h*(k_h-1) - 1) // s_h) + 1
    out_w = ((input_w + 2*p_w - d_w*(k_w-1) - 1) // s_w) + 1
    
    return out_h, out_w

'''
    
    # Вставляем вспомогательные функции
    content = content[:import_section_end] + helper_functions + '\n' + content[import_section_end:]
    
    # Теперь нужно пропатчить функцию custom_padding_forward
    # Находим начало функции
    func_start = content.find('def custom_padding_forward(input_tensor')
    if func_start == -1:
        print("✗ Не найдена функция custom_padding_forward!")
        return False
    
    # Находим начало блока try внутри функции
    try_block_start = content.find('try:', func_start)
    
    # Добавляем код для вычисления ожидаемых размеров после блока try:
    size_computation_code = '''
        # ===== ИСПРАВЛЕНИЕ: Вычисляем ожидаемый размер выходного тензора =====
        b, c, orig_h, orig_w = input_tensor.shape
        k_h, k_w = weight.shape[2], weight.shape[3]
        
        # Обрабатываем разные форматы padding для вычисления ожидаемого размера
        if isinstance(padding, str):
            if padding == 'same':
                s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
                expected_out_h = (orig_h + s_h - 1) // s_h
                expected_out_w = (orig_w + s_w - 1) // s_w
            elif padding == 'valid':
                s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
                d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
                expected_out_h = ((orig_h - d_h*(k_h-1) - 1) // s_h) + 1
                expected_out_w = ((orig_w - d_w*(k_w-1) - 1) // s_w) + 1
            else:
                expected_out_h = orig_h
                expected_out_w = orig_w
        else:
            p_val = padding if isinstance(padding, int) else (padding[0] if isinstance(padding, (tuple, list)) else 0)
            expected_out_h, expected_out_w = compute_expected_output_size(
                orig_h, orig_w, (k_h, k_w), stride, (p_val, p_val), dilation
            )
        # ===== КОНЕЦ ИСПРАВЛЕНИЯ =====
        
'''
    
    # Вставляем после try:
    content = content[:try_block_start + 5] + size_computation_code + content[try_block_start + 5:]
    
    # Теперь нужно добавить проверку размеров перед каждым return
    # Находим все return statements в функции custom_padding_forward
    
    # Простой подход: добавляем проверку перед финальными return
    # Ищем паттерны типа "return F.conv2d" или "return out_"
    
    # Заменяем паттерны return
    patterns_to_replace = [
        ('return out_cube', '''out_cube = ensure_output_size_match(out_cube, expected_out_h, expected_out_w)
            return out_cube'''),
        ('return out_pano', '''out_pano = ensure_output_size_match(out_pano, expected_out_h, expected_out_w)
            return out_pano'''),
        ('return out_advanced', '''out_advanced = ensure_output_size_match(out_advanced, expected_out_h, expected_out_w)
            return out_advanced'''),
        ('return out_final', '''out_final = ensure_output_size_match(out_final, expected_out_h, expected_out_w)
            return out_final'''),
    ]
    
    for old, new in patterns_to_replace:
        content = content.replace(old, new)
    
    # Записываем пропатченный файл
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Патч успешно применен!")
    return True


if __name__ == '__main__':
    # Путь к файлу, который нужно пропатчить
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, 'asymmetric_tiling_UNIFIED__24_.py')
    
    print("═" * 70)
    print("ПАТЧ ДЛЯ ADVANCED TILING v3.0 - ИСПРАВЛЕНИЕ РАЗМЕРОВ ТЕНЗОРОВ")
    print("═" * 70)
    print()
    
    if not os.path.exists(target_file):
        print(f"✗ Файл не найден: {target_file}")
        print()
        print("Убедитесь, что этот скрипт находится в той же папке, что и:")
        print("asymmetric_tiling_UNIFIED__24_.py")
        sys.exit(1)
    
    print(f"Файл для патча: {target_file}")
    print()
    
    # Применяем патч
    if apply_patch(target_file):
        print()
        print("═" * 70)
        print("✓ ПАТЧ УСПЕШНО ПРИМЕНЕН!")
        print("═" * 70)
        print()
        print("Что было исправлено:")
        print("  1. Добавлены функции ensure_output_size_match и compute_expected_output_size")
        print("  2. В custom_padding_forward добавлено вычисление ожидаемых размеров")
        print("  3. Все выходные тензоры теперь проверяются и корректируются")
        print()
        print("Теперь проблема 'The size of tensor a must match the size of tensor b'")
        print("должна быть решена!")
        print()
    else:
        print()
        print("✗ Не удалось применить патч")
        print("Проверьте файл и попробуйте применить исправление вручную")
        sys.exit(1)
