import torch
import comfy.utils
import math
import torch.nn.functional as F
import random
import cv2
import numpy as np

class ImageScaleToTotalPixelsRound64:
    upscale_methods = ["bilinear", "nearest-exact", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",), 
            "upscale_method": (s.upscale_methods,),
            "megapixels": ("FLOAT", {"default": 5.63, "min": 0.01, "max": 16.0, "step": 0.01}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "custom_node_experiments"

    def upscale(self, image, upscale_method, megapixels):
        samples = image.movedim(-1,1)
        total = int(megapixels * 1024 * 1024)

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by) // 64 * 64
        height = round(samples.shape[2] * scale_by) // 64 * 64
        
        print("upscale to ", width, height)

        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

class ImageBlendLighter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "custom_node_experiments"

    def blend(self, image1, image2, blend_factor, image3=None, image4=None, image5=None, 
             image6=None, image7=None, image8=None, image9=None):
        # Собираем все непустые изображения в список
        images = [img for img in [image1, image2, image3, image4, image5, 
                                image6, image7, image8, image9] if img is not None]
        
        # Получаем размеры первого изображения как целевые
        batch, target_height, target_width, channels = images[0].shape
        
        # Масштабируем все изображения к размеру первого
        scaled_images = []
        for img in images:
            if img.shape[1:3] != (target_height, target_width):
                # Переставляем размерности для интерполяции [batch, channels, height, width]
                img_channels_first = img.movedim(-1, 1)
                # Масштабируем
                img_resized = F.interpolate(
                    img_channels_first,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False
                )
                # Возвращаем размерности в исходный порядок [batch, height, width, channels]
                img = img_resized.movedim(1, -1)
            scaled_images.append(img)
        
        # Применяем метод Lighter последовательно ко всем изображениям
        result = scaled_images[0]
        for img in scaled_images[1:]:
            result = torch.maximum(result, img)
        
        # Применяем коэффициент смешивания
        if blend_factor < 1.0:
            result = image1 * (1 - blend_factor) + result * blend_factor
        
        return (result,)

class ImageOffset:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "offset_x": ("INT", {
                "default": 0, 
                "min": -4096,
                "max": 4096,
                "step": 1
            }),
            "offset_y": ("INT", {
                "default": 0,
                "min": -4096,
                "max": 4096,
                "step": 1
            }),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "offset"
    CATEGORY = "custom_node_experiments"

    def offset(self, image, offset_x, offset_y):
        # Получаем размеры изображения [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Создаем новый тензор, заполненный нулями (черный цвет)
        result = torch.zeros_like(image)
        
        # Определяем границы копирования для источника и назначения
        src_start_y = max(0, -offset_y)
        src_end_y = min(height, height - offset_y)
        src_start_x = max(0, -offset_x)
        src_end_x = min(width, width - offset_x)
        
        dst_start_y = max(0, offset_y)
        dst_end_y = min(height, height + offset_y)
        dst_start_x = max(0, offset_x)
        dst_end_x = min(width, width + offset_x)
        
        # Копируем часть изображения с учетом смещения для всего батча
        result[:, dst_start_y:dst_end_y, dst_start_x:dst_end_x, :] = \
            image[:, src_start_y:src_end_y, src_start_x:src_end_x, :]
        
        return (result,)

class RGBtoRYGCBM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, image):
        # Получаем размеры входного изображения [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Проверяем, что у нас RGB изображение
        if channels != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        # Создаем выходной тензор для 6 каналов
        result = torch.zeros((batch, height, width, 6), device=image.device, dtype=image.dtype)
        
        # Получаем RGB каналы
        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]
        
        # Вычисляем смешанные цвета
        Y = torch.minimum(R, G)  # Желтый (минимум из R и G)
        C = torch.minimum(G, B)  # Голубой (минимум из G и B)
        M = torch.minimum(R, B)  # Пурпурный (минимум из R и B)
        
        # Усиливаем основные цвета в 2 раза и вычитаем половину смешанных
        R = torch.maximum(R*2 - Y/2 - M/2, torch.zeros_like(R))  # Усиливаем R и убираем половину Y и M
        G = torch.maximum(G*2 - Y/2 - C/2, torch.zeros_like(G))  # Усиливаем G и убираем половину Y и C
        B = torch.maximum(B*2 - C/2 - M/2, torch.zeros_like(B))  # Усиливаем B и убираем половину C и M
        
        # Заполняем каналы RYGCBM
        result[..., 0] = R  # Красный (усиленный)
        result[..., 1] = Y  # Желтый
        result[..., 2] = G  # Зеленый (усиленный)
        result[..., 3] = C  # Голубой
        result[..., 4] = B  # Синий (усиленный)
        result[..., 5] = M  # Пурпурный
        
        return (result,)

class RYGCBMtoRGB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "blend_mode": (["average", "maximum", "minimum"],),
            "normalize": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, image, blend_mode, normalize):
        # Получаем размеры входного изображения [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Проверяем, что у нас RYGCBM изображение
        if channels != 6:
            raise ValueError("Input image must be RYGCBM (6 channels)")
        
        # Создаем выходной тензор для RGB
        result = torch.zeros((batch, height, width, 3), device=image.device, dtype=image.dtype)
        
        # Получаем RYGCBM каналы
        R = image[..., 0]  # Красный
        Y = image[..., 1]  # Желтый
        G = image[..., 2]  # Зеленый
        C = image[..., 3]  # Голубой
        B = image[..., 4]  # Синий
        M = image[..., 5]  # Пурпурный
        
        # Вычисляем RGB на основе выбранного метода смешивания
        if blend_mode == "average":
            # R участвует в R, Y, M
            result[..., 0] = (R + Y/2 + M/2) / 2
            # G участвует в Y, G, C
            result[..., 1] = (Y/2 + G + C/2) / 2
            # B участвует в C, B, M
            result[..., 2] = (C/2 + B + M/2) / 2
        elif blend_mode == "maximum":
            # Для каждого канала берем максимум из участвующих цветов
            result[..., 0] = torch.maximum(R, torch.maximum(Y/2, M/2))  # R
            result[..., 1] = torch.maximum(G, torch.maximum(Y/2, C/2))  # G
            result[..., 2] = torch.maximum(B, torch.maximum(C/2, M/2))  # B
        else:  # minimum
            # Для каждого канала берем минимум из участвующих цветов
            result[..., 0] = torch.minimum(R, torch.minimum(Y/2, M/2))  # R
            result[..., 1] = torch.minimum(G, torch.minimum(Y/2, C/2))  # G
            result[..., 2] = torch.minimum(B, torch.minimum(C/2, M/2))  # B
        
        # Нормализация результата
        if normalize:
            # Находим максимальное значение для каждого пикселя по всем каналам
            max_vals = torch.maximum(result[..., 0], 
                                   torch.maximum(result[..., 1], 
                                               result[..., 2]))
            # Избегаем деления на ноль
            max_vals = torch.maximum(max_vals, torch.tensor(1e-8, device=max_vals.device))
            # Нормализуем каждый канал
            result[..., 0] = result[..., 0] / max_vals
            result[..., 1] = result[..., 1] / max_vals
            result[..., 2] = result[..., 2] / max_vals
        
        return (result,)

class ExtractImageChannel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "channel": ("INT", {
                "default": 0,
                "min": 0,
                "max": 64,  # Достаточно большое значение для поддержки разных форматов
                "step": 1
            }),
            "output_mode": (["single_channel", "rgb_repeated"],),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract"
    CATEGORY = "custom_node_experiments"

    def extract(self, image, channel, output_mode):
        # Получаем размеры входного изображения [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Проверяем, что запрошенный канал существует
        if channel >= channels:
            raise ValueError(f"Channel {channel} does not exist. Image has {channels} channels.")
        
        # Извлекаем нужный канал
        selected_channel = image[..., channel]
        
        if output_mode == "single_channel":
            # Возвращаем один канал, сохраняя размерность
            return (selected_channel.unsqueeze(-1),)
        else:  # rgb_repeated
            # Создаем RGB изображение, повторяя выбранный канал три раза
            result = torch.zeros((batch, height, width, 3), device=image.device, dtype=image.dtype)
            result[..., 0] = selected_channel  # R
            result[..., 1] = selected_channel  # G
            result[..., 2] = selected_channel  # B
            return (result,)

class MatchRYGCBMColors:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),        # Входное изображение для корректировки
            "reference": ("IMAGE",),     # Образцовое изображение
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_colors"
    CATEGORY = "custom_node_experiments"

    def match_colors(self, image, reference):
        # Проверяем, что оба изображения 6-канальные
        if image.shape[-1] != 6 or reference.shape[-1] != 6:
            raise ValueError("Both images must be 6-channel RYGCBM images")
        
        # Вычисляем среднее и стандартное отклонение для каждого канала
        # Используем keepdim=True для сохранения размерности для бродкастинга
        input_mean = torch.mean(image, dim=(1, 2), keepdim=True)
        input_std = torch.std(image, dim=(1, 2), keepdim=True)
        
        ref_mean = torch.mean(reference, dim=(1, 2), keepdim=True)
        ref_std = torch.std(reference, dim=(1, 2), keepdim=True)
        
        # Выводим статистики в консоль
        channels = ['R', 'Y', 'G', 'C', 'B', 'M']
        print("\nInput image statistics:")
        for i, ch in enumerate(channels):
            print(f"{ch}: mean = {input_mean[0,0,0,i]:.4f}, std = {input_std[0,0,0,i]:.4f}")
        
        print("\nReference image statistics:")
        for i, ch in enumerate(channels):
            print(f"{ch}: mean = {ref_mean[0,0,0,i]:.4f}, std = {ref_std[0,0,0,i]:.4f}")
        
        # Избегаем деления на ноль
        eps = 1e-8
        input_std = torch.maximum(input_std, torch.ones_like(input_std) * eps)
        ref_std = torch.maximum(ref_std, torch.ones_like(ref_std) * eps)
        

        # Применяем статистики образца
        matched = (image - input_mean)  * ref_std / input_std + ref_mean
        
        result = matched

        # Ограничиваем значения в диапазоне [0, 1]
        #result = torch.clamp(matched, 0, 1)
        
        # Выводим статистики результата
        result_mean = torch.mean(result, dim=(1, 2), keepdim=True)
        result_std = torch.std(result, dim=(1, 2), keepdim=True)
        
        print("\nResult image statistics:")
        for i, ch in enumerate(channels):
            print(f"{ch}: mean = {result_mean[0,0,0,i]:.4f}, std = {result_std[0,0,0,i]:.4f}")
        
        print("\n" + "-"*50)  # Разделитель для удобства чтения
        
        return (result,)

class TextCommaToWeighted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"default": ""}),
            "weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, text, weight):
        if text is None:
            return ("",)
        parts = [p.strip() for p in str(text).split(",")]
        parts = [p for p in parts if p]
        if not parts:
            return ("",)
        formatted = ", ".join([f"({p}:{weight})" for p in parts])
        return (formatted,)

class TextCommaToRandomWeighted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"default": ""}),
            "min_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
            "max_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, text, min_weight, max_weight, seed):
        if text is None:
            return ("",)
        random.seed(seed)
        parts = [p.strip() for p in str(text).split(",")]
        parts = [p for p in parts if p]
        if not parts:
            return ("",)
        formatted = ", ".join([f"({p}:{random.uniform(min_weight, max_weight):.2f})" for p in parts])
        return (formatted,)

def rgb_to_xyz(rgb):
    mask = rgb > 0.04045
    rgb[mask] = torch.pow((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92
    
    # RGB to XYZ matrix multiplication
    xyz_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device)
    
    return torch.tensordot(rgb, xyz_matrix.T, dims=([3], [0]))

def xyz_to_lab(xyz):
    # D65 illuminant
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device)
    
    # Scale by reference white
    xyz_scaled = xyz / xyz_ref_white
    
    # Nonlinear distortion and linear transformation
    mask = xyz_scaled > 0.008856
    xyz_scaled[mask] = torch.pow(xyz_scaled[mask], 1/3)
    xyz_scaled[~mask] = 7.787 * xyz_scaled[~mask] + 16/116
    
    x, y, z = xyz_scaled.unbind(dim=3)
    
    # Vector operations for L, a, b
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    
    return torch.stack([L, a, b], dim=3)

def lab_to_xyz(lab):
    L, a, b = lab.unbind(dim=3)
    
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    
    xyz = torch.stack([x, y, z], dim=3)
    
    # D65 illuminant
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=lab.device)
    
    mask = xyz > 0.206893
    xyz[mask] = torch.pow(xyz[mask], 3)
    xyz[~mask] = (xyz[~mask] - 16/116) / 7.787
    
    return xyz * xyz_ref_white

def xyz_to_rgb(xyz):
    # XYZ to RGB matrix multiplication
    rgb_matrix = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], device=xyz.device)
    
    rgb = torch.tensordot(xyz, rgb_matrix.T, dims=([3], [0]))
    
    # Inverse companding
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * torch.pow(rgb[mask], 1/2.4) - 0.055
    rgb[~mask] = 12.92 * rgb[~mask]
    
    return rgb

def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    return xyz_to_lab(xyz)

def lab_to_rgb(lab):
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb(xyz)

class RGBtoLAB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, image):
        return (rgb_to_lab(image),)

class LABtoRGB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, image):
        rgb = lab_to_rgb(image)
        return (torch.clamp(rgb, 0, 1),)

class ImageStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "reference": ("IMAGE",),
            "ratio": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.01}),
            "reproj_thresh": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            "show_matches": ("BOOLEAN", {"default": False}),
        }}
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stitched_image", "matches_visualization")
    FUNCTION = "stitch"
    CATEGORY = "custom_node_experiments"

    def detect_and_describe(self, image):
        """Обнаруживает ключевые точки и извлекает дескрипторы SIFT"""
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Создаем SIFT детектор
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(gray, None)
        
        # Конвертируем keypoints в numpy array
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reproj_thresh):
        """Сопоставляет ключевые точки между изображениями"""
        # Создаем matcher
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
        
        matches = []
        # Применяем тест Лоу
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        # Вычисляем гомографию если достаточно совпадений
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
            return (matches, H, status)
        
        return None

    def draw_matches(self, imageA, imageB, kpsA, kpsB, matches, status):
        """Создает визуализацию совпадений"""
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        
        # Создаем изображение для визуализации
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        
        # Рисуем линии совпадений
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis

    def stitch(self, image1, reference, ratio, reproj_thresh, show_matches):
        """Основная функция совмещения изображений"""

        image2 = reference
        
        # Конвертируем из torch tensor в numpy array
        # image1 и image2 имеют формат [batch, height, width, channels]
        # ComfyUI использует формат [batch, height, width, channels] с значениями [0, 1]
        img1_np = (image1[0].cpu().numpy() * 255).astype(np.uint8)  # Берем первый элемент батча
        img2_np = (image2[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Конвертируем из RGB в BGR для OpenCV
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
        
        # Обнаруживаем ключевые точки и дескрипторы
        (kps1, features1) = self.detect_and_describe(img1_np)
        (kps2, features2) = self.detect_and_describe(img2_np)
        
        print(f"Обнаружено {len(kps1)} ключевых точек в первом изображении")
        print(f"Обнаружено {len(kps2)} ключевых точек во втором изображении")
        
        # Сопоставляем ключевые точки
        M = self.match_keypoints(kps1, kps2, features1, features2, ratio, reproj_thresh)
        
        if M is None:
            print("Недостаточно совпадений для создания панорамы")
            # Возвращаем первое изображение как fallback
            return (image1, torch.zeros_like(image1))
        
        (matches, H, status) = M
        print(f"Найдено {len(matches)} совпадений")
        
        # Применяем перспективное преобразование
        result = cv2.warpPerspective(img1_np, H, (img1_np.shape[1], img1_np.shape[0]))
        
        # Конвертируем результат обратно из BGR в RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Создаем визуализацию совпадений если нужно
        if show_matches:
            matches_vis = self.draw_matches(img1_np, img2_np, kps1, kps2, matches, status)
            # Конвертируем визуализацию из BGR в RGB
            matches_vis = cv2.cvtColor(matches_vis, cv2.COLOR_BGR2RGB)
            matches_vis_tensor = torch.from_numpy(matches_vis.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            matches_vis_tensor = torch.zeros_like(image1)
        
        # Конвертируем результат обратно в torch tensor в формате ComfyUI
        # [batch, height, width, channels] с значениями [0, 1]
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor, matches_vis_tensor)

class ImageMirrorPad:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "n": ("INT", {"default": 16, "min": 0, "max": 4096, "step": 1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad"
    CATEGORY = "custom_node_experiments"

    def pad(self, image, n):
        # image: [batch, height, width, channels], float in [0,1]
        if n == 0:
            return (image,)
        batch, h, w, c = image.shape
        # Эффективный паддинг не может превышать (h-1) и (w-1) для отражения
        n_eff = int(min(n, max(h - 1, 0), max(w - 1, 0)))
        # Если невозможно отразить (очень маленькая картинка), просто вернем исходное
        if n_eff == 0:
            return (image,)

        # Создаем выходной тензор
        out = torch.zeros((batch, h + 2 * n_eff, w + 2 * n_eff, c), device=image.device, dtype=image.dtype)

        # Центр
        out[:, n_eff:n_eff + h, n_eff:n_eff + w, :] = image

        # Левые и правые вертикальные полосы (отражение по X)
        for i in range(n_eff):
            # слева: зеркально отражаем столбцы - берем столбец (i) и копируем его
            out[:, n_eff:n_eff + h, i, :] = image[:, :,n_eff - i, :]
            # справа: зеркально отражаем столбцы - берем столбец (w-1-i) и копируем его
            out[:, n_eff:n_eff + h, n_eff + w + i, :] = image[:, :, w - 1 - i, :]

        # Верхние и нижние горизонтальные полосы (отражение по Y)
        for j in range(n_eff):
            # сверху: зеркально отражаем строки - берем строку (j) и копируем её
            out[:, j, n_eff:n_eff + w, :] = image[:, n_eff - j, :, :]
            # снизу: зеркально отражаем строки - берем строку (h-1-j) и копируем её
            out[:, n_eff + h + j, n_eff:n_eff + w, :] = image[:, h - 1 - j, :, :]

        # Углы: отражение по обоим направлениям
        for j in range(n_eff):
            for i in range(n_eff):
                # верх-лево
                out[:, j, i, :] = image[:, n_eff - j,  n_eff - i, :]
                # верх-право
                out[:, j, n_eff + w + i, :] = image[:, n_eff -j, w - 1 - i, :]
                # низ-лево
                out[:, n_eff + h + j, i, :] = image[:, h - 1 - j, n_eff -i, :]
                # низ-право
                out[:, n_eff + h + j, n_eff + w + i, :] = image[:, h - 1 - j, w - 1 - i, :]

        return (out,)

class ImageCropBorders:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "n": ("INT", {"default": 16, "min": 0, "max": 4096, "step": 1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "custom_node_experiments"

    def crop(self, image, n):
        # image: [batch, height, width, channels]
        if n == 0:
            return (image,)
        batch, h, w, c = image.shape
        top = n
        left = n
        bottom = max(0, h - n)
        right = max(0, w - n)
        if bottom <= top or right <= left:
            # if crop exceeds image, return empty border-cropped to minimal valid
            return (image[:, 0:0, 0:0, :],)
        result = image[:, top:bottom, left:right, :]
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageScaleToTotalPixelsRound64": ImageScaleToTotalPixelsRound64,
    "ImageBlendLighter": ImageBlendLighter,
    "ImageOffset": ImageOffset,
    "RGBtoRYGCBM": RGBtoRYGCBM,
    "RYGCBMtoRGB": RYGCBMtoRGB,
    "ExtractImageChannel": ExtractImageChannel,
    "MatchRYGCBMColors": MatchRYGCBMColors,
    "TextCommaToWeighted": TextCommaToWeighted,
    "TextCommaToRandomWeighted": TextCommaToRandomWeighted,
    "RGBtoLAB": RGBtoLAB,
    "LABtoRGB": LABtoRGB,
    "ImageMirrorPad": ImageMirrorPad,
    "ImageCropBorders": ImageCropBorders,
    "ImageStitcher": ImageStitcher,
}