import torch
import comfy.utils
import math
import torch.nn.functional as F

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
            "threshold": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            }),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "custom_node_experiments"

    def convert(self, image, threshold):
        # Получаем размеры входного изображения [batch, height, width, channels]
        batch, height, width, channels = image.shape
        
        # Проверяем, что у нас RGB изображение
        if channels != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        # Создаем выходной тензор для 6 каналов
        result = torch.zeros((batch, height, width, 6), device=image.device, dtype=image.dtype)
        
        # Получаем RGB каналы и применяем порог к ним
        R = torch.where(image[..., 0] > threshold, image[..., 0], torch.zeros_like(image[..., 0]))
        G = torch.where(image[..., 1] > threshold, image[..., 1], torch.zeros_like(image[..., 1]))
        B = torch.where(image[..., 2] > threshold, image[..., 2], torch.zeros_like(image[..., 2]))
        
        # Заполняем каналы RYGCBM
        # R - Красный (оставляем как есть)
        result[..., 0] = R
        
        # Y - Желтый (среднее красного и зеленого)
        result[..., 1] = (R + G) / 2
        
        # G - Зеленый (оставляем как есть)
        result[..., 2] = G
        
        # C - Голубой (среднее синего и зеленого)
        result[..., 3] = (G + B) / 2
        
        # B - Синий (оставляем как есть)
        result[..., 4] = B
        
        # M - Пурпурный (среднее красного и синего)
        result[..., 5] = (R + B) / 2
        
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

NODE_CLASS_MAPPINGS = {
    "ImageScaleToTotalPixelsRound64": ImageScaleToTotalPixelsRound64,
    "ImageBlendLighter": ImageBlendLighter,
    "ImageOffset": ImageOffset,
    "RGBtoRYGCBM": RGBtoRYGCBM,
    "RYGCBMtoRGB": RYGCBMtoRGB,
    "ExtractImageChannel": ExtractImageChannel,
}