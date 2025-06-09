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

NODE_CLASS_MAPPINGS = {
    "ImageScaleToTotalPixelsRound64": ImageScaleToTotalPixelsRound64,
    "ImageBlendLighter": ImageBlendLighter,
    "ImageOffset": ImageOffset,
}