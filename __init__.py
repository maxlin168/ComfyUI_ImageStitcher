# __init__.py

# 🚨 關鍵變更: 從 ImageStitcher.py 匯入節點映射
# 確保 ComfyUI 能夠找到並載入您的節點類別
try:
    from .ImageStitcher import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # 如果找不到 ImageStitcher.py，則設置為空字典以避免錯誤
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print("Warning: Could not import NODE_CLASS_MAPPINGS from ImageStitcher.py.")


# 設置您的自定義節點套件的詳細資訊
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
# 如果 ImageStitcher.py 內沒有定義 NODE_DISPLAY_NAME_MAPPINGS，則使用空字典
NODE_DISPLAY_NAME_MAPPINGS = NODE_DISPLAY_NAME_MAPPINGS if 'NODE_DISPLAY_NAME_MAPPINGS' in locals() else {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 套件資訊 (適用於 ComfyUI Manager)
WEB_DIRECTORY = "./web" # 如果有前端 JavaScript 檔案，請指定其路徑

MANIFEST = {
    "name": "ComfyUI_ImageStitcher", 
    "version": "1.0.0", 
    "author": "maxlin168",
    "description": "ComfyUI experimental nodes for advanced image processing and stitching.",
    "tags": ["image", "utility", "stitching", "color", "lab"],
    "min_version": 1100,
    "github": "https://github.com/maxlin168/ComfyUI_ImageStitcher"
}
