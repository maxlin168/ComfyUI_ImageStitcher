AI 圖像處理實驗室 (AI Image Processing Lab Nodes)
==================================================

這是一個為 ComfyUI 設計的自定義節點集合，包含各種圖像處理、顏色轉換和實用工具，特別是為高階圖像合成和對齊工作流程而優化。

✨ 主要功能亮點
* ImageStitcher：基於 SIFT/RANSAC 的圖像自動對齊/拼接節點，支持動態尺寸控制和視覺化。
* 顏色空間處理：提供 RGB/LAB 和獨特的 RGB/RYGCBM 顏色轉換，用於進階顏色匹配。
* 實用工具：包含圖像混合、偏移、尺寸調整到特定像素數（Round 64/Qwen 兼容）等節點。

📥 安裝指南 (Installation)

步驟 1: 安裝依賴
此節點集需要 opencv-python 庫來執行 SIFT 特徵檢測和圖像透視變換。請確保您的 Python 環境中已安裝此庫：

pip install opencv-python

步驟 2: 安裝節點
1. 導航到您的 ComfyUI 安裝目錄下的 custom_nodes 資料夾。
2. 將此程式碼檔案（例如命名為 ai_image_lab_nodes.py）放入 custom_nodes 資料夾中。
3. 重新啟動 ComfyUI。

您應該能在 ComfyUI 的節點搜尋功能中，找到一個名為 custom_node_experiments 的分類，其中包含所有自定義節點。

---

🛠 節點列表與詳解

1. 核心圖像對齊：ImageStitcher

此節點使用 SIFT 算法偵測兩張圖像之間的共同特徵點，計算透視變換矩陣（Homography），將 image1 準確對齊到 reference 圖像的視角。

- 參數名稱: image1
- 類型: IMAGE
- 描述: 待變形/對齊的圖像。

- 參數名稱: reference
- 類型: IMAGE
- 描述: 作為參考座標系的目標圖像。

- 參數名稱: feature_detection_size_mode
- 類型: STRING
- 描述: 特徵檢測時的統一縮放尺寸。image1 或 reference。這能確保在固定分辨率下進行穩定的匹配。

- 參數名稱: output_size_mode
- 類型: STRING
- 描述: 最終輸出的 stitched_image 尺寸。image1 或 reference。

- 參數名稱: upscale_method
- 類型: STRING
- 描述: 圖像縮放/變形時使用的插值方法。

- 參數名稱: ratio
- 類型: FLOAT
- 描述: 匹配距離比率，用於過濾較差的關鍵點匹配 (e.g., Lowe's ratio test)。

- 參數名稱: reproj_thresh
- 類型: FLOAT
- 描述: RANSAC 算法的重投影誤差閾值，用於濾除外點 (outliers)。

- 參數名稱: show_matches
- 類型: BOOLEAN
- 描述: 是否輸出匹配點的可視化圖像。

- 輸出 1 (stitched_image)
- 類型: IMAGE
- 描述: 已經變形並對齊到 reference 視角的 image1。

- 輸出 2 (matches_visualization)
- 類型: IMAGE
- 描述: 特徵匹配的視覺化圖。其寬度為 stitched_image 的兩倍，高度一致。

2. 顏色處理與匹配
* RGBtoLAB / LABtoRGB: 實現標準的 RGB 到 L*a*b* 顏色空間轉換和反轉。L*a*b* 空間在感知上更均勻，常用於顏色比較和匹配。
* RGBtoRYGCBM / RYGCBMtoRGB: 一種非標準的 6 通道顏色分解，有助於進行更細緻的顏色通道控制或實驗。
* MatchRYGCBMColors: 在 RYGCBM 空間中，基於均值 (mean) 和標準差 (std) 進行顏色統計匹配，將 image 的顏色分佈調整到接近 reference。

3. 圖像實用工具
* ImageScaleToTotalPixelsRound64: 將圖像尺寸縮放到一個接近指定百萬像素 (Megapixels) 總數的尺寸，並確保寬度和高度都是 64 的倍數。
* ImageScaleToQwen: 類似於上一個節點，但確保尺寸是 16 的倍數，適用於一些對尺寸有特定要求的模型（例如 Qwen-VL）。
* ImageBlendLighter: 將多張圖像進行像素級的最大值（Lighter Blend）混合。
* ImageOffset: 將圖像內容向指定 X/Y 軸方向平移，邊緣用黑色填充。
* ImageMirrorPad / ImageCropBorders: 用於創建基於邊界鏡像的填充和相應的裁剪功能。
* ExtractImageChannel: 從多通道圖像中提取單個通道。

4. 文本處理工具
* TextCommaToWeighted: 將逗號分隔的提示詞列表，轉換為帶有統一權重（例如 (prompt:0.4)）的格式。
* TextCommaToRandomWeighted: 將提示詞轉換為帶有隨機權重（在指定範圍內）的格式，用於提示詞實驗。

---

👏 致謝 (Acknowledgement)

ImageStitcher 節點的部分邏輯和結構，以及部分顏色處理節點，受到了以下 GitHub 儲存庫的啟發和參考：

* https://github.com/addddd2/AI_Generated_nodes

感謝開源社群對這些功能實現的貢獻。

---

📝 備註
* 由於 SIFT 算法是計算密集型的，在使用 ImageStitcher 時，如果輸入圖過大，建議先在 feature_detection_size_mode 中選擇一個適中的參考尺寸，以提高性能和匹配成功率。
* 此節點集仍在實驗階段 (custom_node_experiments 分類)，功能可能會根據需求進行調整和優化。
