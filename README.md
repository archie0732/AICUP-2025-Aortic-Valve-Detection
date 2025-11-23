# AI CUP 2025 秋季賽：電腦斷層心臟肌肉影像分割競賽 II - 主動脈瓣物件偵測

技術報告書：模型開發與訓練策略

1. 硬體設備與運算環境 (Hardware Environment)

|項目 (Item) | 規格與配置 (Specification) | 說明 (Description)|
|-----------|-----|----|
運算平台|TWCC 容器運算服務 (CCS)|高效能運算環境|
|GPU|NVIDIA Tesla V100-SXM2-32GB | 具備 32GB VRAM | 足以支撐 Extra-Large 模型與較大 Batch Size 的訓練需求 |
|CPU | 8 Cores | 提供足夠的資料預處理能力|
|記憶體 (RAM) | 180 GB | 極大的記憶體空間 | 允許開啟 cache=True 將所有訓練資料載入 RAM 以加速訓練 |
|作業系統 | Linux (Ubuntu) | 標準深度學習環境|
|軟體環境 | "PyTorch 24.08, Ultralytics 8.3" | 使用最新的 YOLOv8 框架與相容的 PyTorch 版本|

本次競賽採用 台灣杉二號 (Taiwania 2) 之容器運算服務 (TWCC CCS) 進行模型訓練。相較於 Google Colab，TWCC 提供了更穩定且高效能的運算資源，使我們能夠訓練參數量更大的 YOLOv8x 模型並使用全資料集進行訓練。
<p align="center">
  <a href="https://ko-fi.com/arch1e0732"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on ko-fi!" /></a>
</p>
