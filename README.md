# AI CUP 2025 秋季賽：電腦斷層心臟肌肉影像分割競賽 II - 主動脈瓣物件偵測

技術報告書：模型開發與訓練策略

## 目錄

- [1. 硬體設備與運算環境 (Hardware Environment)](##1.硬體設備與運算環境 (Hardware Environment))
- [2. 資料前處理 (Data Preprocessing)](##2.資料前處理 (Data Preprocessing))
- [3. 模型選擇與架構 (Model Architecture)](##3.模型選擇與架構 (Model Architecture))
- [4. 訓練策略與參數設定 (Training Strategy)](##4.訓練策略與參數設定 (Training Strategy))
- [5. 推論與後處理優化 (Inference & Post-processing)](##5.推論與後處理優化 (Inference & Post-processing))
- [6. 結論](##6.結論)
- [7. 斗內](https://ko-fi.com/arch1e0732)

## 1. 硬體設備與運算環境 (Hardware Environment)

|項目 (Item) | 規格與配置 (Specification) | 說明 (Description)|
|-----------|-----|----|
運算平台|TWCC 容器運算服務 (CCS)|高效能運算環境|
|GPU|NVIDIA Tesla V100-SXM2-32GB | 具備 32GB VRAM | 足以支撐 Extra-Large 模型與較大 Batch Size 的訓練需求 |
|CPU | 8 Cores | 提供足夠的資料預處理能力|
|記憶體 (RAM) | 180 GB | 極大的記憶體空間 | 允許開啟 cache=True 將所有訓練資料載入 RAM 以加速訓練 |
|作業系統 | Linux (Ubuntu) | 標準深度學習環境|
|軟體環境 | "PyTorch 24.08, Ultralytics 8.3" | 使用最新的 YOLOv8 框架與相容的 PyTorch 版本|

本次競賽採用 台灣杉二號 (Taiwania 2) 之容器運算服務 (TWCC CCS) 進行模型訓練。相較於 Google Colab，TWCC 提供了更穩定且高效能的運算資源，使我們能夠訓練參數量更大的 YOLOv8x 模型並使用全資料集進行訓練。


## 2. 資料前處理 (Data Preprocessing)

針對主動脈瓣檢測任務，我們對官方提供的資料集進行了重組與優化，以最大化模型的學習效能。

 >   全資料訓練 (Full Data Training)： 官方 Baseline 預設將 50 位病患資料切分為 30 位訓練、20 位驗證 。為了提升模型的泛化能力，我們將驗證集 (Validation Set) 全部合併回訓練集 (Training Set)，使用完整的 50 位病患資料 進行訓練。這增加了 40% 的訓練數據，顯著提升了模型對不同案例的適應性。

- 影像尺寸調整： 原始 PNG 影像尺寸為 512x512 pixel 。我們在訓練時將輸入尺寸 (imgsz) 提升至 640x640，這有助於模型捕捉主動脈瓣邊緣更細微的特徵，雖然增加了運算負擔，但在 V100 GPU 上是可行的。

- 記憶體快取 (RAM Caching)： 利用 TWCC 180GB 的記憶體優勢，我們在訓練參數中開啟 cache=True，將所有影像數據預先載入記憶體，大幅減少磁碟 I/O 時間，加速訓練迭代。


## 3. 模型選擇與架構 (Model Architecture)

本次競賽選擇 YOLOv8 作為核心檢測框架，並經歷了由小至大的模型迭代過程。

-    模型版本：YOLOv8x (Extra Large)

>-    選擇理由：
>> 1. 初期使用 yolov8n (Nano) 進行流程驗證 (Baseline score ~0.8)。
>> 2. 中期使用 yolov8l (Large) 提升特徵提取能力。
>> 3. 最終採用 yolov8x，這是 YOLOv8 系列中參數量最大、性能最強的版本。雖然訓練速度較慢，但其深層網路結構能更有效地識別主動脈瓣在 CT 影像中模糊不清的邊界。


## 4. 訓練策略與參數設定 (Training Strategy)

為了在有限的競賽時間內達到最佳收斂效果，我們採用了以下進階訓練策略：

|參數 (Hyperparameter) | 設定值 (Value) | 策略說明 (Strategy Rationale) |
|--|--|--|
|Epochs|100 ~ 120,遠高於 Baseline 的 20 epochs|確保模型充分收斂|
|Batch Size,12 ~ 16|針對 V100 32GB VRAM 最佳化|在不發生 OOM (Out of Memory) 的前提下最大化|
|Optimizer | SGD / AdamW | 使用動量優化器加速收斂 |
|Learning Rate,cos_lr=True  | 採用餘弦退火 (Cosine Annealing) 策略 | 在訓練後期穩定收斂至全域最佳解|
|Close Mosaic | 15 | 在最後 15 個 Epochs 關閉馬賽克增強 (Mosaic Augmentation)，讓模型專注於真實影像的特徵學習|
|Workers|1|針對容器環境的共享記憶體限制進行調整，避免 Bus Error


## 5. 推論與後處理優化 (Inference & Post-processing)

在測試階段 (Test Phase)，針對 16,620 張測試圖片 進行預測時，我們採用了以下技巧以提升分數並避免記憶體溢出：

1. 測試時增強 (TTA, Test Time Augmentation)： 開啟 augment=True，模型會在預測時自動對影像進行多尺度縮放與翻轉，並融合預測結果。這雖然會增加推論時間，但能顯著提升預測的準確度與穩定性。

2. 串流預測 (Streaming)： 設定 stream=True，以生成器 (Generator) 模式逐批處理測試影像，避免一次性將所有預測結果載入 RAM 導致記憶體不足 (OOM) 的問題。

3. 非極大值抑制 (NMS) 微調：

   - Confidence Threshold (conf): 設定為極低的 0.001，確保不漏檢任何可能的目標 (High Recall)，後續交由 mAP 指標進行評估。

   - IoU Threshold: 調整至 0.65 左右，以優化重疊框的合併效果。

## 6. 結論

透過使用 TWCC 的 V100 GPU 算力，我們成功將模型從輕量級的 YOLOv8n 升級至 YOLOv8x，並配合全資料訓練與 TTA 等策略，將成績從 Baseline 的 0.8 大幅提升至 0.96+。這證明了硬體資源的升級與精細的訓練策略調整，對於電腦視覺競賽成績有著決定性的影響。


<p align="center">
  <a href="https://ko-fi.com/arch1e0732"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on ko-fi!" /></a>
</p>
