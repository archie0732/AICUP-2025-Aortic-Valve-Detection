# AI CUP 2025 秋季賽：電腦斷層心臟肌肉影像分割競賽 II - 主動脈瓣物件偵測

技術報告書：模型開發與訓練策略

## 🏆 最終成績

| Public Score | Private Score (最終成績) | 名次 | 總上傳次數 |
| :---: | :---: | :---: | :---: |
| 0.961211 | **0.970346** | **20** | 23 |

> **Highlight:** 本次競賽中，雖然 Pseudo-Labeling 策略在 Public Leaderboard 分數略有下降 (0.967 -> 0.961)，但在 Private Leaderboard 中展現了極強的泛化能力，最終以 **0.9703** 取得隊伍最佳成績。

![](./meme.png)

## 目錄

- [1. 硬體設備與運算環境 (Hardware Environment)](#1-硬體設備與運算環境-hardware-environment)
- [2. 資料前處理 (Data Preprocessing)](#2-資料前處理-data-preprocessing)
- [3. 模型選擇與架構 (Model Architecture)](#3-模型選擇與架構-model-architecture)
- [4. 訓練策略與參數設定 (Training Strategy)](#4-訓練策略與參數設定-training-strategy)
- [5. 推論與後處理優化 (Inference & Post-processing)](#5-推論與後處理優化-inference--post-processing)
- [6. 結論與心得 (Conclusion)](#6-結論與心得-conclusion)
- [7. 贊助 (Donate)](#7-贊助-donate)

---

## 1. 硬體設備與運算環境 (Hardware Environment)

本次競賽採用 **台灣杉二號 (Taiwania 2)** 之容器運算服務 (TWCC CCS) 進行模型訓練。相較於 Google Colab，TWCC 提供了更穩定且高效能的運算資源，使我們能夠執行大規模的偽標籤訓練任務。

| 項目 (Item) | 規格與配置 (Specification) | 說明 (Description) |
| :--- | :--- | :--- |
| **運算平台** | TWCC 容器運算服務 (CCS) | 高效能運算環境 |
| **GPU** | **NVIDIA Tesla V100-SXM2-32GB** | 具備 32GB VRAM，足以支撐 YOLO11x 與 Batch Size 12 的訓練需求 |
| **CPU** | 8 Cores | 提供足夠的資料預處理與解壓縮能力 |
| **記憶體 (RAM)** | **180 GB** | 極大的記憶體空間，允許在訓練初期開啟 `cache=True` 加速，後期改用 `cache='disk'` 處理偽標籤大數據 |
| **作業系統** | Linux (Ubuntu) | 標準深度學習環境 |
| **軟體環境** | PyTorch 24.08, Ultralytics 8.3 | 使用最新的 YOLO 框架與相容的 PyTorch 版本 |

---

## 2. 資料前處理 (Data Preprocessing)

針對主動脈瓣檢測任務，我們採取了「數據清洗」與「偽標籤擴增」雙重策略，這是突破 0.97 分數的關鍵。

### A. 全資料訓練 (Full Data Training)
官方 Baseline 預設將 50 位病患資料切分為 30 位訓練、20 位驗證。為了提升模型的泛化能力，我們將 **驗證集 (Validation Set) 全部合併回訓練集**，使用完整的 **50 位病患資料** 進行訓練，增加模型對不同案例的適應性。

### B. 數據清洗 (Data Cleaning)
利用初步訓練的高精度模型對官方訓練集進行「反向檢查」，篩選出模型預測與標註差異過大的樣本，並使用 **labelImg** 進行人工校正（修正漏標與邊界框誤差）。修正後的乾淨數據使模型基礎分數提升至 0.9675。

### C. 偽標籤技術 (Pseudo-Labeling) - **奪冠關鍵**
面對僅有 50 筆訓練資料但有 16,620 筆測試資料的極端情況，我們採用了半監督學習策略：
1. 使用最強的「人工修正版模型」對測試集進行推論。
2. 篩選信心分數 (Confidence) > **0.85** 的高可信度預測框。
3. 將這些預測結果作為「偽標籤」，與原始訓練集混合，進行第二階段的自我訓練 (Self-Training)。
此舉讓訓練資料量暴增數百倍，大幅提升了模型對測試集特徵的覆蓋率。

---

## 3. 模型選擇與架構 (Model Architecture)

經歷了多次迭代（YOLOv8n -> YOLOv8l -> YOLOv9e），我們最終選定 **YOLO11x** 作為決戰模型。

- **最終模型：** **YOLO11x (Extra Large)**
- **選擇理由：**
    1. **SOTA 性能**：YOLO11 是 Ultralytics 最新發布的架構，其 C3k2 與 C2PSA 模組在特徵提取上優於 v8。
    2. **大模型優勢**：在 V100 32GB 的支援下，使用 Extra Large 版本能捕捉主動脈瓣模糊邊緣的細微特徵。
    3. **適應性**：在偽標籤的大數據訓練下，大模型較不易過擬合，能有效吸收海量數據的特徵。

---

## 4. 訓練策略與參數設定 (Training Strategy)

為了在有限的競賽時間內達到最佳收斂效果，我們採用了以下進階訓練策略：

| 參數 (Hyperparameter) | 設定值 (Value) | 策略說明 (Strategy Rationale) |
| :--- | :--- | :--- |
| **Epochs** | **120** | 針對偽標籤的大量數據，120 Epochs 能確保模型充分收斂且避免過度擬合 |
| **Batch Size** | **12** | 針對 YOLO11x 在 V100 上的記憶體極限進行最佳化 |
| **Optimizer** | **Auto (SGD)** | 配合 `cos_lr=True` (餘弦退火) 使用，確保訓練後期穩定收斂 |
| **Close Mosaic** | **15** | 在最後 15 個 Epochs 關閉馬賽克增強，讓模型專注於真實影像的特徵學習 |
| **Cache** | **Disk** | 由於加入偽標籤後資料量暴增，改用硬碟快取避免 RAM OOM (Out of Memory) |
| **Workers** | **2** | 適度增加 Workers 以加速大數據的讀取效率 |

---

## 5. 推論與後處理優化 (Inference & Post-processing)

在最終提交階段，我們放棄了單純的模型融合 (WBF)，轉而採用極致的單體模型優化策略：

1. **測試時增強 (TTA, Test Time Augmentation)：**
   開啟 `augment=True`，模型在預測時會自動進行多尺度縮放與翻轉並融合結果，顯著提升了邊緣檢測的穩定性。

2. **串流預測 (Streaming)：**
   設定 `stream=True`，以生成器模式處理 16,620 張測試影像，避免記憶體溢出。

3. **非極大值抑制 (NMS) 微調：**
   - **Confidence Threshold:** 設定為極低的 **0.001**，確保高 Recall（不漏抓）。
   - **IoU Threshold:** 調整至 **0.65**，優化重疊框的合併效果。

---

## 6. 結論與心得 (Conclusion)

本次競賽從 Baseline 的 0.8 一路突破至 0.97+，我們深刻體會到 **「數據品質 > 模型架構」** 的道理。

* **關鍵轉折點 1：數據清洗**
    修正官方標註錯誤後，分數由 0.965 提升至 0.9675，證明了乾淨數據的重要性。
* **關鍵轉折點 2：偽標籤 (Pseudo-Labeling)**
    雖然引入偽標籤後，Public Score 因部分雜訊而微幅下降 (0.961)，但在 Private Score 上卻大幅提升至 **0.9703**。這證實了在小樣本 (Few-shot) 的醫療影像競賽中，利用海量無標註測試集進行半監督學習，是提升模型泛化能力最有效的手段。

最終，我們依靠 **YOLO11x + 人工清洗數據 + 偽標籤自我訓練** 的組合拳，成功達成了競賽目標。

然後我最想說的是

![](https://p3-pc-sign.douyinpic.com/tos-cn-i-0813c000-ce/oIwAaiGxoEANbD51ABENALOGiAeEeIf93AiJYk~tplv-dy-aweme-images:q75.webp?biz_tag=aweme_images&from=327834062&lk3s=138a59ce&s=PackSourceEnum_SEARCH&sc=image&se=false&x-expires=1767420000&x-signature=M9dxF5VxSlPQ69%2Blvn3yw%2Bf2eRk%3D)

---

## 7. 贊助作者買 iPad (Donate)

我很需要一台平板謝謝

<p align="center">
  <a href="https://ko-fi.com/arch1e0732">
    <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on ko-fi!" />
  </a>
</p>
