import os
from ultralytics import YOLO

# 1. 載入模型 (一樣用最強的 L 模型)
print("載入 YOLOv8-Large 模型 (準備於 GPU 0 進行高解析度訓練)...")
model = YOLO('yolov8l.pt') 

# 2. 開始高解析度訓練
results = model.train(
    data="./aortic_valve_full.yaml", # 全資料設定
    epochs=150,       # 800px 訓練比較慢，跑 150 次應該足夠收斂
    
    # --- 關鍵修改：高解析度 ---
    imgsz=800,        # 【關鍵】放大圖片！讓模型看更細
    batch=8,          # 【關鍵】因為圖變大了，Batch 要調小一點避免 OOM
    
    device=0,         # 【關鍵】使用閒置的 GPU 0
    
    name='aortic_run_L_HD_800', # 取名為 HD (High Definition)
    
    workers=1,        # 維持 1 避免 Bus Error
    patience=0,       # 強制跑完
    cache=True,       # 180GB RAM 夠大，快取 800px 的圖也沒問題
    augment=True,     # 開啟增強
    
    # --- 優化參數 ---
    close_mosaic=20,  
    cos_lr=True,      
    optimizer='auto'  # 使用預設
)

print("高解析度 (HD) 模型訓練完成！")
