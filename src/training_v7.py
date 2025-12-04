import zipfile
import os
import shutil
from ultralytics import YOLO

zip_filename = "tbrain-42人工標注資料用.zip"  
extract_to = "./temp_human_data"
target_train_label_dir = "./datasets/train/labels" 

print(f"📦 正在解壓縮 {zip_filename} ...")
if not os.path.exists(zip_filename):
    print(f"❌ Cannot Find  {zip_filename}")
else:
    if os.path.exists(extract_to): shutil.rmtree(extract_to)
    
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        source_label_dir = None
        for root, dirs, files in os.walk(extract_to):
            if "train" in os.path.basename(root).lower() and len([f for f in files if f.endswith(".txt")]) > 0:
                source_label_dir = root
                break
        
        if not source_label_dir:
            print("❌ Cannot Find dir path")
        else:
            print(f"✅ modify dir path：{source_label_dir}")
            
            os.makedirs(target_train_label_dir, exist_ok=True)
            
            files = [f for f in os.listdir(source_label_dir) if f.endswith(".txt")]
            count = 0
            for f in files:
                src = os.path.join(source_label_dir, f)
                dst = os.path.join(target_train_label_dir, f)
                shutil.copy(src, dst)
                count += 1
            

            
            model = YOLO('yolo11x.pt') 
            
            results = model.train(
                data="./aortic_valve_full.yaml",
                epochs=100,       
                batch=12,         
                imgsz=640,        
                device=0,         
                name='aortic_run_11x_Human_Corrected', 
                
                workers=2,
                patience=0,
                cache=True,       
                augment=True,
                close_mosaic=15,  
                cos_lr=True,
                optimizer='auto'
            )
            
            print("🏆 人工修正版模型訓練完成！")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
