import os
import shutil
from ultralytics import YOLO


val_img_dir = "./datasets/val/images"
val_lbl_dir = "./datasets/val/labels"
train_img_dir = "./datasets/train/images"
train_lbl_dir = "./datasets/train/labels"


if os.path.exists(val_img_dir):
    for f in os.listdir(val_img_dir):
        src = os.path.join(val_img_dir, f)
        dst = os.path.join(train_img_dir, f)
        if not os.path.exists(dst): shutil.move(src, dst)
            
    for f in os.listdir(val_lbl_dir):
        src = os.path.join(val_lbl_dir, f)
        dst = os.path.join(train_lbl_dir, f)
        if not os.path.exists(dst): shutil.move(src, dst)
   

print('debug: 001')
base_dir = "./datasets"
yaml_content = f"""
path: {os.path.abspath(base_dir)}
train: train/images
val: train/images
test: test/images

names:
  0: aortic_valve
"""


with open("aortic_valve_full.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)


print("載入 YOLOv8-Large 模型...")
model = YOLO('yolov8l.pt')  



results = model.train(
    data="./aortic_valve_full.yaml",
    epochs=100,       
    batch=16,         
    imgsz=640,        
    device=0,         
    name='aortic_run_final',
    workers=1,        
    patience=0,       
    cache=True,       
    augment=True      
)

print("訓練完成！")
