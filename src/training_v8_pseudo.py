import os
import shutil
from ultralytics import YOLO


model_path = "./runs/detect/aortic_run_11x_Human_Corrected/weights/best.pt"
model = YOLO(model_path)

base_dir = "./datasets"
test_img_dir = f"{base_dir}/test/images"
clean_train_dir = "./datasets/train" 
pseudo_dir = "./datasets_pseudo_final/train"

if os.path.exists(pseudo_dir): shutil.rmtree(pseudo_dir)
os.makedirs(f"{pseudo_dir}/images", exist_ok=True)
os.makedirs(f"{pseudo_dir}/labels", exist_ok=True)

for f in os.listdir(f"{clean_train_dir}/images"):
    shutil.copy(f"{clean_train_dir}/images/{f}", f"{pseudo_dir}/images/{f}")
for f in os.listdir(f"{clean_train_dir}/labels"):
    shutil.copy(f"{clean_train_dir}/labels/{f}", f"{pseudo_dir}/labels/{f}")

results = model.predict(source=test_img_dir, imgsz=640, device=0, conf=0.85, verbose=False, stream=True)

count = 0
for result in results:
    if len(result.boxes) > 0:
        filename = os.path.basename(result.path)
        txt_name = filename.replace('.png', '.txt')
        
        with open(f"{pseudo_dir}/labels/{txt_name}", 'w') as f:
            for box in result.boxes:
                cls = int(box.cls.item())
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x} {y} {w} {h}\n")
        
        shutil.copy(result.path, f"{pseudo_dir}/images/{filename}")
        count += 1

print(f"✅ Generate {count}'s picture")

yaml_content = f"""
path: {os.path.abspath("./datasets_pseudo_final")}
train: train/images
val: train/images
names:
  0: aortic_valve
"""
with open("aortic_pseudo_final.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)


model = YOLO('yolo11x.pt') 

results = model.train(
    data="aortic_pseudo_final.yaml",
    epochs=120,       
    batch=12,         
    imgsz=640,        
    device=0,         
    name='aortic_run_11x_Pseudo_Final',
    
    workers=2,
    cache='disk',     
    patience=0,
    augment=True,
    close_mosaic=15,
    cos_lr=True,
    optimizer='auto'
)

print("🏆 明天的冠軍模型訓練完成！")
