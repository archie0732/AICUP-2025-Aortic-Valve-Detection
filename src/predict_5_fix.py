import os
import glob
from ultralytics import YOLO


search_pattern = "./runs/detect/*11x_HD*/weights/best.pt"
possible_models = glob.glob(search_pattern)

if possible_models:
    model_path = max(possible_models, key=os.path.getmtime)
    print(f"🎉 best.pt：{model_path}")
else:
    print("⚠️ cannot find best.pt...")
    all_runs = glob.glob('./runs/detect/*/weights/best.pt')
    model_path = max(all_runs, key=os.path.getmtime)
    print(f"✅ auto import：{model_path}")

model = YOLO(model_path)

submission_file = "submission_11x_HD_896.txt"
base_dir = "./datasets"


with open(submission_file, 'w') as f:
    results = model.predict(
        source=f"{base_dir}/test/images",
        
        imgsz=896,   
        augment=True,
        
        device=0,     
        conf=0.001,  
        iou=0.65,     
        verbose=False,
        stream=True   
    )
    
    count = 0
    for result in results:
        count += 1
        if count % 1000 == 0: print(f"finish {count}...")
        
        filename = os.path.basename(result.path).replace(".png", "")
        boxes = result.boxes
        if len(boxes) > 0:
            for k in range(len(boxes)):
                cls = int(boxes.cls[k].item())
                conf = boxes.conf[k].item()
                x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                
                
                line = f"{filename} {cls} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                f.write(line)

print(f"✅ path：{submission_file}")
