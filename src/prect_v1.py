import os
from ultralytics import YOLO

best_model_path = "./runs/detect/aortic_run_final/weights/best.pt"

if not os.path.exists(best_model_path):
    print("cannot find the mdoel path")
else:
    model = YOLO(best_model_path)
    
    
    submission_file = "submission_final_100epochs.txt"
    base_dir = "./datasets"
    with open(submission_file, 'w') as f:
        results = model.predict(
            source=f"{base_dir}/test/images",
            imgsz=640,
            device=0,
            conf=0.001,   
            iou=0.6,      
            verbose=False,
            stream=True,  
            augment=True  
        )
        
        count = 0
        for result in results:
            count += 1
            if count % 1000 == 0:
                print(f"finishing {count}")

            filename = os.path.basename(result.path).replace(".png", "")
            boxes = result.boxes
            
            if len(boxes) > 0:
                for k in range(len(boxes)):
                    cls = int(boxes.cls[k].item())
                    conf = boxes.conf[k].item()
                    x1, y1, x2, y2 = boxes.xyxy[k].tolist()
                    
                    # 寫入格式
                    line = f"{filename} {cls} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    f.write(line)

    
    print(f"data path: '{submission_file}'")
