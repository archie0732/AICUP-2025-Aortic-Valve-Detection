import os
from ultralytics import YOLO

model = YOLO('yolov9e.pt') 

# 2. 開始訓練
results = model.train(
    data="./aortic_valve_full.yaml", 
    
    epochs=120,       
    imgsz=640,        
    
    
    batch=8,          
    
    device=1,         
    name='aortic_run_YOLOv9e', 
    
    workers=1,        
    patience=0,       
    cache=True,       
    augment=True,     
    
    close_mosaic=15,  
    cos_lr=True,      
    optimizer='auto'  
)

print("YOLOv9e Complete")
