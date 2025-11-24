import os
from ultralytics import YOLO

print("import YOLOv8-Large model (on GPU 1 trainning)...")
model = YOLO('yolov8l.pt') 

results = model.train(
    data="./aortic_valve_full.yaml", 
    epochs=200,      
    batch=16,         
    imgsz=640,       
    
    
    device=1,         
    
    name='aortic_run_v3_200',
    workers=1,        
    patience=0,       
    cache=True,       
    augment=True,     
    
    close_mosaic=15,  
    cos_lr=True,      
    optimizer='auto'  
)

print("complete!")
