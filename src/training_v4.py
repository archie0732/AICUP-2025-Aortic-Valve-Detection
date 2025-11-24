import os
from ultralytics import YOLO

model = YOLO('yolov8l.pt') 

results = model.train(
    data="./aortic_valve_full.yaml",
    epochs=150,       
    
    imgsz=800,       
    batch=8,          
    
    device=0,         
    
    name='aortic_run_L_HD_800', 
    
    workers=1,        
    patience=0,       
    cache=True,       
    augment=True,     
    
    close_mosaic=20,  
    cos_lr=True,      
    optimizer='auto'  
)

