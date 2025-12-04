import os
from ultralytics import YOLO

print("👎 nm RAM 炸了")
model = YOLO('yolo11x.pt') 

results = model.train(
    data="./aortic_valve_full.yaml", 
    
    imgsz=896,        
    batch=4,          
    
    # NM 要用 Disk
    cache='disk', 
    
    workers=1,    
    device=0,         
    epochs=100,       
    name='aortic_run_11x_HD_896_disk', 
    
    patience=0,       
    augment=True,     
    close_mosaic=20,  
    cos_lr=True,      
    optimizer='auto'  
)

print("YOLO11x OK Please 🛐🛐🛐🛐🛐🛐🛐🛐🛐🛐🛐🛐")
