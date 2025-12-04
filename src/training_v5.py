# high res
import os
from ultralytics import YOLO

model = YOLO('yolov8l.pt') 

results = model.train(
    data="./aortic_valve_full.yaml", 
    
    imgsz=896,        
    batch=8,          
    
    epochs=120,      
    device=0,         
    name='aortic_run_L_HighRes_896', 
    
    workers=1,
    patience=0,
    cache=True,      
    augment=True,     
    close_mosaic=15,  
    cos_lr=True,      
    optimizer='auto'  
)

print("high res model finish!")
