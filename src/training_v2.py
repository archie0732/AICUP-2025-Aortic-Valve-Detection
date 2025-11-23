from ultralytics import YOLO



model = YOLO('yolov8x.pt') 


results = model.train(
    data="./aortic_valve_full.yaml", 
    epochs=120,      
    batch=12,        
    imgsz=640,        
    device=0,         
    name='aortic_run_X_final',
    workers=2,        
    patience=0,       
    cache=True,       
    augment=True,     
    
    
    close_mosaic=15,  
    cos_lr=True,      
    optimizer='AdamW'
)

print("答辯模型訓練完成！")
