import os
from ultralytics import YOLO
from ensemble_boxes import *


model_path = "./runs/detect/aortic_run_11x_Human_Corrected/weights/best.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌")

model = YOLO(model_path)

base_dir = "./datasets"
test_img_dir = f"{base_dir}/test/images"


preds_tta = {}
results_tta = model.predict(
    source=test_img_dir, imgsz=640, device=0, 
    conf=0.001, iou=0.65, augment=True,
    verbose=False, stream=True
)

for res in results_tta:
    fname = os.path.basename(res.path).replace(".png", "")
    if fname not in preds_tta: preds_tta[fname] = {'boxes':[], 'scores':[], 'labels':[]}
    if len(res.boxes) > 0:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds_tta[fname]['boxes'].append([x1/512, y1/512, x2/512, y2/512])
            preds_tta[fname]['scores'].append(box.conf.item())
            preds_tta[fname]['labels'].append(int(box.cls.item()))

preds_std = {}
results_std = model.predict(
    source=test_img_dir, imgsz=640, device=0, 
    conf=0.001, iou=0.65, augment=False,
    verbose=False, stream=True
)

for res in results_std:
    fname = os.path.basename(res.path).replace(".png", "")
    if fname not in preds_std: preds_std[fname] = {'boxes':[], 'scores':[], 'labels':[]}
    if len(res.boxes) > 0:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds_std[fname]['boxes'].append([x1/512, y1/512, x2/512, y2/512])
            preds_std[fname]['scores'].append(box.conf.item())
            preds_std[fname]['labels'].append(int(box.cls.item()))

print("⚗️ 正在進行自我融合 (TTA x2 + Standard x1)...")
output_filename = "submission_Last_Dance_097.txt"

weights = [2, 1] 
iou_thr = 0.65
skip_box_thr = 0.001

with open(output_filename, 'w') as f_out:
    all_files = set(list(preds_tta.keys()) + list(preds_std.keys()))
    
    for fname in all_files:
        run_data = [
            preds_tta.get(fname, {'boxes':[], 'scores':[], 'labels':[]}),
            preds_std.get(fname, {'boxes':[], 'scores':[], 'labels':[]})
        ]
        
        boxes_list = [run['boxes'] for run in run_data]
        scores_list = [run['scores'] for run in run_data]
        labels_list = [run['labels'] for run in run_data]
        
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = int(box[0]*512), int(box[1]*512), int(box[2]*512), int(box[3]*512)
            f_out.write(f"{fname} {int(label)} {score:.4f} {x1} {y1} {x2} {y2}\n")

print(f"🏆 path：{output_filename}")
