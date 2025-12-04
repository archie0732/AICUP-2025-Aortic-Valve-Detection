from ensemble_boxes import *
import os

submission_files = [
    "submission_YOLO11x.txt",      # 0.9653 (YOLO11x)
    "submission_L_200_TTA.txt",    # 0.965  (YOLOv8l 200ep)
    "submission_HD_896.txt"        # 0.961  (YOLOv8l 896px)
]


weights = [3, 2, 1] 

iou_thr = 0.65       
skip_box_thr = 0.001 


preds = {}
for i, file_path in enumerate(submission_files):
    if not os.path.exists(file_path):
        print(f"❌ cannot find path：{file_path}")
        continue
    
    print(f"Read：{file_path} (weight: {weights[i]})")
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            fname = parts[0]
            cls = int(parts[1])
            conf = float(parts[2])
            x1 = float(parts[3]) / 512
            y1 = float(parts[4]) / 512
            x2 = float(parts[5]) / 512
            y2 = float(parts[6]) / 512
            
            if fname not in preds:
                preds[fname] = [{'boxes': [], 'scores': [], 'labels': []} for _ in range(len(submission_files))]
            
            preds[fname][i]['boxes'].append([x1, y1, x2, y2])
            preds[fname][i]['scores'].append(conf)
            preds[fname][i]['labels'].append(cls)

output_filename = "submission_Ensemble_3_Models_Target_097.txt"

with open(output_filename, 'w') as f_out:
    for fname, run_data in preds.items():
        boxes, scores, labels = weighted_boxes_fusion(
            [run['boxes'] for run in run_data],
            [run['scores'] for run in run_data],
            [run['labels'] for run in run_data],
            weights=weights, 
            iou_thr=iou_thr, 
            skip_box_thr=skip_box_thr
        )
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = int(box[0]*512), int(box[1]*512), int(box[2]*512), int(box[3]*512)
            f_out.write(f"{fname} {int(label)} {score:.4f} {x1} {y1} {x2} {y2}\n")

print(f"🎉 最終融合完成！請下載 {output_filename}")
