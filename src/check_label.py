import zipfile
import os
import shutil
import filecmp

zip_file_path = "tbrain-42人工標注資料用.zip"  
extract_to = "./teacher_data_temp" 
original_label_dir = "./datasets/train/labels"

if os.path.exists(extract_to):
    shutil.rmtree(extract_to) 

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

new_label_dir = None
for root, dirs, files in os.walk(extract_to):
    if 'trainlabel' in dirs: 
        new_label_dir = os.path.join(root, 'trainlabel')
        break
    if len([f for f in files if f.endswith('.txt')]) > 10:
        if 'train' in os.path.basename(root).lower():
            new_label_dir = root
            break


    
    
    if not os.path.exists(original_label_dir):
        original_files = set()
    else:
        original_files = set([f for f in os.listdir(original_label_dir) if f.endswith('.txt')])
    
    new_files_set = set([f for f in os.listdir(new_label_dir) if f.endswith('.txt')])
    
    added_files = new_files_set - original_files
    common_files = new_files_set & original_files
    modified_files = []
    
    for f in common_files:
        path_old = os.path.join(original_label_dir, f)
        path_new = os.path.join(new_label_dir, f)
        
        with open(path_old, 'r') as f1, open(path_new, 'r') as f2:
            if f1.read().strip() != f2.read().strip():
                modified_files.append(f)

    print("="*40)
    print(f"1. Ok/Error： {len(added_files)}")
    print(f"2. different： {len(modified_files)}")
    print("="*40)
    
    if len(added_files) > 0 or len(modified_files) > 0:
        print("🎉 Ok")
    else:
        print("😐 Same")
