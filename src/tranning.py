import locale
import ultralytics
import gdown
import os
import shutil
from ultralytics import YOLO


def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding

ultralytics.checks()


path = (
    "https://drive.google.com/uc?export=download&id=1taWLMdSNG3bgMbxpmBwgn0cn4MeAzY5A"
)
path2 = (
    "https://drive.google.com/uc?export=download&id=1Ave-crFSqT2nrJ8x9sgMHl8FsmvGjwNO"
)
path3 = (
    "https://drive.google.com/uc?export=download&id=1sGGZIo-Ys5cyhN6qOpVhbAZBpQyvDR4y"
)

gdown.download(path, "/content/training_image.zip")
gdown.download(path2, "/content/training_label.zip")
gdown.download(path3, "/content/aortic_valve_colab.yaml")


def find_patient_root(root):
    """往下找，直到找到含有 patientXXXX 的資料夾"""
    for dirpath, dirnames, filenames in os.walk(root):
        if any(d.startswith("patient") for d in dirnames):
            return dirpath
    return root  # fallback


IMG_ROOT = find_patient_root("./training_image")
LBL_ROOT = find_patient_root("./training_label")

print("IMG_ROOT =", IMG_ROOT)
print("LBL_ROOT =", LBL_ROOT)


def ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


ensure_clean_dir("./datasets/train/images")
ensure_clean_dir("./datasets/train/labels")
ensure_clean_dir("./datasets/val/images")
ensure_clean_dir("./datasets/val/labels")


def move_patients(start, end, split):
    for i in range(start, end + 1):
        patient = f"patient{i:04d}"
        img_dir = os.path.join(IMG_ROOT, patient)
        lbl_dir = os.path.join(LBL_ROOT, patient)
        if not os.path.isdir(lbl_dir):
            continue

        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(lbl_dir, fname)
            base, _ = os.path.splitext(fname)  # 取出檔名不含副檔名
            img_path = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_path):
                print(f"找不到對應圖片: {img_path}")
                continue

            shutil.copy2(img_path, f"./datasets/{split}/images/")
            shutil.copy2(label_path, f"./datasets/{split}/labels/")


move_patients(1, 30, "train")

move_patients(31, 50, "val")

print("完成移動！")


print("訓練集圖片數量 : ", len(os.listdir("./datasets/train/images")))
print("訓練集標記數量 : ", len(os.listdir("./datasets/train/labels")))
print("驗證集圖片數量 : ", len(os.listdir("./datasets/val/images")))
print("驗證集標記數量 : ", len(os.listdir("./datasets/val/labels")))


model = YOLO("yolo12n.pt")
results = model.train(
    data="./aortic_valve_colab.yaml",
    epochs=20,  # 跑幾個epoch
    batch=16,  # batch_size
    imgsz=640,  # 圖片大小640*640
    device=0,  # 使用GPU進行訓練
)
