#chest_xray_classifier


from google.colab import drive
import os
import shutil
from tqdm import tqdm
import glob
import pandas as pd

drive.mount('/content/drive', force_remount=True)

nih_root_path = '/content/drive/MyDrive/Datasets/NIH-Chest-X-ray/unzipped'
merged_image_path = os.path.join(nih_root_path, 'all_images_fast')
os.makedirs(merged_image_path, exist_ok=True)

image_paths = glob.glob(os.path.join(nih_root_path, 'images_*/images/*.png'))
image_paths = image_paths[:60000]

print(f"\nFound {len(image_paths)} images. Starting copy...")

for i, filepath in enumerate(tqdm(image_paths, desc="Copying to all_images_fast")):
    filename = os.path.basename(filepath)
    dst = os.path.join(merged_image_path, filename)
    if not os.path.exists(dst):
        shutil.copy(filepath, dst)

print("\nDone copying images to all_images_fast\n")

nih_csv_path = os.path.join(nih_root_path, 'Data_Entry_2017.csv')
covid_images_path = '/content/drive/MyDrive/Datasets/covid/unzipped/COVID-19_Radiography_Dataset/COVID/images'
target_base = '/content/drive/MyDrive/Datasets/chestxray_8class_fast'

classes = ['COVID', 'No Finding', 'Pneumonia', 'Cardiomegaly',
           'Effusion', 'Infiltration', 'Atelectasis', 'Mass']

for cls in classes:
    os.makedirs(os.path.join(target_base, cls.replace(' ', '_')), exist_ok=True)

print("\nCollecting NIH class images...")
df = pd.read_csv(nih_csv_path)
nih_images_path = merged_image_path

def collect_images_for_class(label, max_images=1000):
    label_folder = os.path.join(target_base, label.replace(' ', '_'))
    os.makedirs(label_folder, exist_ok=True)

    filtered_df = df[df['Finding Labels'].str.contains(label, case=False, na=False)].copy()
    filtered_df = filtered_df.sample(frac=1, random_state=42)

    count = 0
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"Collecting {label}"):
        if count >= max_images:
            break

        try:
            filename = row['Image Index'].strip()
            src = os.path.join(nih_images_path, filename)
            dst = os.path.join(label_folder, filename)

            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
                count += 1

        except Exception as e:
            print(f"[ERROR] {e}")

    print(f"{label}: {count} images copied.")

nih_classes = classes[1:]
for label in nih_classes:
    collect_images_for_class(label, max_images=1000)

def collect_covid_images(max_images=1000):
    dst_dir = os.path.join(target_base, 'COVID')
    os.makedirs(dst_dir, exist_ok=True)
    count = 0

    for fname in tqdm(os.listdir(covid_images_path), desc="Collecting COVID"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(covid_images_path, fname)
            dst = os.path.join(dst_dir, fname)

            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
                count += 1

        if count >= max_images:
            break

    print(f"COVID: {count} images copied.")

collect_covid_images(max_images=1500)

print("\nDataset prepared with 8 class folders")
