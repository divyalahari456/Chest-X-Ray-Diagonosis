# chest_xray_classifier.py

"mounted my google drive with colab to use the datasets which are uploaded in google drive"


from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
from tqdm import tqdm
import shutil

# Paths
nih_csv_path = '/content/drive/MyDrive/Datasets/NIH-Chest-X-ray/unzipped/Data_Entry_2017.csv'
nih_images_path = '/content/drive/MyDrive/Datasets/NIH-Chest-X-ray/unzipped/images'
covid_images_path = '/content/drive/MyDrive/Datasets/covid/unzipped/COVID-19_Radiography_Dataset/COVID/images'
target_base = '/content/drive/MyDrive/Datasets/chestxray_8class'

# Load NIH labels
df = pd.read_csv(nih_csv_path)

classes = ['COVID', 'No Finding', 'Pneumonia', 'Cardiomegaly',
           'Effusion', 'Infiltration', 'Atelectasis', 'Mass']

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(target_base, cls.replace(' ', '_')), exist_ok=True)

nih_classes = classes[1:]  # skip 'COVID' for now

def collect_images_for_class(label, max_images=3000):
    label_folder = os.path.join(target_base, label.replace(' ', '_'))
    count = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        labels = row['Finding Labels'].split('|')
        if label in labels:
            filename = row['Image Index']
            src = os.path.join(nih_images_path, filename)
            dst = os.path.join(label_folder, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
                count += 1
        if count >= max_images:
            break

for label in nih_classes:
    collect_images_for_class(label)

def collect_covid_images(max_images=3000):
    dst_dir = os.path.join(target_base, 'COVID')
    os.makedirs(dst_dir, exist_ok=True)

    count = 0
    for fname in tqdm(os.listdir(covid_images_path), desc="Copying COVID"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(covid_images_path, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                count += 1
        if count >= max_images:
            break

collect_covid_images()

