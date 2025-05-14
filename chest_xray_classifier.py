# chest_xray_classifier.py


import pandas as pd
import numpy as np
from pathlib import Path

classes = [
    "Pneumonia", "Edema", "Atelectasis", "No Finding", 
    "COVID-19", "Lung Opacity", "Fibrosis"
]

sources = {
    "Pneumonia": Path("/content/drive/MyDrive/Datasets/Pneumonia"),
    "Edema": Path("/content/drive/MyDrive/Datasets/Edema"),
    "Atelectasis": Path("/content/drive/MyDrive/Datasets/Atelectasis"),
    "No Finding": Path("/content/drive/MyDrive/Datasets/No_Finding"),
    "COVID-19": Path("/content/drive/MyDrive/Datasets/COVID"),
    "Lung Opacity": Path("/content/drive/MyDrive/Datasets/Lung_Opacity"),
    "Fibrosis": Path("/content/drive/MyDrive/Datasets/Fibrosis")
}

output_dir = Path("/content/drive/MyDrive/Datasets/combined_dataset")
output_dir.mkdir(exist_ok=True)

data = []

for cls in classes:
    class_output_dir = output_dir / cls
    class_output_dir.mkdir(exist_ok=True)
    
    all_files = list(sources[cls].glob("*.jpg")) + list(sources[cls].glob("*.png")) + list(sources[cls].glob("*.jpeg"))
    selected_files = np.random.choice(all_files, size=1500, replace=False)
    
    for file_path in selected_files:
        new_path = class_output_dir / file_path.name
        new_path.write_bytes(file_path.read_bytes())
        data.append({
            "image_path": str(new_path),
            "label": cls
        })

df = pd.DataFrame(data)
df.to_csv("/content/drive/MyDrive/Datasets/dataset.csv", index=False)