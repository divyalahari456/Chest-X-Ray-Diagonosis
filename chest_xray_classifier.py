#chest_xray_classifier
# drive.mount('/content/drive', force_remount=True)

# nih_root_path = '/content/drive/MyDrive/Datasets/NIH-Chest-X-ray/unzipped'
# merged_image_path = os.path.join(nih_root_path, 'all_images_fast')
# os.makedirs(merged_image_path, exist_ok=True)

# image_paths = glob.glob(os.path.join(nih_root_path, 'images_*/images/*.png'))
# image_paths = image_paths[:60000]

# print(f"\nFound {len(image_paths)} images. Starting copy...")

# for i, filepath in enumerate(tqdm(image_paths, desc="Copying to all_images_fast")):
#     filename = os.path.basename(filepath)
#     dst = os.path.join(merged_image_path, filename)
#     if not os.path.exists(dst):
#         shutil.copy(filepath, dst)

# print("\nDone copying images to all_images_fast\n")

# nih_csv_path = os.path.join(nih_root_path, 'Data_Entry_2017.csv')
# covid_images_path = '/content/drive/MyDrive/Datasets/covid/unzipped/COVID-19_Radiography_Dataset/COVID/images'
# target_base = '/content/drive/MyDrive/Datasets/chestxray_8class_fast'

# classes = ['COVID', 'No Finding', 'Pneumonia', 'Cardiomegaly',
#            'Effusion', 'Infiltration', 'Atelectasis', 'Mass']

# for cls in classes:
#     os.makedirs(os.path.join(target_base, cls.replace(' ', '_')), exist_ok=True)

# print("\nCollecting NIH class images...")
# df = pd.read_csv(nih_csv_path)
# nih_images_path = merged_image_path
# def collect_images_for_class(label, max_images=1000):
#     label_folder = os.path.join(target_base, label.replace(' ', '_'))
#     os.makedirs(label_folder, exist_ok=True)

#     filtered_df = df[df['Finding Labels'].str.contains(label, case=False, na=False)].copy()
#     filtered_df = filtered_df.sample(frac=1, random_state=42)

#     count = 0
#     for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"Collecting {label}"):
#         if count >= max_images:
#             break

#         try:
#             filename = row['Image Index'].strip()
#             src = os.path.join(nih_images_path, filename)
#             dst = os.path.join(label_folder, filename)

#             if os.path.exists(src) and not os.path.exists(dst):
#                 shutil.copy(src, dst)
#                 count += 1

#         except Exception as e:
#             print(f"[ERROR] {e}")

#     print(f"{label}: {count} images copied.")

# nih_classes = classes[1:]
# for label in nih_classes:
#     collect_images_for_class(label, max_images=1000)

# def collect_covid_images(max_images=1000):
#     dst_dir = os.path.join(target_base, 'COVID')
#     os.makedirs(dst_dir, exist_ok=True)
#     count = 0

#     for fname in tqdm(os.listdir(covid_images_path), desc="Collecting COVID"):
#         if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
#             src = os.path.join(covid_images_path, fname)
#             dst = os.path.join(dst_dir, fname)

#             if os.path.exists(src) and not os.path.exists(dst):
#                 shutil.copy(src, dst)
#                 count += 1

#         if count >= max_images:
#             break

#     print(f"COVID: {count} images copied.")

# collect_covid_images(max_images=1500)

# print("\nDataset prepared with 8 class folders")
# import tensorflow as tf
# from tensorflow.keras import layers

# data_dir = '/content/drive/MyDrive/Datasets/chestxray_8class_fast'
# img_size = (224, 224)
# batch_size = 32
# seed = 42

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='training',
#     seed=seed,
#     image_size=img_size,
#     batch_size=batch_size
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset='validation',
#     seed=seed,
#     image_size=img_size,
#     batch_size=batch_size
# )

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# data_augmentation = tf.keras.Sequential([
#     layers.Rescaling(1./255),  # Normalize pixel values (0-1)
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.05),
#     layers.RandomZoom(0.1),
#     layers.RandomTranslation(0.1, 0.1),
#     layers.RandomContrast(0.2),
# ])
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# CONFIGURATION
base_dir = "./data"                      # path to your original dataset (with train/val/test)
output_dir = "./dataset_70_15_15"        # path to store new split dataset
class_to_remove = "Viral Pneumonia"      # set None if not removing anything

input_size = (224, 224, 3)
batch_size = 32
num_classes = 4
epochs_head = 5
epochs_finetune = 20
lr_head = 1e-3
lr_ft = 1e-5
dropout_rate = 0.3
random_state = 42


# DATASET SPLITTING (70 / 15 / 15)
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

new_train_dir = os.path.join(output_dir, "train")
new_val_dir = os.path.join(output_dir, "val")
new_test_dir = os.path.join(output_dir, "test")

os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)
shutil.copytree(test_dir, new_test_dir, dirs_exist_ok=True)  # keep test unchanged

classes = os.listdir(train_dir)
for cls in classes:
    print(f"Processing {cls}...")
    cls_train = [os.path.join(train_dir, cls, f) for f in os.listdir(os.path.join(train_dir, cls))]
    cls_val = [os.path.join(val_dir, cls, f) for f in os.listdir(os.path.join(val_dir, cls))]
    all_images = cls_train + cls_val

    # 70% train, 15% val (from combined 85%)
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.176, random_state=random_state, shuffle=True)

    os.makedirs(os.path.join(new_train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(new_val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, os.path.join(new_train_dir, cls, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(new_val_dir, cls, os.path.basename(img)))

print("✅ Done! New dataset is in:", output_dir)


# REMOVING A CLASS (OPTIONAL)
if class_to_remove:
    splits = ["train", "val", "test"]
    for split in splits:
        class_path = os.path.join(output_dir, split, class_to_remove)
        if os.path.exists(class_path):
            shutil.rmtree(class_path)
            print(f"✅ Deleted: {class_path}")
        else:
            print(f"⚠️ Not found: {class_path}")



# DATASET SUMMARY
splits = ["train", "val", "test"]
for split in splits:
    split_path = os.path.join(output_dir, split)
    print(f"\n🔹 {split.upper()} split:")
    total_images = 0
    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   {cls}: {count}")
            total_images += count
    print(f"   ➡ Total {split} images: {total_images}")



# MODEL TRAINING
data_dir = output_dir
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.9, 1.1],
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_size[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_densenet_4class.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Build model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_size)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout_rate)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Train classifier head only
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr_head),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Training Head ---\n")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_head,
    callbacks=callbacks
)

# Phase 2: Fine-tune entire model
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr_ft),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Fine-tuning Whole Model ---\n")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_finetune,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print("✅ Best model saved as 'best_densenet_4class.h5'")
