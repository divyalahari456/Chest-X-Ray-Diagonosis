#chest_xray_classifier

import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#training and validation paths
train_dir = "train_path"
val_dir   = "validation_path"

#given - classes
target_classes = ['Covid-19', 'Emphysema', 'Normal', 'Pneumonia-Bacterial']


#create training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

#create validation data generator without augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

#define training generator
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    classes=target_classes,
    subset="training",
    shuffle=True,
    seed=42
)

#define validation generator
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    classes=target_classes,
    subset="validation",
    shuffle=False,
    seed=42
)

print("Classes:", train_generator.class_indices)

#model building with pretrained model-densenet-121
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(len(target_classes), activation="softmax")(x)


model = models.Model(inputs=base_model.input, outputs=output_layer)

#compile model with initial learning rate
initial_optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=initial_optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#model-checkpoint
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath="best_densenet.h5",
        save_best_only=True
    )
]


#training the model
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

#fine-tuning
fine_tune_optimizer = optimizers.Adam(learning_rate=1e-5)
model.compile(
    optimizer=fine_tune_optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(
    val_generator,
    verbose=1
)
print(f"\nFinal DenseNet121 Accuracy ({len(target_classes)} classes): {acc*100:.2f}%")