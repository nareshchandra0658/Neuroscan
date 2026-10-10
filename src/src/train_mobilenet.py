
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

DATA_DIR = (
    r"D:\NEUROSCAN\NEUROSCAN\data"
    r"\Epic and CSCR hospital Dataset"
)

TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")

MODEL_DIR = r"D:\NEUROSCAN\NEUROSCAN\models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR, "neuroscan_mobilenetv2.keras"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("=" * 55)
print("NEUROSCAN - MOBILENETV2 TRAINING")
print("=" * 55)

# --------------------------------------------------
# 2. CHECK DATASET FOLDERS
# --------------------------------------------------

for folder in [TRAIN_DIR, TEST_DIR]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(folder, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

print("Dataset folders verified.")

# --------------------------------------------------
# 3. LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="int",
    validation_split=0.20,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="int",
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
    label_mode="int",
    shuffle=False
)

print("Class mapping:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# 4. CALCULATE CLASS WEIGHTS
# --------------------------------------------------

class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

for _, labels in train_ds:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=len(CLASS_NAMES)
    )

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=np.repeat(
        np.arange(len(CLASS_NAMES)),
        class_counts
    )
)

class_weights = {
    i: float(weights[i])
    for i in range(len(CLASS_NAMES))
}

print("Training class counts:", class_counts)
print("Class weights:", class_weights)

# --------------------------------------------------
# 5. DATA AUGMENTATION
# --------------------------------------------------

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomZoom(0.08)
], name="data_augmentation")

# --------------------------------------------------
# 6. LOAD MOBILENETV2
# --------------------------------------------------

# ImageNet weights normally download on the first run.
# An internet connection is required for this step.

base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze pretrained layers.
base_model.trainable = False

inputs = layers.Input(
    shape=(*IMAGE_SIZE, 3),
    name="mri_image"
)

x = augmentation(inputs)

# MobileNetV2 expects input values approximately in [-1, 1].
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="tumor_class"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 7. CALLBACKS
# --------------------------------------------------

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# --------------------------------------------------
# 8. FIRST TRAINING STAGE
# --------------------------------------------------

print("\nSTAGE 1: Training classification layers")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

# --------------------------------------------------
# 9. FINE-TUNE LAST MOBILENETV2 LAYERS
# --------------------------------------------------

print("\nSTAGE 2: Fine-tuning selected pretrained layers")

base_model.trainable = True

# Freeze most layers; train only the final 30 layers,
# keeping BatchNormalization layers frozen.
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False
    else:
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history1.epoch),
    epochs=len(history1.epoch) + FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
            verbose=1
        )
    ]
)

# --------------------------------------------------
# 10. LOAD BEST CHECKPOINT
# --------------------------------------------------

best_model = tf.keras.models.load_model(MODEL_PATH)

print("\nBest saved model loaded.")

# --------------------------------------------------
# 11. EVALUATE ON THE TEST DATASET
# --------------------------------------------------

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 12. CLASSIFICATION REPORT
# --------------------------------------------------

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

with open(
    os.path.join(MODEL_DIR, "mobilenet_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)
    file.write(
        f"\nTest accuracy: {test_accuracy * 100:.2f}%\n"
    )

# --------------------------------------------------
# 13. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 7))
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=30,
    values_format="d"
)
ax.set_title("NeuroScan MobileNetV2 - Test Confusion Matrix")
fig.tight_layout()

fig.savefig(
    os.path.join(MODEL_DIR, "mobilenet_confusion_matrix.png"),
    dpi=200
)
plt.close(fig)

# --------------------------------------------------
# 14. TRAINING GRAPHS
# --------------------------------------------------

train_acc = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)
val_acc = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)
train_loss = (
    history1.history["loss"]
    + history2.history["loss"]
)
val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

epochs = range(1, len(train_acc) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs, train_acc, label="Training")
axes[0].plot(epochs, val_acc, label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, train_loss, label="Training")
axes[1].plot(epochs, val_loss, label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
fig.savefig(
    os.path.join(MODEL_DIR, "mobilenet_training_graphs.png"),
    dpi=200
)
plt.close(fig)

# --------------------------------------------------
# 15. SAVE CLASS NAMES AND TRAINING HISTORY
# --------------------------------------------------

with open(
    os.path.join(MODEL_DIR, "mobilenet_class_names.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(CLASS_NAMES, file, indent=4)

combined_history = {
    "accuracy": train_acc,
    "val_accuracy": val_acc,
    "loss": train_loss,
    "val_loss": val_loss
}

with open(
    os.path.join(MODEL_DIR, "mobilenet_history.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(combined_history, file, indent=4)

print("\n" + "=" * 55)
print("NEUROSCAN MOBILENETV2 TRAINING COMPLETED")
print("=" * 55)
print("Best model:", MODEL_PATH)
print("Test accuracy:", f"{test_accuracy * 100:.2f}%")
print("Report: mobilenet_report.txt")
print("Confusion matrix: mobilenet_confusion_matrix.png")
print("Training graphs: mobilenet_training_graphs.png")
print("Class names: mobilenet_class_names.json")
print("History: mobilenet_history.json")
