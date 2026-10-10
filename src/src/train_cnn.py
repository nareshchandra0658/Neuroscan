
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(r"D:\NEUROSCAN\NEUROSCAN")

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "Epic and CSCR hospital Dataset"
)

TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

print("NeuroScan CNN Training Started")
print("TensorFlow version:", tf.__version__)
print("Dataset:", DATA_DIR)

# ==========================================
# 2. CHECK DATASET PATHS
# ==========================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Testing folder not found: {TEST_DIR}"
    )

for folder in [TRAIN_DIR, TEST_DIR]:
    missing = [
        name for name in CLASS_NAMES
        if not (folder / name).is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing classes in {folder}: {missing}"
        )

# ==========================================
# 3. LOAD AND PREPROCESS IMAGES
# ==========================================

print("\nLoading training and validation data...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

print("\nLoading independent test data...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass labels:")
for index, name in enumerate(CLASS_NAMES):
    print(index, "=", name)

# Improve data-loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==========================================
# 4. BUILD CNN MODEL
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),

    augmentation,

    # Normalize pixel values from 0-255 to 0-1
    tf.keras.layers.Rescaling(1.0 / 255),

    # CNN Block 1
    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # CNN Block 2
    tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # CNN Block 3
    tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # CNN Block 4
    tf.keras.layers.Conv2D(
        256, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Classification layers
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    # Four output classes
    tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )
], name="NeuroScan_CNN")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# ==========================================
# 5. TRAINING CALLBACKS
# ==========================================

model_path = MODEL_DIR / "neuroscan_cnn.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final training history
history_path = MODEL_DIR / "training_history.json"

with open(history_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        file,
        indent=4
    )

# ==========================================
# 7. LOAD BEST MODEL AND EVALUATE
# ==========================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(model_path)

print("\nEvaluating independent test dataset...")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_path = MODEL_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("NeuroScan - Test Confusion Matrix")
plt.tight_layout()

cm_path = MODEL_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.show()

# ==========================================
# 10. TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(
    history.history["val_loss"],
    label="Validation"
)
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

graph_path = MODEL_DIR / "training_graphs.png"
plt.savefig(graph_path, dpi=200)
plt.show()

# ==========================================
# 11. SAVE CLASS NAMES
# ==========================================

class_path = MODEL_DIR / "class_names.json"

with open(class_path, "w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\n" + "=" * 45)
print("NEUROSCAN TRAINING COMPLETED")
print("=" * 45)

print("Saved model:", model_path)
print("Saved classes:", class_path)
print("Saved history:", history_path)
print("Saved report:", report_path)
print("Saved confusion matrix:", cm_path)
print("Saved training graphs:", graph_path)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
