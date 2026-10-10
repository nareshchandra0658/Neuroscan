
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "neuroscan_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "mobilenet_class_names.json"

IMAGE_SIZE = (160, 160)


def load_model_and_classes():
    """Load the trained model and its class labels."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != model.output_shape[-1]:
        raise ValueError(
            "The number of class names does not match "
            "the model output."
        )

    return model, class_names


def prepare_image(image):
    """Convert an uploaded image into the model's input format."""

    image = image.convert("RGB")
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image, dtype=np.float32)

    # Keep pixel values in the 0-255 range.
    # The saved model performs its own MobileNetV2 rescaling.
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, class_names, image):
    """Return the predicted class and all class probabilities."""

    image_array = prepare_image(image)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    results = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": results,
    }
