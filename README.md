# Neuroscan
# NeuroScan — Brain Tumor MRI Classification

## 1. Project Overview

NeuroScan is an academic deep-learning project developed to classify brain MRI images into four categories using a MobileNetV2-based image classification model.

The project uses MRI images collected from the combined Epic and CSCR hospital datasets. A Streamlit web application provides an interface where users can upload an MRI image and view the model's predicted class.

### Objectives

- Classify brain MRI images into four categories.
- Apply deep learning and transfer learning using MobileNetV2.
- Develop a user-friendly Streamlit web interface.
- Evaluate model performance using a held-out test dataset.
- Demonstrate the application of AI in medical-image research.

### Classification Categories

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary Tumor

## 2. Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Model Architecture:** MobileNetV2
- **Web Framework:** Streamlit
- **Image Processing:** Pillow and NumPy
- **Dataset:** Combined Epic and CSCR hospital MRI datasets
- **Development Environment:** Visual Studio Code

## 3. Model Architecture

NeuroScan uses MobileNetV2, a lightweight convolutional neural network designed for efficient image classification.

MobileNetV2 uses inverted residual blocks and depthwise separable convolutions to reduce computational cost while extracting useful image features.

### Model Workflow

1. **Input Layer:** Receives a brain MRI image.
2. **Preprocessing:** Resizes the image and applies the same pixel normalization or preprocessing used during model training.
3. **MobileNetV2 Backbone:** Extracts important visual features from the MRI image.
4. **Classification Head:** Converts the extracted features into scores for the four target classes.
5. **Prediction Layer:** Produces the predicted class using the model's output scores.
6. **Streamlit Interface:** Displays the uploaded image and prediction to the user.

The exact input dimensions, classification head, and preprocessing method must match the trained model implementation.

### Model Performance

The reported test accuracy of the MobileNetV2 model is approximately **79.54%** in the project's evaluation.

This result represents performance on the evaluated dataset only. It does not establish clinical accuracy or reliability in real-world medical settings.

## 4. Dataset Structure

The project uses a combined MRI dataset from Epic and CSCR hospitals, organized into separate training and testing directories.

Expected directory structure:

```text
NEUROSCAN/
│
├── app/
│   ├── streamlit_app.py
│   └── predictor.py
│
├── data/
│   └── Epic and CSCR hospital Dataset/
│       ├── Train/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       │
│       └── Test/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
│
├── models/
│   ├── neuroscan_mobilenetv2.keras
│   └── mobilenet_class_names.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Dataset directory descriptions:**

- `Train/`: Contains images used for model training.
- `Test/`: Contains images reserved for model evaluation.
- `glioma/`: Images labelled as glioma.
- `meningioma/`: Images labelled as meningioma.
- `notumor/`: Images labelled as no tumor.
- `pituitary/`: Images labelled as pituitary tumor.

Ensure that folder names and class-index mappings match those used during training. If your existing dataset uses names such as `notumer` or `pituitarya`, verify the labels before renaming folders.

The paths shown above are the expected structure. Adjust them if your actual project uses different locations or filenames.

## 5. Installation Requirements

Before running the application, install:

- Python compatible with the TensorFlow version in `requirements.txt`
- pip package manager
- Visual Studio Code
- Required Python libraries
- The trained model file and its class-name mapping

### Step 1: Open the project folder

Open VS Code and select the main NeuroScan project folder.

Open a terminal using:

**Terminal → New Terminal**

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Step 4: Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create it with the packages required by your implementation and compatible versions of TensorFlow, Streamlit, NumPy, and Pillow.

### Step 5: Verify the model files

Ensure that your project contains the trained model and class-name mapping at the locations expected by your prediction code.

For example:

```text
models/neuroscan_mobilenetv2.keras
models/mobilenet_class_names.json
```

These are example paths. Use the actual filenames configured in your project. Installing Python packages does not generate a trained model automatically.

## 6. Running the Streamlit Application

After installing the dependencies and confirming that the required model files are available, run the following command from the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

Streamlit will start a local web server. Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

### How to Use the Application

1. Open NeuroScan in your browser.
2. Upload a supported brain MRI image.
3. Allow the application to preprocess the image.
4. View the predicted category and any confidence score displayed by the application.
5. Repeat with another image if required.

The displayed result is a model prediction, not a medical diagnosis.

To stop the application, press `Ctrl + C` in the terminal.

## 7. Reported Evaluation

| Metric | Result |
|---|---|
| Model | MobileNetV2 |
| Task | Four-class MRI image classification |
| Reported test accuracy | Approximately 79.54% |
| Application interface | Streamlit |

The reported accuracy should be interpreted in the context of the dataset and evaluation procedure used in this project.

For a more complete evaluation, consider reporting:

- Confusion matrix
- Precision and recall for each class
- F1-score for each class
- Sensitivity and specificity where appropriate
- Number of images evaluated in each class

## 8. Model Limitations and Medical Disclaimer

**Important: NeuroScan is an academic project, not a clinically validated diagnostic tool.**

The MobileNetV2 model achieved approximately **79.54% test accuracy** in the reported project evaluation. This result does not mean the model is suitable for diagnosing brain tumors in patients.

The project has the following limitations:

- **Limited validation:** The reported evaluation does not establish clinical validity or safety.
- **Dataset dependency:** Performance may change when images come from different hospitals, scanners, acquisition protocols, or patient populations.
- **Classification errors:** The model may incorrectly classify an MRI image or produce misleading confidence scores.
- **Data quality:** Image quality, class imbalance, incorrect labels, and preprocessing differences can affect performance.
- **Incomplete evaluation:** Overall accuracy alone does not show how often each tumor category is incorrectly classified.
- **Generalization:** Performance on the evaluated test set does not guarantee similar performance on unseen real-world medical images.

### Intended Use

This application is intended only for academic learning, research demonstrations, and educational purposes.

**Do not use NeuroScan to diagnose or rule out a brain tumor, make treatment decisions, or replace professional medical evaluation.** Medical concerns should be assessed by qualified healthcare professionals using appropriate clinical procedures.

## 9. Troubleshooting

### Streamlit command not found

Activate the virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Model file not found

Check that the model path in `app/predictor.py` matches the actual saved model location.

### Incorrect class predictions

Verify the image preprocessing pipeline, image dimensions, RGB channel order, normalization method, and class-index mapping.

### Dataset folder errors

Check that the training and testing folders contain the expected class names and that the dataset-loading code uses the correct directory paths.

### TensorFlow installation errors

Check Python and TensorFlow version compatibility, then install the versions specified by your project dependencies.

## 10. Future Enhancements

- Improve model performance through additional training and appropriate data augmentation.
- Evaluate generalization on an independent dataset.
- Add class-wise evaluation metrics and a confusion matrix.
- Improve error handling and image validation.
- Add model versioning and reproducible evaluation reports.
- Conduct appropriate clinical research and independent validation before considering any clinical application.

## 11. Conclusion

NeuroScan demonstrates the use of MobileNetV2 and Streamlit for academic brain MRI image classification. It provides a practical example of integrating deep learning with a web interface while highlighting the importance of evaluation, model limitations, and responsible use of AI in healthcare.

**Project:** NeuroScan — Brain Tumor MRI Classification  
**Architecture:** MobileNetV2  
**Interface:** Streamlit  
**Reported test accuracy:** Approximately 79.54%  
**Purpose:** Academic and educational use only
