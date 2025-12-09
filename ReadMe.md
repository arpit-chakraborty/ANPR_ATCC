````markdown
# Automated Vision System: ANPR & Traffic Classification

This repository contains two main components for automated vision tasks:

1.  A **Streamlit web application (`app.py`)** for real-time inference, featuring **Automatic Number Plate Recognition (ANPR)** and **Automatic Traffic Count and Classification (ATCC)**.
2.  **Jupyter notebooks** detailing the training and data preparation processes for the underlying **YOLOv8** detection models.

***

## 🚀 Features

* **ANPR (Automatic Number Plate Recognition):** Detects license plates in images or videos using a trained YOLO model (`yolo_ANPR.pt`) and performs Optical Character Recognition (OCR) on the detected plates.
* **ATCC (Automatic Traffic Count and Classification):** Detects and classifies various traffic objects (e.g., cars, pedestrians, signs) for counting and analysis, using a trained YOLO model (`yolo_ATCC.pt`).
* **Interactive UI:** A user-friendly interface built with **Streamlit** to easily upload and process files.
* **Data Logging:** Generates a downloadable **CSV log** of all detections during video processing.

***

## 🛠️ Setup and Installation

### Prerequisites

You need a Python environment (**Python 3.8+** recommended).

### 1. Install Python Dependencies

The core application and training processes rely on the following libraries:

| Dependency | Purpose |
| :--- | :--- |
| `ultralytics` | Core library for YOLOv8 model handling and training. |
| `streamlit` | Used to run the interactive web application (`app.py`). |
| `fast_plate_ocr` | OCR library used specifically for the ANPR functionality in `app.py`. |
| `opencv-python`, `numpy`, `pillow`, `pandas` | General computer vision and data processing utilities. |

```bash
# Recommended installation for Streamlit app and basic usage
pip install -qU ultralytics opencv-python pillow pandas streamlit

# Install OCR library (needed for ANPR functionality)
pip install fast_plate_ocr

# If replicating training, ensure notebook dependencies are met
pip install numpy opencv-python-headless
````

### 2\. Model Files

The Streamlit application requires the following trained YOLO model weights (`.pt` files) in the root directory:

  * `yolo_ANPR.pt` (for license plate detection)
  * `yolo_ATCC.pt` (for traffic object detection)

-----

## 💡 Usage

### Running the Streamlit App

Start the application from your terminal:

```bash
streamlit run app.py
```

### Application Workflow

1.  Select the **Model Type** in the sidebar: **ANPR** or **ATCC**.
2.  **Upload** an image or video file via the interface.
3.  The application processes the file, displaying the annotated result and providing a **Download Processed Video/Image** button and a **Detailed Detection Log** (for videos).

-----

## 📚 Training and Data Preparation

The provided Jupyter notebooks detail the training and data preparation processes used to create the models.

### 1\. ANPR Model Training (`anpr-license-traning.ipynb`)

  * **Model:** **YOLOv8** trained specifically for license plate detection.
  * **Target Class:** Single class: `license_plate` (0).
  * **Training Details:** Trained using the `ultralytics` library for 30 epochs.
  * **Validation Performance:** Final mAP50: **0.854**, mAP50-95: **0.475**.

### 2\. ATCC Model Training (`atcc-bdd100k.ipynb`)

  * **Model:** **YOLOv8** trained for multi-class traffic object classification.
  * **Dataset:** Uses the **BDD100K** dataset.
  * **Preprocessing:** The notebook provides scripts to convert BDD100K JSON annotations into the standard YOLO format (`.txt` files) with normalized bounding box coordinates.
  * **Classes:** 10 traffic-related classes including `car`, `truck`, `bus`, `pedestrian`, `traffic light`, etc.
  * **Validation Performance:** Final mAP50: **0.587**, mAP50-95: **0.325**.

<!-- end list -->
