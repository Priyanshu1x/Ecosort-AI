# EcoSort-AI: Explainable Waste Classification System

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

EcoSort-AI is a computer vision system for automated waste classification across four categories: **Dry, Wet, Recyclable, and E-waste**.

The project combines **deep learning, transfer learning, and explainable AI** in a local Streamlit application. In addition to returning a predicted waste category, the application generates a **Grad-CAM heatmap** to visualize the image regions that contributed to the model's decision.

The system was developed with a focus on balancing **classification performance, computational efficiency, interpretability, and practical local deployment**.

---

## Project Overview

Manual waste segregation is often inconsistent and difficult to scale. EcoSort-AI addresses this problem by using image classification to automatically identify the category of a waste item from an uploaded image.

The application currently supports four classes:

| Class                | Description                               |
| -------------------- | ----------------------------------------- |
| **Dry Waste**        | General dry, non-wet waste                |
| **Wet Waste**        | Organic or moisture-containing waste      |
| **Recyclable Waste** | Materials suitable for recycling          |
| **E-waste**          | Discarded electronic and electrical items |

### Key Capabilities

* Image-based waste classification using PyTorch
* Four-class classification: **Dry, Wet, Recyclable, E-waste**
* Transfer learning with pretrained CNN architectures
* Local inference through a **Streamlit** web interface
* Image processing using **OpenCV**
* **Grad-CAM** visualization for model interpretability
* Comparative evaluation of multiple CNN architectures

The primary objective is not only to achieve higher classification accuracy, but also to make the model's predictions more interpretable through visual explanations.

---

## Architecture & Design Decisions

The model development process followed an iterative architecture evaluation strategy.

### 1. Custom Baseline CNN

The project initially used a custom CNN as a baseline model.

While the model was able to learn the general visual characteristics of the four waste categories, it struggled with classes containing visually similar objects, particularly **Dry** and **Recyclable** waste.

The baseline achieved:

**67.69% overall accuracy**

This established a reference point for evaluating more capable architectures.

---

### 2. Transfer Learning

To improve feature extraction and generalization, the project was upgraded to transfer learning using pretrained convolutional neural networks.

One of the evaluated architectures was **ResNet18**, which achieved:

**85.44% overall accuracy**

This represented a substantial improvement over the custom CNN, indicating that pretrained visual representations were better suited to the classification problem.

---

### 3. EfficientNet-B0 — Final Model

The final system uses **EfficientNet-B0**.

EfficientNet-B0 was selected because it provided the strongest classification performance among the evaluated models while maintaining a relatively lightweight computational footprint.

This trade-off is important for EcoSort-AI because inference is intended to run through a **local Streamlit application**, where excessive model complexity would unnecessarily increase resource requirements and inference latency.

The final EfficientNet-B0 model achieved:

**87.99% overall accuracy**

The resulting architecture provides a practical balance between:

* Classification accuracy
* Model size and computational requirements
* Local inference feasibility
* Compatibility with explainability techniques such as Grad-CAM

### Model Selection Summary

```text
Custom Baseline CNN
        │
        │  67.69% accuracy
        ▼
Transfer Learning
        │
        ├── ResNet18
        │      85.44% accuracy
        │
        └── EfficientNet-B0
               87.99% accuracy
               ↓
          Final Deployed Model
```

---

## Performance Metrics

The evaluated models produced the following overall accuracy results:

| Model               | Overall Accuracy | Role                         |
| ------------------- | ---------------: | ---------------------------- |
| Custom Baseline CNN |       **67.69%** | Initial baseline             |
| ResNet18            |       **85.44%** | Transfer-learning comparison |
| EfficientNet-B0     |       **87.99%** | Final deployed model         |

Additional class-level metrics for the final EfficientNet-B0 model include:

| Metric    | Class     |      Score |
| --------- | --------- | ---------: |
| Precision | Wet Waste | **94.68%** |
| Recall    | E-waste   |    **91%** |

These results demonstrate that model performance varies across waste categories, which is expected in a classification problem where visual characteristics can overlap between classes.

In particular, the progression from **67.69% with the baseline CNN to 87.99% with EfficientNet-B0** shows the impact of using pretrained feature representations for this task.

---

## Explainable AI with Grad-CAM

A central component of EcoSort-AI is model interpretability.

Traditional image classifiers provide a class prediction without showing why the model made that prediction. EcoSort-AI addresses this limitation by generating a **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization alongside the prediction.

### Inference Flow

```text
User uploads image
        │
        ▼
Image preprocessing
        │
        ▼
EfficientNet-B0
        │
        ├──────────────► Predicted class
        │
        ▼
      Grad-CAM
        │
        ▼
Activation heatmap
        │
        ▼
Visualization in Streamlit
```

The heatmap highlights image regions that contributed most strongly to the selected prediction.

This provides a visual interpretation of the model's attention and can be useful for:

* Understanding model decisions
* Identifying whether predictions rely on meaningful visual features
* Debugging unexpected classifications
* Increasing transparency during model evaluation

Grad-CAM should be interpreted as a visualization of model activation/importance rather than as a guarantee that the highlighted region represents the true semantic cause of the object.

---

## Technology Stack

| Technology          | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| **Python**          | Core development language                      |
| **PyTorch**         | Model development and inference                |
| **EfficientNet-B0** | Final image classification model               |
| **OpenCV**          | Image processing and computer vision utilities |
| **Grad-CAM**        | Model interpretability                         |
| **Streamlit**       | Local web application and inference interface  |

---

## Setup & Installation

### 1. Clone the Repository

Replace the repository URL with the actual GitHub repository URL:

```bash
git clone [repository-url]
cd Ecosort-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows

**Command Prompt:**

```bash
.venv\Scripts\activate
```

**PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Once started, Streamlit will provide a local URL where the EcoSort-AI interface can be accessed in a browser.

---

## Repository Structure

A simplified repository layout is shown below:

```text
Ecosort-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── Dry/
│   ├── Wet/
│   ├── Recyclable/
│   └── ewaste/
│
├── checkpoints/
│   └── trained model checkpoints
│
└── ecosort/
    ├── model/
    ├── inference/
    ├── preprocessing/
    └── explainability/
```

> **Note:** The exact internal files under `ecosort/` may vary depending on the current implementation. The structure above represents the intended organization of the project.

---

## Application Workflow

The complete application workflow is:

```text
              ┌─────────────────┐
              │  Upload Image   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Preprocessing   │
              │   / OpenCV      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ EfficientNet-B0 │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      ┌──────────────┐    ┌──────────────┐
      │ Class        │    │ Grad-CAM     │
      │ Prediction   │    │ Explanation  │
      └──────────────┘    └──────┬───────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │ Heatmap      │
                         │ Visualization│
                         └──────────────┘
```

---

## Why Transfer Learning?

The baseline experiment showed that a custom CNN was insufficient for consistently distinguishing visually similar waste categories.

Transfer learning provides access to feature representations learned from large-scale image datasets. Instead of learning all visual features from scratch, the model can leverage pretrained representations and adapt them to the waste-classification task.

The observed performance progression was:

```text
Baseline CNN      → 67.69%
ResNet18          → 85.44%
EfficientNet-B0   → 87.99%
```

This experimental progression informed the final architecture rather than selecting a model solely based on theoretical complexity.

---

## Design Considerations

### Accuracy vs. Computational Cost

A larger or more computationally expensive architecture is not automatically preferable for this application.

EcoSort-AI targets local interactive inference, making computational efficiency an important deployment consideration. EfficientNet-B0 was therefore selected based on the combination of classification performance and practical inference requirements.

### Interpretability

Model accuracy alone does not explain how a prediction was produced.

The integration of Grad-CAM allows the application to expose model-focused image regions, providing an additional layer of transparency during inference and evaluation.

### Modular Development

The project separates core responsibilities such as:

```text
Data / Preprocessing
        ↓
Model
        ↓
Inference
        ↓
Explainability
        ↓
Application Interface
```

This structure makes it easier to evaluate alternative models, modify inference logic, and extend the application independently of the user interface.

---

## Future Scope

Several improvements can extend EcoSort-AI beyond the current local deployment:

### Cloud Deployment

Deploy the inference service to a cloud environment so that the model can be accessed remotely rather than requiring local execution.

Potential directions include:

* Containerized deployment
* REST API-based inference
* Cloud-hosted Streamlit application
* Scalable inference infrastructure

### Inference Optimization

Further optimize inference speed and resource usage for interactive applications.

Potential approaches include:

* Model quantization
* Reduced-precision inference
* ONNX/TorchScript-based optimization
* Batch and preprocessing optimization
* Hardware-specific acceleration

### Additional Improvements

Future iterations could also explore:

* Improved class-level performance through targeted data collection
* More robust evaluation on unseen waste images
* Additional waste categories
* Real-time camera-based classification
* Confidence calibration and uncertainty analysis
* Expanded explainability evaluation

---

## Project Objective

EcoSort-AI is designed as an end-to-end demonstration of how a computer vision model can progress from a simple baseline to a transfer-learning-based system and finally into an interpretable application.

The project emphasizes the complete ML workflow:

```text
Problem Definition
       ↓
Baseline Model
       ↓
Performance Evaluation
       ↓
Architecture Comparison
       ↓
Transfer Learning
       ↓
Final Model Selection
       ↓
Explainability
       ↓
Application Deployment
```

Rather than treating model accuracy as the only objective, the project considers **performance, computational practicality, interpretability, and deployment constraints** as part of the system design.
