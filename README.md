# Histopathologic Cancer Detection

End-to-End Deep Learning Pipeline with CNN Training, Evaluation, and Dockerized Inference API

## 1. Project Overview

This project delivers a complete, end-to-end machine learning solution for binary classification of histopathologic image patches, distinguishing between cancerous and non-cancerous tissue samples.

The solution is built around a Convolutional Neural Network (CNN) implemented in PyTorch, specifically designed to learn spatial patterns from microscopic image data. The project follows a structured and reproducible ML workflow, covering the full lifecycle from raw data exploration to model inference.

Key components of the project include:

Exploratory data analysis and visualization to understand image characteristics and class balance

Iterative model development, including multiple CNN architectures and hyperparameter variations

Training and validation of models with systematic performance comparison

Selection of a final, stable model based on validation accuracy and learning behavior

Packaging of the trained model into a reusable inference pipeline

The final outcome is a reproducible machine learning system capable of accepting a histopathologic image as input and returning a binary cancer prediction with an associated confidence score. The project demonstrates both strong technical execution and practical considerations required for deploying deep learning models in applied medical imaging contexts.

## 2. Problem Statement

Histopathologic image analysis is a critical but time-consuming task in medical diagnostics.
The objective of this project is to:

Automatically classify histopathologic tissue image patches as cancerous or non-cancerous using deep learning.

This is formulated as a binary image classification problem.

## 3. Dataset Description

Dataset: Histopathologic Cancer Detection (Kaggle)

🔗 https://www.kaggle.com/competitions/histopathologic-cancer-detection

The dataset consists of histopathologic image patches used for binary classification of cancer presence at the tissue level. Each image represents a fixed-size patch extracted from larger whole-slide images commonly used in digital pathology workflows.

Image Characteristics:

File format: .tif (TIFF)

Image type: RGB histopathologic image patches

Patch size: Fixed-size image patches (consistent dimensions across the dataset)

Labels:

Each image is associated with a binary label indicating the presence of cancerous tissue:

1 → Cancer

0 → Non-Cancer

Labels are provided in a CSV file mapping image identifiers to their corresponding class.


## 4. Project Structure
```
histopathologic-cancer-detection/
│
├── data/
│   ├── train_labels.csv
│   ├── subset_labels.csv
│   └── plots/
│
├── plots/
│   ├── class_distribution.png
│   ├── cancer_samples.png
│   ├── non_cancer_samples.png
│   ├── mean_images.png
│   ├── pixel_intensity_hist.png
│   ├── cnn_v1_val_accuracy.png
│   ├── cnn_v2_val_accuracy.png
│   └── cnn_v3_val_accuracy.png
│
├── eda.ipynb
├── Model_Training.ipynb
│
├── train.py
├── predict.py
├── predict_cloud.py
│
├── model_cnn_v1.pth
│
├── requirements.txt
├── requirements-train.txt
│
├── Dockerfile
├── Dockerfile.cloud
├── .dockerignore
├── .gitignore
├── .gitattributes
│
└── README.md
```


## 5. Exploratory Data Analysis (EDA)


EDA was performed to understand data characteristics and guide model design.

Key EDA Steps

Class distribution analysis

Visualization of cancer vs non-cancer samples

Pixel intensity distribution

Mean image visualization per class

Files: eda.ipynb

Output plots stored in /plots

These analyses confirmed:

Binary class imbalance considerations

The need for convolutional feature extraction

Suitability of CNN-based approach


## 6. Model Development & Training

#### Objective

The objective of the modeling phase was to evaluate multiple machine learning and deep learning approaches for binary classification of histopathologic images (cancer vs. non-cancer).
Models were trained incrementally, starting from simple baselines and progressing toward convolutional neural networks, allowing performance and stability comparisons across model families.

All models were trained and evaluated using a balanced subset of the dataset to ensure fair performance assessment.

### Classical Machine Learning Models:

Classical models were trained using flattened image pixel values to establish baseline performance before applying deep learning.

### Logistic Regression

Purpose:
Logistic Regression was used as a baseline linear classifier to evaluate whether simple decision boundaries could separate cancerous and non-cancerous images.

Input Representation

Images normalized and flattened into vectors

Approximately 27,648 features per image

#### Results:

Accuracy: 0.585

Class 0 (Non-Cancer):

Precision: 0.60

Recall:    0.52

F1-score:  0.56

Class 1 (Cancer):

Precision: 0.58

Recall:    0.65

F1-score:  0.61

#### Conclusion:

Performance slightly above random guessing

Limited ability to capture spatial image patterns

Served strictly as a reference baseline

### Random Forest Classifier

Purpose:
Random Forest was selected to model non-linear relationships while remaining interpretable and robust to noise.

Model Characteristics:

Ensemble of decision trees

No manual feature engineering

Handles non-linear boundaries better than linear models

#### Results:

Accuracy: 0.715

Class 0 (Non-Cancer):
Precision: 0.69

Recall:    0.77

F1-score:  0.73

Class 1 (Cancer):
Precision: 0.74

Recall:    0.66

F1-score:  0.70


Additional Notes:

Number of input features: 27,648

Mean feature importance: 3.62e-05

#### Conclusion:

Significant improvement over Logistic Regression

Still constrained by flattened image representation

Motivated transition to convolutional models

### Convolutional Neural Networks (CNNs)

CNNs were implemented to learn spatial features directly from image data, avoiding information loss caused by flattening.

All CNN models used:

Binary Cross-Entropy loss

Adam optimizer

Fixed number of epochs (5)

Train / validation split

Same image preprocessing pipeline

### CNN Version 1 — Baseline CNN

Purpose:
Establish a deep learning baseline using a simple convolutional architecture without regularization.

Architecture Characteristics:

Shallow CNN

Limited number of convolution layers

No dropout

Minimal complexity

#### Training Results

Epoch	Train Loss	Validation Accuracy:

1	17.87	0.50

2	16.26	0.62

3	13.99	0.78

4	12.26	0.77

5	10.85	0.75

#### Conclusion

Rapid learning during early epochs

Highest observed validation accuracy (78%)

Slight performance drop after peak, indicating mild overfitting

Strong improvement over classical ML models

### CNN Version 2 — Dropout Regularization

Purpose:
Evaluate whether dropout regularization improves generalization and reduces overfitting.

Changes from CNN v1

Dropout layers added

Same base convolutional structure

##### Training Results:

Epoch	Train Loss	Validation Accuracy:

1	21.15	0.50

2	17.24	0.50

3	17.04	0.57

4	16.61	0.51

5	15.75	0.59

#### Conclusion:

Slower convergence

Validation accuracy remained unstable

Regularization was too strong for the dataset size

Underperformed compared to baseline CNN

### CNN Version 3 — Hyperparameter Tuning

Purpose:
Improve model stability through controlled hyperparameter tuning.

Tuning Adjustments

Learning rate modification

Architectural refinement

Improved training stability

#### Training Results:

Epoch	Train Loss	Validation Accuracy:

1	20.41	0.50

2	17.35	0.65

3	17.18	0.51

4	16.27	0.58

5	15.46	0.66

#### Conclusion:

More stable than CNN v2

Did not surpass CNN v1 peak performance

Demonstrated diminishing returns from added complexity

### Model Comparison Summary:
Model	Validation Accuracy:

Logistic Regression	0.585

Random Forest	0.715

CNN v1	0.780

CNN v2 (Dropout)	0.590

CNN v3 (Tuned)	0.660

#### Final Model Selection:

Selected Model: CNN Version 1

#### Justification:

Highest validation accuracy

Fast convergence

Stable training behavior

Simplest architecture among CNNs

Best performance-to-complexity tradeoff

#### Saved Model:

model_cnn_v1.pth

## 7. Training Execution


Option A: Notebook (Model_Training.ipynb)

Option B: Script-Based Training (python train.py)

Training-specific dependencies are listed in:
requirements-train.txt

## 8. Inference Pipeline

A Flask-based REST API was built to serve the trained model.

Endpoint

POST /predict

Input

Multipart form

Image file (.tif, .png, .jpg)

Output:
{
  "prediction": "non-cancer",
  
  "confidence": 0.6432
}

## 9. Clone Repository

git clone https://github.com/Oliverajovanovic90/histopathologic-cancer-detection.git

cd histopathologic-cancer-detection

## 10. Run Locally (Without Docker)

Create Virtual Environment:

python -m venv .venv

source .venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Start API:

python predict.py


API runs at:

http://localhost:5000

## 11. Run with Docker
Build Image:

docker build -t cancer-detection-api .

Run Container:

docker run -p 5000:5000 cancer-detection-api

## Test the API
curl -X POST http://localhost:5000/predict \
  -F "file=@data/images_subset/IMAGE_NAME.tif"


Example response:

{
  "prediction": "non-cancer",
  
  "confidence": 0.6432
}

## 12. Cloud Deployment (AWS)

To demonstrate production readiness, the trained model and inference API were prepared for deployment using AWS container services.


A cloud-specific inference script and Docker configuration were created to isolate deployment concerns from local experimentation: predict_cloud.py

Cloud-safe Flask inference service loading the final trained model (model_cnn_v1.pth) and exposing a /predict endpoint:Dockerfile.cloud

Lightweight container configuration optimized for inference-only workloads.

The Docker image was built locally and pushed to Amazon Elastic Container Registry (ECR). The service was then configured to run on Amazon Elastic Container Service (ECS) using a task definition with CPU-based execution and explicit port mapping (5000).

The deployed service was successfully tested via HTTP requests against the live endpoint, returning class predictions and confidence scores consistent with local inference results.

To prevent unnecessary cloud charges, all ECS tasks and related AWS resources were intentionally stopped after validation. The repository retains all deployment artifacts required to redeploy the service at any time.

#### Example Inference Endpoint:

When deployed, the service exposes an HTTP endpoint similar to:

POST http://<public-ip-or-load-balancer>:5000/predict


Example request:

curl -X POST http://<endpoint>/predict \
  -F "file=@sample_image.tif"


Example response:

{
  "prediction": "non-cancer",
  "confidence": 0.64
}


## 13. Conclusion

This project demonstrates an end-to-end machine learning workflow for automated histopathologic image classification, with the goal of distinguishing cancerous from non-cancerous tissue samples using digitized microscopy images.

The work began with careful data exploration and preparation, including controlled sub-sampling to address computational constraints while maintaining class balance. Early exploratory analysis validated label distributions, image characteristics, and confirmed the suitability of the dataset for supervised learning.

A progressive modeling strategy was intentionally adopted. Classical machine learning models were implemented first to establish baseline performance and validate the learning signal within the data. Logistic Regression provided a minimal reference point, while Random Forests demonstrated the benefit of non-linear decision boundaries but remained limited by flattened image representations.

The project then transitioned to convolutional neural networks, enabling direct learning from spatial image structure. Multiple CNN architectures were trained, evaluated, and compared. This iterative experimentation revealed that:

Convolutional models substantially outperform classical approaches for image-based tasks.

Increased architectural complexity does not necessarily yield better performance.

A well-designed, simpler CNN can provide superior accuracy, stability, and generalization.

The final selected model achieved strong validation performance while maintaining architectural simplicity and reproducibility. This balance is critical in real-world environments where interpretability, training efficiency, and deployment feasibility are as important as raw accuracy.

Beyond model training, the project also emphasized operational readiness, ensuring the final model could be saved, reloaded, and used consistently for inference. This positions the solution for integration into downstream systems such as diagnostic pipelines, clinical decision support tools, or batch image screening workflows.

Overall, this project delivers a robust and scalable foundation for automated histopathologic image classification. It demonstrates how disciplined data handling, incremental modeling, and evidence-based model selection can produce reliable results in applied machine learning settings. The approach and insights gained here can be extended to larger datasets, more advanced architectures, and real-time inference systems in medical imaging and beyond.
