# SkyView Aerial Landscape Classification

This repository contains implementations of four machine learning models for classifying aerial landscape images from the SkyView: An Aerial Landscape Dataset. The models include two deep learning approaches (ResNet-50 and EfficientNet-B0) and two traditional machine learning approaches (K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)). The project evaluates these models on both balanced and imbalanced (long-tail) versions of the dataset.
Table of Contents
•	Dataset
•	Models
•	Repository Structure
•	Installation
•	Usage
•	Results
•	Contributing
Dataset
The SkyView: An Aerial Landscape Dataset (available on Kaggle) contains 12,000 aerial images across 15 classes: Agriculture, Airport, Beach, City, Desert, Forest, Grassland, Highway, Lake, Mountain, Parking, Port, Railway, Residential, and River. Each class has 800 images in the balanced setting. For the imbalanced (long-tail) setting, class counts range from 800 to 50 images 
•	Balanced Dataset: 12,000 images (800 per class).
Four models are implemented to classify the SkyView dataset:
1.	ResNet-50 (resnet-final.ipynb):
o	Type: Deep learning (CNN).
o	Features: ResNet-50 (ImageNet), fine-tuned with custom fully connected layer.
o	Imbalance Handling: Weighted loss, weighted random sampling, aggressive augmentation.
o	Hardware: GPU (e.g., NVIDIA P100).
o	Performance (Imbalanced): Validation F1-score: 0.9684, Test Accuracy: ~96%.
2.	EfficientNet-B0 (aerial_longtail_version-2.ipynb):
o	Type: Deep learning (CNN).
o	Features: EfficientNet-B0 (ImageNet), fine-tuned with custom classifier.
o	Imbalance Handling: Relies on data augmentation (no explicit class weights/sampling).
o	Hardware: GPU (e.g., T4).
o	Performance (Imbalanced): Expected F1-score: ~0.95–0.98, Test Accuracy: ~95% (estimated).
3.	KNN (KNN.ipynb):
o	Type: Traditional machine learning (K-Nearest Neighbors).
o	Features: Hand-crafted Local Binary Pattern (LBP) features with PCA.
o	Imbalance Handling: None (SMOTE misapplied to balanced data).
o	Hardware: CPU.
o	Performance (Balanced): Test Accuracy: 0.4662 (not evaluated on imbalanced data).
4.	SVM (improved-svm.ipynb):
o	Type: Traditional machine learning (Support Vector Machine).
o	Features: Hand-crafted LBP features.
o	Imbalance Handling: Class weights.
o	Hardware: CPU.
o	Performance (Imbalanced): Test Accuracy: 0.6193, Weighted F1: 0.6243, Macro F1: 0.5356.
Repository Structure

![image](https://github.com/user-attachments/assets/8d683e45-e17b-4aee-a7a5-0b48bf299c15)


Installation
1.	Clone the Repository:
git clone https://github.com/<your-username>/skyview-classification.git
cd skyview-classification
2.	Install Dependencies:
Create a virtual environment and install the required packages:
o	python -m venv venv
o	source venv/bin/activate  # On Windows: venv\\Scripts\\activate
o	pip install -r requirements.txt
The requirements.txt includes:
torch
torchvision
scikit-learn
scikit-image
opencv-python
numpy
matplotlib
seaborn
tqdm
kagglehub
pandas
imblearn
pillow

Hardware Requirements:
o	ResNet-50 & EfficientNet: GPU (e.g., NVIDIA P100 or T4) recommended for faster training.
o	KNN & SVM: CPU sufficient.
o	Ensure sufficient RAM (~16GB) for processing 12,000 images.
Usage
1.	Download the Dataset:
The notebooks automatically download the SkyView dataset using kagglehub.dataset_download. Ensure your Kaggle API is configured.
2.	Run the Notebooks:
o	Open Jupyter Notebook:
o	jupyter notebook
o	Navigate to the notebooks/ directory and run the desired notebook:
	resnet-final.ipynb: Trains and evaluates ResNet-50 on balanced and imbalanced datasets.
	aerial_longtail_version-2.ipynb: Trains and evaluates EfficientNet-B0 on the dataset.
	KNN.ipynb: Trains and evaluates KNN on the dataset.
	improved-svm.ipynb: Trains and evaluates SVM on balanced and imbalanced datasets.
3.	Outputs:
o	ResNet-50 & EfficientNet: Training loss, classification reports, confusion matrices, and precision/recall/F1 plots per epoch.
o	KNN: Training/test accuracy, confusion matrix.
o	SVM: Accuracy, weighted/macro F1-scores, classification report, confusion matrix.
o	Outputs (e.g., plots) are displayed inline or saved (e.g., knn_confusion_matrix.png).

Results
•	ResNet-50 outperforms others due to robust imbalance handling (weighted loss, sampling).
•	EfficientNet is efficient and likely comparable but may struggle with tail classes without class weights.
•	SVM performs moderately but fails on tail classes.
•	KNN is the least effective, with severe overfitting.
Contributing
Contributions are welcome! To contribute:
1.	Fork the repository.
2.	Create a new branch: git checkout -b feature/your-feature.
3.	Make changes and commit: git commit -m "Add your feature".
4.	Push to your branch: git push origin feature/your-feature.
5.	Open a pull request with a detailed description.
