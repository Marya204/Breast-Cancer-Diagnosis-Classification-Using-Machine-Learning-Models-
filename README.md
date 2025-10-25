# Breast-Cancer-Diagnosis-Classification-Using-Machine-Learning-Models-
A comprehensive machine learning project for classifying breast cancer tumors as benign or malignant using the Wisconsin Diagnostic Breast Cancer Dataset. This project implements and compares multiple classification algorithms to identify the most effective model for cancer diagnosis prediction.
## Project Overview
This project performs end-to-end machine learning analysis including data exploration, preprocessing, model training, hyperparameter optimization, and model evaluation to predict breast cancer diagnosis with high accuracy.
## Dataset
•	Source: Breast Cancer Wisconsin (Diagnostic) Dataset
•	Features: 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast mass
•	Target Variable: Diagnosis (Malignant/Benign)
•	Instances: 569 samples
## Technologies & Libraries
•	Python 3.x
•	Data Processing: Pandas, NumPy
•	Visualization: Matplotlib, Seaborn
•	Machine Learning: Scikit-learn
•	Deployment: Flask, Flask-ngrok
## Machine Learning Models
Four classification algorithms were implemented and compared:
1.	Logistic Regression - Best performing model
2.	K-Nearest Neighbors (KNN)
3.	Decision Tree Classifier
4.	Support Vector Machine (SVM)
   
## Methodology
1.	Data Exploration & Visualization 
o	Statistical analysis of features
o	Correlation heatmaps
o	Pairplot visualizations for feature relationships
2.	Data Preprocessing 
o	Missing value detection and handling
o	Outlier identification and removal
o	Feature normalization using MinMaxScaler
o	Label encoding for target variable
3.	Model Development 
o	Train-test split (80-20 ratio)
o	Model training on four different algorithms
o	Hyperparameter tuning using GridSearchCV
4.	Model Evaluation 
o	Accuracy score comparison
o	Classification reports (precision, recall, F1-score)
o	Feature importance analysis
5.	Deployment 
o	Flask web application for real-time predictions
o	REST API endpoint for diagnosis prediction
## Results
•	Logistic Regression: Highest accuracy and best overall performance
•	Successfully identified key features contributing to cancer diagnosis
•	Achieved reliable classification between benign and malignant tumors
## Usage
python
### Train models
python train_models.py

# Run Flask application
python app.py
## Key Features
•	Comprehensive data preprocessing pipeline
•	Multiple model comparison framework
•	Hyperparameter optimization
•	Feature importance visualization
•	Web-based prediction interface


