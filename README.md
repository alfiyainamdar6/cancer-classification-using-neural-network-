# cancer-classification-using-neural-network



📌 Project Overview

This project uses a Neural Network (Multi-Layer Perceptron / MLP) to classify breast cancer tumors into two categories:

🔴 Malignant – cancerous

🟢 Benign – non-cancerous


The project uses the Breast Cancer Wisconsin (Diagnostic) dataset available through Scikit-learn. The dataset contains 569 samples, 30 numerical features, and 2 target classes.


⚠️ Note: This is an educational machine-learning project and is not a medical diagnosis system.


🎯 Objectives

The main objectives of this project are:

Understand a binary classification problem.

Perform data preprocessing.

Split the dataset into training and testing sets.

Build a Neural Network classification model.

Train the model using training data.

Evaluate model performance.

Predict whether a tumor is malignant or benign.

Understand how Neural Networks can be applied to healthcare datasets.



📊 Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset


The dataset is provided by Scikit-learn's load_breast_cancer() function.

Property	Value
Total Samples	569
Number of Features	30
Number of Classes	2
Malignant Samples	212
Benign Samples	357


The 30 features are numerical measurements derived from digitized images of breast mass cell nuclei.

Target Classes

0 → Malignant

1 → Benign


🧠 What is a Neural Network?

A Neural Network is a machine-learning model inspired by the structure of biological neural systems.

A simple neural network contains:


Input Layer

     ↓
Hidden Layer(s)

     ↓
Output Layer



For this project:


30 Input Features

       ↓
Hidden Layer

       ↓
Hidden Layer
       ↓
       
   Output
       ↓
       
Malignant / Benign


The Scikit-learn MLPClassifier implements a Multi-Layer Perceptron classifier and trains using backpropagation.


🛠️ Technologies Used

🐍 Python

📊 NumPy

🐼 Pandas

📈 Matplotlib

🤖 Scikit-learn

📓 Jupyter Notebook

💻 Git & GitHub



📦 Libraries Installation


Install the required libraries using:

pip install numpy pandas matplotlib scikit-learn jupyter

🔄 Project Workflow

Dataset

   ↓
Data Loading

   ↓
Data Understanding
   ↓
   
Data Preprocessing

   ↓
Train-Test Split
   ↓
   
Feature Scaling

   ↓
Neural Network Model

   ↓
Model Training

   ↓
Prediction

   ↓
Model Evaluation


🧪 Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


📥 Step 2: Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Features:", X.shape)
print("Target:", y.shape)

Expected shape:

Features: (569, 30)
Target: (569,)
✂️ Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Here:

80% → Training data
20% → Testing data


📏 Step 4: Feature Scaling

Neural Networks generally benefit from appropriately scaled input features.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

The scaler is fitted only on the training data and then applied to the test data.



🤖 Step 5: Create Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
Parameters
Parameter	Meaning
hidden_layer_sizes	Number of neurons in hidden layers
activation='relu'	Activation function
solver='adam'	Optimization algorithm
max_iter=500	Maximum training iterations
random_state=42	Reproducible results

MLPClassifier supports activation functions such as ReLU and optimization methods including Adam and SGD.


🏋️ Step 6: Train the Model
model.fit(X_train, y_train)

During training, the Neural Network learns patterns from the training data.



🔮 Step 7: Make Predictions
y_pred = model.predict(X_test)

print(y_pred)

The model predicts one of the two classes:

0 → Malignant
1 → Benign


📊 Step 8: Model Evaluation
Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
Classification Report
print(classification_report(y_test, y_pred))
Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

The classification report can provide metrics such as:

Precision
Recall
F1-score
Support

📉 Confusion Matrix Visualization
import seaborn as sns

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

📈 Model Architecture
              INPUT
                │
        30 Cancer Features
                │
                ▼
       ┌─────────────────┐
       │ Hidden Layer 1   │
       │   64 Neurons     │
       │      ReLU        │
       └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │ Hidden Layer 2   │
       │   32 Neurons     │
       │      ReLU        │
       └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │ Output Layer     │
       │ Binary Class     │
       └─────────────────┘
                │
          ┌─────┴─────┐
          ▼           ▼
      Malignant     Benign

          
📁 Project Folder Structure
Cancer-Classification-Neural-Network/
│
├── 📓 cancer_classification.ipynb
│
├── 📄 README.md
│
├── 📊 requirements.txt
│
├── 📁 images/
│   ├── confusion_matrix.png
│   └── model_results.png
│
└── 📁 models/
    └── neural_network_model.pkl


        
📋 requirements.txt

Create a file named:

requirements.txt


Add:

numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter


💡 Key Concepts Learned


Through this project, I learned:

Machine Learning classification
Binary classification
Dataset exploration
Train-test splitting
Feature scaling
Neural Networks
Multi-Layer Perceptron
ReLU activation
Adam optimizer
Model training
Prediction
Confusion Matrix
Accuracy
Precision
Recall
F1-score
GitHub project organization


🚀 Future Improvements

This project can be improved by:

Comparing Neural Network with Logistic Regression.
Comparing Neural Network with Random Forest.
Hyperparameter tuning.
Adding cross-validation.
Creating a Streamlit web application.
Adding interactive prediction.
Saving and loading the trained model.
Creating a better visualization dashboard.


⚠️ Disclaimer

This project is created for educational and machine-learning practice purposes only.

It should not be used for medical diagnosis, treatment, or clinical decision-making. Real-world medical systems require clinically validated data, appropriate testing, regulatory review, and qualified healthcare professionals.


👩‍💻 Author

Alfiya Inamdar

Artificial Intelligence & Data Science Student


Skills Used

Python

• Machine Learning

• Neural Networks

• Scikit-learn

• Data Science


⭐ If You Like This Project

If this project helped you understand Neural Networks and Machine Learning, consider giving the repository a  on GitHub.


📚 References

Scikit-learn Breast Cancer Dataset documentation.
Scikit-learn Neural Network / MLP documentation.
