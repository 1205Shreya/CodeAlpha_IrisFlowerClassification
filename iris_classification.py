# =========================================
# IMPORT LIBRARIES
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================
# LOAD DATASET
# =========================================

# Read dataset from CSV file
df = pd.read_csv("dataset/iris.csv")

# =========================================
# DATASET INFORMATION
# =========================================

# Show first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset information
print("\nDATASET INFO:")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# =========================================
# EXPLORATORY DATA ANALYSIS
# =========================================

# Shape of dataset
print("\nDATASET SHAPE:")
print(df.shape)

# Column names
print("\nCOLUMN NAMES:")
print(df.columns)

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Count flower species
print("\nFLOWER SPECIES COUNT:")
print(df['Species'].value_counts())

# =========================================
# DATA VISUALIZATION
# =========================================

# Countplot of flower species
sns.countplot(x='Species', data=df)

plt.title("Count of Iris Flower Species")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()

plt.show(block=True)

# Pairplot visualization
sns.pairplot(df, hue='Species')

plt.show()

# =========================================
# DATA PREPROCESSING
# =========================================

# Remove unnecessary Id column
df = df.drop(columns=['Id'])

# Features (input data)
X = df.drop('Species', axis=1)

# Target (output data)
y = df['Species']

print("\nFEATURES:")
print(X.head())

print("\nTARGET:")
print(y.head())

# =========================================
# TRAIN TEST SPLIT
# =========================================

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SHAPE:")
print(X_train.shape)

print("\nTESTING DATA SHAPE:")
print(X_test.shape)

# =========================================
# MODEL TRAINING
# ========================================= 

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# =========================================
# MODEL PREDICTION
# =========================================

# Make predictions
y_pred = model.predict(X_test)

print("\nPREDICTED VALUES:")
print(y_pred)

# Compare actual vs predicted values
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nACTUAL VS PREDICTED:")
print(comparison.head(10))

# =========================================
# MODEL EVALUATION
# =========================================

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy)

# Classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# =========================================
# CONFUSION MATRIX
# =========================================

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()