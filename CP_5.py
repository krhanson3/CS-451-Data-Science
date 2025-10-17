#CS 451/551
#Coding Practice Session 5
#Hanson, Kaitlyn 
#krhanson3@crimson.ua.edu

#Python for Machine Learning

#1 Data-Loading and Exploration
import pandas as pd 
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

X = df[iris.feature_names]
Y = df['species']


print("shape of features:", X.shape)
print("shape of target:", Y.shape)
print("feature names:", iris.feature_names)
print("target names:", iris.target_names)

print("top 12:", X[:12])

#2 Train Test Split 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

X = df[iris.feature_names]
Y = df['species']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y)

print("Train set shape:", X_train.shape, Y_train.shape)
print("Test set shape:", X_test.shape, Y_test.shape)


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie(np.bincount(Y_train), labels=iris.target_names, autopct='%1.1f%%')
axes[0].set_title("Train Set Distribution")
axes[1].pie(np.bincount(Y_test), labels=iris.target_names, autopct='%1.1f%%')
axes[1].set_title("Test Set Distribution")
plt.show()