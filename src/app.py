import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import math
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from pickle import dump
import utils as eda
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"

df = pd.read_csv(url , sep= ";")
st.dataframe(df)
target = 'label'

# Eliminar duplicados / verificacion

total_data = df.drop_duplicates()

st.write("Shape original:", df.shape)
st.write("Shape sin duplicados:", total_data.shape)
st.write("Duplicados eliminados:", df.duplicated().sum())
st.write("¿Hay duplicados en total_data?:", total_data.duplicated().any())
st.write(total_data.head(2))

total_data = total_data.copy()
total_data['label'] = (total_data['quality'] >= 6).astype(int) #creo variable label 

# X, y
X = total_data.drop(columns=['quality', 'label'])  # 
y = total_data['label']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN con k fijo (ajusta k si quieres)
k = 15
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Evaluación
y_pred = knn.predict(X_test_scaled)
st.write("accuracy:", accuracy_score(y_test, y_pred))
st.write("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
st.write("\nclassification_report:\n", classification_report(y_test, y_pred, target_names=["not_good", "good"]))

# Bucle de optimización de k
k_values = range(1, 20 + 1)
train_accuracies, test_accuracies = [], []

st.write("k\tTrain Acc\tTest Acc")
st.write("-" * 30)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    st.write(f"{k}\t{train_acc:.4f}\t\t{test_acc:.4f}")

# Mejor k
best_k = k_values[np.argmax(test_accuracies)]
st.write(f"\nMejor k: {best_k} | Mejor accuracy en test: {max(test_accuracies):.4f}")

# Entrenar y evaluar con mejor k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

st.write("\nMétricas con el mejor k:")
st.write("accuracy:", accuracy_score(y_test, y_pred))
st.write("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
st.write("\nclassification_report:\n", classification_report(y_test, y_pred, target_names=["not_good", "good"]))

fig = plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, 'o-', label='Train Accuracy', color='blue')
plt.plot(k_values, test_accuracies, 'o-', label='Test Accuracy', color='red')
plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
plt.xlabel('k (número de vecinos)')
plt.ylabel('Accuracy')
plt.title('Optimización de k en KNN (total_data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(list(k_values))
st.pyplot(fig)