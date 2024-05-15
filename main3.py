import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar los datos desde un archivo CSV
data = pd.read_csv('winequality-white.csv', delimiter=';')

# Mostrar las primeras filas para verificar la carga correcta
print(data.head())

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar los clasificadores
rf_classifier = RandomForestClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42)
knn_classifier = KNeighborsClassifier()
svm_classifier = SVC(random_state=42)
nb_classifier = GaussianNB()

# Entrenar los clasificadores
rf_classifier.fit(X_train_scaled, y_train)
lr_classifier.fit(X_train_scaled, y_train)
knn_classifier.fit(X_train_scaled, y_train)
svm_classifier.fit(X_train_scaled, y_train)
nb_classifier.fit(X_train_scaled, y_train)

# Predecir en los datos de prueba
rf_pred = rf_classifier.predict(X_test_scaled)
lr_pred = lr_classifier.predict(X_test_scaled)
knn_pred = knn_classifier.predict(X_test_scaled)
svm_pred = svm_classifier.predict(X_test_scaled)
nb_pred = nb_classifier.predict(X_test_scaled)

# Calcular métricas
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)

rf_precision = precision_score(y_test, rf_pred, average='weighted')
lr_precision = precision_score(y_test, lr_pred, average='weighted')
knn_precision = precision_score(y_test, knn_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted')
nb_precision = precision_score(y_test, nb_pred, average='weighted')

rf_recall = recall_score(y_test, rf_pred, average='weighted')
lr_recall = recall_score(y_test, lr_pred, average='weighted')
knn_recall = recall_score(y_test, knn_pred, average='weighted')
svm_recall = recall_score(y_test, svm_pred, average='weighted')
nb_recall = recall_score(y_test, nb_pred, average='weighted')

rf_f1 = f1_score(y_test, rf_pred, average='weighted')
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
nb_f1 = f1_score(y_test, nb_pred, average='weighted')

# Imprimir métricas
print("Random Forest:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print()
print("Logistic Regression:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1 Score:", lr_f1)
print()
print("K-Nearest Neighbors:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)
print()
print("Support Vector Machines:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)
print()
print("Naive Bayes:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
