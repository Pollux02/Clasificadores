import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

# Cargar datos del archivo CSV sin incluir la primera fila como encabezados
data = pd.read_csv("AutoInsurSweden.csv", header=None, names=["X", "Y"], skiprows=1)

# Separar características (X) y etiquetas (Y)
X = data["X"].values.reshape(-1, 1)
Y = data["Y"].values

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Inicializar los modelos
linear_regression_model = LinearRegression()
knn_model = KNeighborsRegressor(n_neighbors=5)
svm_model = SVR(kernel='linear')
neural_network_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)

# Entrenar los modelos
linear_regression_model.fit(X_train, Y_train)
knn_model.fit(X_train, Y_train)
svm_model.fit(X_train, Y_train)
neural_network_model.fit(X_train, Y_train)

# Predecir en el conjunto de prueba
linear_regression_pred = linear_regression_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
neural_network_pred = neural_network_model.predict(X_test)

# Función para calcular y mostrar métricas
def print_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

# Calcular y mostrar métricas para cada modelo
print_metrics("Linear Regression", Y_test, linear_regression_pred)
print_metrics("K-Nearest Neighbors", Y_test, knn_pred)
print_metrics("Support Vector Machine", Y_test, svm_pred)
print_metrics("Neural Network", Y_test, neural_network_pred)
