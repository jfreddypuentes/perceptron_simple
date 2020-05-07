import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def download_data():
    # Los datos iris.dat están en la misma ruta.
    df = pd.read_csv('iris.data', header=None)
    return df


def plot_data(df):
    # Seleccionar setosa y versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extraer longitud de sepalo y longitud de petalo 
    X = df.iloc[0:100, [0,2]].values

    # Plot
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('Longitud sepalo [cm]')
    plt.ylabel('Longitud petalo [cm]')
    plt.legend(loc='upper left')
    plt.show()


def get_train_target_data():
    df = download_data()

    # Seleccionar setosa y versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extraer longitud de sepalo y longitud de petalo 
    X = df.iloc[0:100, [0,2]].values
    return (X, y)

