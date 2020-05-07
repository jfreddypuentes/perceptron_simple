from perceptron_simple import Perceptron
import iris_data as iris
import matplotlib.pyplot as plt
import custom_plotting as cp

# Obtencion de datos
X, y = iris.get_train_target_data()

# Inicializacion del perceptron y entrenamiento
perceptron = Perceptron(eta=0.1, n_iter=7)
perceptron.fit(X, y)

# Dibujo las iterciones de entrenamiento y convergencia.
cp.plot_errors_diagrams(perceptron)

# Dibujar las regiones de clasificacion.
cp.plot_desicion_regions(X, y, classifier=perceptron)
