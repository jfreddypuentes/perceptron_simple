# Perceptron
Programación del modelo más sencillo de red neuronal creado por Frank Rosenblatt.
La neurona artificial tiene un discrimidor o estimdor lineal.


### Tasa de aprendizaje
El modelo aprende con una tasa de aprendizaje de 0.1 en 7 iteraciones. A partir de la 6 ya converge.

### Obtención de datos
~~~
X, y = iris.get_train_target_data()
~~~

### Inicializacion del perceptron y entrenamiento
~~~
perceptron = Perceptron(eta=0.1, n_iter=7)
perceptron.fit(X, y)
~~~

### Dibujar las iterciones de entrenamiento y convergencia.
~~~
cp.plot_errors_diagrams(perceptron)
~~~

### Dibujar las regiones de clasificación.
~~~
cp.plot_desicion_regions(X, y, classifier=perceptron)
~~~
