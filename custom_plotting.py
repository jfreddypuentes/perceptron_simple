from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

DEBUG_ENABLED = True

def plot_errors_diagrams(classifier=None):
    ''' Pintar las epocas y los errores de clasificacion '''
    if classifier:
        plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o')
        plt.xlabel('Epocas')
        plt.ylabel('Numero de actualizaciones')
        plt.show()
    else:
        if DEBUG_ENABLED: print('ERROR: En plot_errors_diagrams(.) el parametros classifier es None')


def plot_desicion_regions(X, y, classifier, resolution=0.02):
    ''' Pintar las 2 regiones de clasificación del modelo '''
    if X is not None:
        if y is not None:
            if classifier:
                # Generador de colors.
                markers = ('s', 'x', 'o', '^', 'v')
                colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
                cmap = ListedColormap(colors[:len(np.unique(y))])

                # Representar la superficie de desicion.
                x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                    np.arange(x2_min, x2_max, resolution))
                Z = classifier.predict( np.array([xx1.ravel(), xx2.ravel()]).T )
                Z = Z.reshape(xx1.shape)

                plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
                plt.xlim(xx1.min(), xx1.max())
                plt.ylim(xx2.min(), xx2.max())

                # Representar muestras de clase.
                for idx, cl in enumerate(np.unique(y)):
                    plt.scatter(x=X[y == cl, 0],
                                y=X[y == cl, 1],
                                alpha=0.8,
                                c=colors[idx],
                                marker=markers[idx],
                                label=cl,
                                edgecolor='black')
                
                plt.xlabel('Longitud sepalo [cm]')
                plt.ylabel('Longitud petalo [cm]')
                plt.legend(loc='upper left')
                plt.show()
            else:
                if DEBUG_ENABLED: print('ERROR: En plot_desicion_regions(....) el parametro classifier es nulo')
        else:
            if DEBUG_ENABLED: print('ERROR: En plot_desicion_regions(....) el parametro y es nulo')    
    else:
        if DEBUG_ENABLED: print('ERROR: En plot_desicion_regions(....) el parametro X es nulo')