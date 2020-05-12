import numpy as np
import pandas as pd
# Si bien lo implementamos, no lo usaremos, ya que demostro no aportar a nuestro accuracy de prediccion
# El uso de numpy inspirado y su util metodo 'np.linalg.eig' lo saque de  https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
# como menciono en el informe, no usamos pca al final, ya que no aporto al accuracy

def PCA(data, n_componentes):
    # seleccionadas = [columnas] #las columnas que queremos usar para el pca. Data debe ser un DF de pandas

    # x = data.loc[:, seleccionadas].values  Sin label !!!!!
    # y = data.loc[:, [labels]].values

    # primero se saca la matriz de covarianza, usando la media y la desviacion estandar de los datos.
    # Luego sacamos los eigen vectors

    # Ordenamos los eigen vectors
    pares = list(zip(np.linalg.eig(np.cov((x - x.mean()) / (x.std()).T))[0],
                     np.linalg.eig(np.cov((x - x.mean()) / (x.std()).T))[1]))
    pares.sort(key=lambda vector: vector[0], reverse=True)
    vectores_seleccionados = []

    # sacamos los N primeros
    for component in range(columnas):
        vectores_seleccionados.append(pares[component][1])

    matriz_scatter = np.hstack(
        map(lambda vector: vectores_seleccionados.reshape(len(seleccionadas), 1), vectores_seleccionados))
    return pd.concat([pd.DataFrame(data=(x - x.mean()) / (x.std()).dot(matriz_scatter),
                                   columns=[f'PC{componente}' for componente in range(1, n_componentes + 1)]),
                      data[['char']]], axis=1)