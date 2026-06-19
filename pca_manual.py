"""Implementación matemática del PCA usando operaciones de NumPy."""

import numpy as np


def centrar_datos(X):
    """Resta a cada variable su media para ubicar los datos alrededor del origen."""
    # axis=0 calcula la media por columna, es decir, por cada variable.
    media = np.mean(X, axis=0)
    return X - media


def matriz_covarianza(X):
    """Calcula la matriz de covarianza usando las variables como columnas."""
    # np.cov espera variables en filas; por eso se transpone X.
    return np.cov(X.T)


def autovalores_autovectores(cov):
    """Obtiene y ordena autovalores y autovectores de mayor a menor varianza."""
    # La matriz de covarianza es simetrica; eigh es estable para este caso.
    autovalores, autovectores = np.linalg.eigh(cov)
    # Los mayores autovalores representan los componentes mas importantes.
    indices = np.argsort(autovalores)[::-1]
    return autovalores[indices], autovectores[:, indices]


def varianza_explicada(autovalores):
    """Convierte cada autovalor en su porcentaje relativo de varianza explicada."""
    # La suma total de autovalores representa la varianza total del dataset.
    return autovalores / np.sum(autovalores)


def proyectar_datos(X_centrado, autovectores, n_componentes=2):
    """Proyecta los datos centrados sobre los primeros componentes principales."""
    # Cada columna seleccionada corresponde a una direccion principal.
    componentes = autovectores[:, :n_componentes]
    return X_centrado @ componentes


def aplicar_pca(X, n_componentes=2):
    """Ejecuta el flujo completo de PCA y devuelve los resultados principales."""
    # Flujo del algoritmo: centrar, medir covarianza, obtener componentes y proyectar.
    X_centrado = centrar_datos(X)
    covarianza = matriz_covarianza(X_centrado)
    autovalores, autovectores = autovalores_autovectores(covarianza)
    X_reducido = proyectar_datos(X_centrado, autovectores, n_componentes)
    varianza = varianza_explicada(autovalores)

    return X_centrado, covarianza, autovalores, autovectores, X_reducido, varianza
