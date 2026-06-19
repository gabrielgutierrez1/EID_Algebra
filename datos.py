"""Carga y prepara el dataset utilizado por el proyecto."""

from sklearn.datasets import load_iris


def cargar_datos():
    """Carga Iris y devuelve matriz de datos, clases y nombres descriptivos."""
    iris = load_iris()

    # X contiene las mediciones numericas; y contiene la especie de cada muestra.
    X = iris.data
    y = iris.target
    nombres = iris.target_names
    variables = iris.feature_names

    return X, y, nombres, variables
