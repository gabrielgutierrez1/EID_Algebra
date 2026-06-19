"""Funciones auxiliares para visualizar resultados de PCA con Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np


def graficar_original(X, y, nombres, variables=None):
    """Grafica las dos primeras variables originales del dataset."""

    plt.figure(figsize=(8,5))
    etiqueta_x = variables[0] if variables else "Variable 1"
    etiqueta_y = variables[1] if variables else "Variable 2"

    for clase in np.unique(y):
        plt.scatter(
            X[y == clase, 0],
            X[y == clase, 1],
            label=nombres[clase]
        )

    plt.title("Datos Originales")
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()
    plt.grid()
    plt.show()


def graficar_pca(X_reducido, y, nombres):
    """Grafica la proyección final sobre PC1 y PC2."""

    plt.figure(figsize=(8,5))

    for clase in np.unique(y):
        plt.scatter(
            X_reducido[y == clase, 0],
            X_reducido[y == clase, 1],
            label=nombres[clase]
        )

    plt.title("Datos Reducidos con PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid()
    plt.show()


def varianza_explicada(varianza):
    """Muestra la varianza explicada por cada componente principal."""

    plt.figure(figsize=(8,5))

    plt.bar(
        range(1, len(varianza)+1),
        varianza
    )

    plt.title("Varianza Explicada")
    plt.xlabel("Componente")
    plt.ylabel("Varianza")
    plt.grid()
    plt.show()


def varianza_acumulada(varianza):
    """Muestra la suma acumulada de varianza explicada."""

    plt.figure(figsize=(8,5))
    componentes = range(1, len(varianza)+1)
    acumulada = np.cumsum(varianza)

    plt.plot(componentes, acumulada, marker="o")
    plt.title("Varianza Acumulada")
    plt.xlabel("Cantidad de componentes")
    plt.ylabel("Varianza acumulada")
    plt.ylim(0, 1.08)
    plt.grid()
    plt.show()
