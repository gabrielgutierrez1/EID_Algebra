"""Funciones simples de análisis descriptivo por consola."""

import numpy as np


def mostrar_estadisticas(X):
    """Imprime dimensiones y valores básicos de cada variable."""
    print("Dimensiones:", X.shape)

    print("\nPromedios:")
    print(np.mean(X, axis=0))

    print("\nMínimos:")
    print(np.min(X, axis=0))

    print("\nMáximos:")
    print(np.max(X, axis=0))
