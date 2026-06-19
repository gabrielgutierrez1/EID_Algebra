# Proyecto 3 - Visualización y Reducción Dimensional utilizando PCA

## Integrantes

- Gabriel Gutierrez
- Ailyn Melillan
- Nombre Integrante 3
- Nombre Integrante 4

## Descripción

Este proyecto implementa el algoritmo de Análisis de Componentes Principales (PCA, Principal Component Analysis) en Python para reducir la dimensionalidad del dataset Iris.

El objetivo es transformar datos de 4 variables originales a 2 componentes principales, conservando la mayor cantidad posible de información. Para ello se utilizan conceptos de Álgebra Lineal como matrices, covarianza, autovalores, autovectores y proyecciones vectoriales.

La implementación del PCA se realiza manualmente con NumPy. Scikit-Learn se utiliza solo para cargar el dataset Iris, no para aplicar PCA.

## Requerimientos

- Python 3.10 o superior
- Tkinter, incluido normalmente con Python. En Ubuntu puede instalarse con `sudo apt install python3-tk`

### Dependencias de Python

El proyecto utiliza las siguientes librerías:

| Librería | Uso dentro del proyecto |
| --- | --- |
| `numpy` | Operaciones matriciales, medias, covarianza, autovalores, autovectores y proyecciones. |
| `matplotlib` | Creación de gráficos de datos originales, PCA, varianza explicada y varianza acumulada. |
| `scikit-learn` | Carga del dataset Iris mediante `load_iris()`. No se usa `sklearn.PCA`. |
| `tkinter` | Construcción de la interfaz gráfica. |

Para instalar las dependencias de Python:

```bash
pip install numpy matplotlib scikit-learn
```

En algunos sistemas Linux, Tkinter se instala desde el gestor de paquetes del sistema:

```bash
sudo apt install python3-tk
```

## Estructura del Proyecto

```text
PCA-Proyecto/
│
├── main.py
├── datos.py
├── pca_manual.py
├── interfaz.py
├── visualizacion.py
├── analisis.py
└── readme.md
```

| Archivo | Función |
| --- | --- |
| `main.py` | Punto de entrada del programa. Ejecuta la interfaz gráfica. |
| `datos.py` | Carga el dataset Iris desde Scikit-Learn y entrega datos, clases y nombres de variables. |
| `pca_manual.py` | Contiene la implementación matemática del PCA usando NumPy. |
| `interfaz.py` | Construye la interfaz Tkinter, muestra estadísticas, matrices, interpretación y gráficos. |
| `visualizacion.py` | Funciones auxiliares para generar gráficos con Matplotlib. |
| `analisis.py` | Funciones auxiliares para mostrar estadísticas descriptivas por consola. |

## Ejecución

Desde la carpeta del proyecto:

```bash
python3 main.py
```

Al ejecutar el programa se abre una interfaz gráfica con pestañas para revisar el resumen estadístico, la interpretación de resultados, las matrices usadas en PCA y los gráficos.

## Explicación del Código

### 1. Carga de datos

El archivo `datos.py` usa `load_iris()` para obtener:

- `X`: matriz de datos con 150 filas y 4 columnas.
- `y`: clase de cada muestra.
- `nombres`: nombres de las especies.
- `variables`: nombres de las variables originales.

En la matriz `X`, cada fila representa una flor y cada columna representa una medición: largo/ancho del sépalo y largo/ancho del pétalo.

### 2. Implementación manual de PCA

El archivo `pca_manual.py` realiza el flujo matemático:

1. **Centrar los datos:** se resta la media de cada variable.
2. **Calcular covarianza:** se obtiene la matriz que describe cómo varían las variables entre sí.
3. **Obtener autovalores y autovectores:** se calculan sobre la matriz de covarianza.
4. **Ordenar componentes:** los autovalores se ordenan de mayor a menor para identificar las direcciones de mayor varianza.
5. **Proyectar datos:** se multiplican los datos centrados por los dos autovectores principales.
6. **Calcular varianza explicada:** cada autovalor se divide por la suma total de autovalores.

No se utiliza `sklearn.decomposition.PCA`; la reducción se hace directamente con operaciones matriciales de NumPy.

### 3. Interfaz gráfica

El archivo `interfaz.py` construye una aplicación Tkinter con cuatro pestañas:

- **Resumen:** muestra dimensiones, variables, estadísticas básicas, autovalores, autovectores y muestra de datos proyectados.
- **Interpretación:** explica el resultado obtenido y la información conservada.
- **Matrices:** muestra datos centrados, matriz de covarianza y autovectores.
- **Gráficos:** visualiza datos originales, datos reducidos con PCA, varianza explicada y varianza acumulada.

### 4. Visualización

Los gráficos permiten comparar el comportamiento de los datos antes y después del PCA:

- El gráfico original usa las dos primeras variables del dataset para una visualización 2D inicial.
- El gráfico PCA muestra la proyección usando PC1 y PC2.
- El gráfico de varianza explicada muestra cuánto aporta cada componente.
- El gráfico de varianza acumulada muestra cuánta información se conserva al sumar componentes.

## Resultados

Con el dataset Iris, los dos primeros componentes principales conservan aproximadamente:

```text
PC1: 92.46 %
PC2: 5.31 %
PC1 + PC2: 97.77 %
```

Esto significa que al reducir los datos de 4 dimensiones a 2 dimensiones se mantiene cerca del 97.77 % de la varianza total del conjunto de datos.

## Fundamento Matemático

PCA busca nuevas direcciones en el espacio de datos que maximicen la varianza. Estas direcciones son los autovectores de la matriz de covarianza, mientras que los autovalores indican cuánta información captura cada componente principal.

Al proyectar los datos centrados sobre los autovectores principales, se obtiene una representación de menor dimensión que conserva la mayor parte de la estructura original.

## Bibliografía

- Jolliffe, I. T. (2002). Principal Component Analysis.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- Documentación oficial de NumPy.
- Documentación oficial de Scikit-Learn.
- Documentación oficial de Matplotlib.

## Curso

Álgebra Lineal para la Computación

Proyecto de Investigación N°3

Visualización y Reducción Dimensional de Datos utilizando PCA
