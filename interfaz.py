"""Interfaz gráfica para explorar el PCA aplicado al dataset Iris."""

import tkinter as tk
from tkinter import ttk

import matplotlib

# Permite incrustar graficos de Matplotlib dentro de una ventana Tkinter.
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datos import cargar_datos
from pca_manual import aplicar_pca

TAMANO_FIGURA = (9.6, 3.5)
DPI_FIGURA = 110


def formatear_matriz(matriz, precision=4):
    """Convierte arreglos NumPy a texto legible para los paneles de la interfaz."""
    return np.array2string(matriz, precision=precision, suppress_small=True)


def formatear_variables(variables):
    """Enumera los nombres de variables originales del dataset."""
    return "\n".join(f"{indice}. {variable}" for indice, variable in enumerate(variables, start=1))


def formatear_porcentajes(valores):
    """Formatea valores decimales como porcentajes por componente principal."""
    return "\n".join(f"PC{indice}: {valor:.2%}" for indice, valor in enumerate(valores, start=1))


def crear_figura_original(X, y, nombres, variables):
    """Crea el gráfico 2D inicial usando las dos primeras variables originales."""
    figura, eje = plt.subplots(figsize=TAMANO_FIGURA, dpi=DPI_FIGURA)
    colores = ["#2F6DF6", "#F97316", "#10B981"]

    for clase in np.unique(y):
        # Se filtran las filas de cada especie para dibujarlas con distinto color.
        eje.scatter(
            X[y == clase, 0],
            X[y == clase, 1],
            label=nombres[clase],
            s=55,
            alpha=0.9,
            color=colores[int(clase) % len(colores)],
            edgecolors="white",
            linewidths=0.6,
        )

    eje.set_title("Datos Originales", fontsize=16, fontweight="bold", pad=14)
    eje.set_xlabel(variables[0], fontsize=12)
    eje.set_ylabel(variables[1], fontsize=12)
    eje.grid(True, alpha=0.25)
    eje.legend(frameon=False)
    figura.tight_layout(pad=2.0)
    return figura


def crear_figura_pca(X_reducido, y, nombres):
    """Crea el gráfico 2D de los datos proyectados sobre PC1 y PC2."""
    figura, eje = plt.subplots(figsize=TAMANO_FIGURA, dpi=DPI_FIGURA)
    colores = ["#2F6DF6", "#F97316", "#10B981"]

    for clase in np.unique(y):
        # X_reducido ya contiene las coordenadas nuevas calculadas por PCA.
        eje.scatter(
            X_reducido[y == clase, 0],
            X_reducido[y == clase, 1],
            label=nombres[clase],
            s=55,
            alpha=0.9,
            color=colores[int(clase) % len(colores)],
            edgecolors="white",
            linewidths=0.6,
        )

    eje.set_title("Datos Reducidos con PCA", fontsize=16, fontweight="bold", pad=14)
    eje.set_xlabel("PC1", fontsize=12)
    eje.set_ylabel("PC2", fontsize=12)
    # Los ejes en cero ayudan a ubicar el origen de la proyeccion.
    eje.axhline(0, color="#94A3B8", linewidth=0.8, alpha=0.45)
    eje.axvline(0, color="#94A3B8", linewidth=0.8, alpha=0.45)
    eje.grid(True, alpha=0.25)
    eje.legend(frameon=False)
    figura.tight_layout(pad=2.0)
    return figura


def crear_figura_varianza(varianza):
    """Crea un gráfico de barras con la varianza explicada por componente."""
    figura, eje = plt.subplots(figsize=TAMANO_FIGURA, dpi=DPI_FIGURA)
    # Componentes se numera desde 1 para que coincida con PC1, PC2, etc.
    componentes = np.arange(1, len(varianza) + 1)
    barra = eje.bar(
        componentes,
        varianza,
        color=["#2F6DF6", "#F97316", "#10B981", "#8B5CF6"],
    )

    for rect, valor in zip(barra, varianza):
        eje.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.01,
            f"{valor:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    eje.set_title("Varianza Explicada", fontsize=16, fontweight="bold", pad=14)
    eje.set_xlabel("Componente", fontsize=12)
    eje.set_ylabel("Varianza relativa", fontsize=12)
    eje.set_ylim(0, max(varianza) * 1.25)
    eje.grid(True, axis="y", alpha=0.25)
    figura.tight_layout(pad=2.0)
    return figura


def crear_figura_varianza_acumulada(varianza):
    """Crea un gráfico de línea con la varianza acumulada por componentes."""
    figura, eje = plt.subplots(figsize=TAMANO_FIGURA, dpi=DPI_FIGURA)
    componentes = np.arange(1, len(varianza) + 1)
    # La suma acumulada muestra cuanta informacion se conserva al agregar componentes.
    acumulada = np.cumsum(varianza)

    eje.plot(componentes, acumulada, marker="o", color="#2F6DF6", linewidth=2.4)
    eje.fill_between(componentes, acumulada, color="#93C5FD", alpha=0.25)
    eje.axvline(2, color="#F97316", linewidth=1.4, linestyle="--", alpha=0.85)
    eje.axhline(acumulada[1], color="#F97316", linewidth=1.4, linestyle="--", alpha=0.85)

    for x, valor in zip(componentes, acumulada):
        eje.text(x, valor + 0.025, f"{valor:.2%}", ha="center", va="bottom", fontsize=10)

    eje.text(
        2.06,
        acumulada[1] - 0.07,
        f"PC1 + PC2 = {acumulada[1]:.2%}",
        color="#C2410C",
        fontsize=10,
        fontweight="bold",
    )

    eje.set_title("Varianza Acumulada", fontsize=16, fontweight="bold", pad=14)
    eje.set_xlabel("Cantidad de componentes", fontsize=12)
    eje.set_ylabel("Varianza acumulada", fontsize=12)
    eje.set_xticks(componentes)
    eje.set_ylim(0, 1.08)
    eje.grid(True, alpha=0.25)
    figura.tight_layout(pad=2.0)
    return figura


def crear_resumen(X, cov, autovalores, autovectores, varianza, variables):
    """Agrupa estadísticas y resultados del PCA para mostrarlos en la interfaz."""
    # Estas estadisticas permiten hacer un analisis preliminar del dataset.
    media = np.mean(X, axis=0)
    minimo = np.min(X, axis=0)
    maximo = np.max(X, axis=0)

    return {
        "dimensiones": X.shape,
        "media": media,
        "minimo": minimo,
        "maximo": maximo,
        "covarianza": cov,
        "autovalores": autovalores,
        "autovectores": autovectores,
        "varianza": varianza,
        "varianza_acumulada": np.cumsum(varianza),
        "variables": variables,
    }


def varianza_acumulada(varianza, n_componentes=2):
    """Calcula la varianza conservada al usar los primeros componentes."""
    return np.sum(varianza[:n_componentes])


def poblar_texto(widget, contenido):
    """Inserta texto en un widget Text y lo deja en modo solo lectura."""
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert(tk.END, contenido)
    widget.configure(state="disabled")


def crear_texto_interpretacion(resumen):
    """Genera una explicación breve de los resultados para la pestaña de análisis."""
    acumulada_2d = varianza_acumulada(resumen["varianza"])
    pc1 = resumen["varianza"][0]
    pc2 = resumen["varianza"][1]

    return (
        "Interpretación del resultado\n\n"
        f"El dataset Iris tiene {resumen['dimensiones'][0]} muestras y "
        f"{resumen['dimensiones'][1]} variables numéricas originales.\n\n"
        f"PC1 explica {pc1:.2%} de la varianza y PC2 explica {pc2:.2%}. "
        f"Al proyectar los datos a dos dimensiones se conserva {acumulada_2d:.2%} "
        "de la información total medida como varianza.\n\n"
        "Esto indica que la reducción de 4 variables a 2 componentes mantiene "
        "la mayor parte de la estructura del conjunto de datos. En el gráfico PCA "
        "se espera observar separación clara de la clase setosa y una separación "
        "parcial entre versicolor y virginica.\n\n"
        "El gráfico original usa solo las dos primeras variables para poder "
        "visualizar datos en 2D antes de PCA; la proyección PCA, en cambio, "
        "combina las cuatro variables originales."
    )


class AppPCA(tk.Tk):
    """Ventana principal de la aplicación PCA."""

    def __init__(self):
        super().__init__()
        self.title("PCA Interactivo - Iris")
        self.geometry("1240x860")
        self.minsize(1100, 760)
        self.configure(bg="#0F172A")
        # Se controla el cierre para liberar tambien los graficos de Matplotlib.
        self.protocol("WM_DELETE_WINDOW", self._cerrar_aplicacion)
        self.figuras = []
        self.canvases = []

        self._configurar_estilos()
        self._cargar_datos_y_calcular()
        self._crear_interfaz()

    def _cerrar_aplicacion(self):
        """Libera recursos de Tkinter y Matplotlib al cerrar la ventana."""
        for canvas in self.canvases:
            try:
                canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass

        for figura in self.figuras:
            plt.close(figura)

        self.canvases.clear()
        self.figuras.clear()
        self.quit()
        self.destroy()

    def _configurar_estilos(self):
        """Define colores, fuentes y estilos visuales de la interfaz."""
        estilo = ttk.Style(self)
        estilo.theme_use("clam")
        estilo.configure("TFrame", background="#0F172A")
        estilo.configure("Card.TFrame", background="#111827", relief="flat")
        estilo.configure("Header.TFrame", background="#0B1220")
        estilo.configure("TLabel", background="#0F172A", foreground="#E5E7EB", font=("Sans", 11))
        estilo.configure(
            "Title.TLabel",
            background="#0B1220",
            foreground="#F8FAFC",
            font=("Sans", 22, "bold"),
        )
        estilo.configure(
            "Subtitle.TLabel",
            background="#0B1220",
            foreground="#94A3B8",
            font=("Sans", 11),
        )
        estilo.configure(
            "Accent.TLabel",
            background="#111827",
            foreground="#93C5FD",
            font=("Sans", 12, "bold"),
        )
        estilo.configure(
            "Metric.TLabel",
            background="#111827",
            foreground="#F8FAFC",
            font=("Sans", 18, "bold"),
        )
        estilo.configure(
            "MetricValue.TLabel",
            background="#111827",
            foreground="#CBD5E1",
            font=("Sans", 10),
        )
        estilo.configure("TNotebook", background="#0F172A", borderwidth=0)
        estilo.configure("TNotebook.Tab", padding=(18, 10), font=("Sans", 10, "bold"))
        estilo.map("TNotebook.Tab", background=[("selected", "#111827")], foreground=[("selected", "#F8FAFC")])

    def _cargar_datos_y_calcular(self):
        """Carga Iris y calcula todos los resultados necesarios para la app."""
        self.X, self.y, self.nombres, self.variables = cargar_datos()
        # aplicar_pca concentra el procedimiento matematico del proyecto.
        (
            self.X_centrado,
            self.covarianza,
            self.autovalores,
            self.autovectores,
            self.X_reducido,
            self.varianza,
        ) = aplicar_pca(self.X, n_componentes=2)
        self.resumen = crear_resumen(
            self.X,
            self.covarianza,
            self.autovalores,
            self.autovectores,
            self.varianza,
            self.variables,
        )

    def _crear_interfaz(self):
        """Construye encabezado, tarjetas de indicadores y pestañas principales."""
        contenedor = ttk.Frame(self, padding=18)
        contenedor.pack(fill="both", expand=True)

        # Encabezado principal de la aplicacion.
        encabezado = ttk.Frame(contenedor, style="Header.TFrame", padding=(22, 18))
        encabezado.pack(fill="x")

        ttk.Label(encabezado, text="Análisis PCA del dataset Iris", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            encabezado,
            text="Resumen estadístico, matriz de covarianza, autovalores y visualizaciones interactivas.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        indicadores = ttk.Frame(contenedor)
        indicadores.pack(fill="x", pady=(16, 14))

        # Tarjetas con los resultados mas importantes para leer rapidamente.
        self._crear_tarjeta(indicadores, "Muestras", f"{self.resumen['dimensiones'][0]}", "150 filas en Iris")
        self._crear_tarjeta(indicadores, "Variables", f"{self.resumen['dimensiones'][1]}", "4 columnas originales")
        self._crear_tarjeta(
            indicadores,
            "PC1 + PC2",
            f"{varianza_acumulada(self.resumen['varianza']):.2%}",
            "Varianza acumulada mostrada",
        )

        self.notebook = ttk.Notebook(contenedor)
        self.notebook.pack(fill="both", expand=True)

        self._crear_pestana_resumen()
        self._crear_pestana_interpretacion()
        self._crear_pestana_matrices()
        self._crear_pestana_graficos()

    def _crear_tarjeta(self, contenedor, titulo, valor, detalle):
        """Crea una tarjeta compacta para mostrar indicadores principales."""
        tarjeta = ttk.Frame(contenedor, style="Card.TFrame", padding=16)
        tarjeta.pack(side="left", fill="x", expand=True, padx=8)

        ttk.Label(tarjeta, text=titulo, style="Accent.TLabel").pack(anchor="w")
        ttk.Label(tarjeta, text=valor, style="Metric.TLabel").pack(anchor="w", pady=(10, 2))
        ttk.Label(tarjeta, text=detalle, style="MetricValue.TLabel").pack(anchor="w")

    def _crear_texto_scrollable(self, padre):
        """Crea un bloque de texto con barras de desplazamiento."""
        contenedor = ttk.Frame(padre)
        contenedor.pack(fill="both", expand=True, padx=12, pady=12)

        # wrap="none" permite ver matrices anchas sin romper sus columnas.
        text = tk.Text(
            contenedor,
            wrap="none",
            bg="#0B1220",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
            padx=12,
            pady=12,
            height=12,
        )
        scroll_y = ttk.Scrollbar(contenedor, orient="vertical", command=text.yview)
        scroll_x = ttk.Scrollbar(contenedor, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        text.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        contenedor.grid_rowconfigure(0, weight=1)
        contenedor.grid_columnconfigure(0, weight=1)

        return text

    def _crear_pestana_resumen(self):
        """Muestra estadísticas, varianza, proyección y autovalores."""
        pestaña = ttk.Frame(self.notebook)
        self.notebook.add(pestaña, text="Resumen")

        contenedor = ttk.Frame(pestaña, padding=12)
        contenedor.pack(fill="both", expand=True)

        arriba = ttk.Frame(contenedor)
        arriba.pack(fill="x")

        tarjeta_izquierda = ttk.Frame(arriba, style="Card.TFrame", padding=16)
        tarjeta_izquierda.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tarjeta_derecha = ttk.Frame(arriba, style="Card.TFrame", padding=16)
        tarjeta_derecha.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(tarjeta_izquierda, text="Estadísticas básicas", style="Accent.TLabel").pack(anchor="w")
        texto_estadisticas = self._crear_texto_scrollable(tarjeta_izquierda)
        poblar_texto(
            texto_estadisticas,
            "Dimensiones: {}\n\nVariables:\n{}\n\nPromedios:\n{}\n\nMínimos:\n{}\n\nMáximos:\n{}".format(
                self.resumen["dimensiones"],
                formatear_variables(self.resumen["variables"]),
                formatear_matriz(self.resumen["media"]),
                formatear_matriz(self.resumen["minimo"]),
                formatear_matriz(self.resumen["maximo"]),
            ),
        )

        ttk.Label(tarjeta_derecha, text="Varianza explicada", style="Accent.TLabel").pack(anchor="w")
        texto_varianza = self._crear_texto_scrollable(tarjeta_derecha)
        poblar_texto(
            texto_varianza,
            "Varianza explicada:\n{}\n\nVarianza acumulada:\n{}\n\nVarianza acumulada (PC1 + PC2): {:.2%}".format(
                formatear_porcentajes(self.resumen["varianza"]),
                formatear_porcentajes(self.resumen["varianza_acumulada"]),
                varianza_acumulada(self.resumen["varianza"]),
            ),
        )

        abajo = ttk.Frame(contenedor)
        abajo.pack(fill="both", expand=True, pady=(12, 0))

        tarjeta_reducido = ttk.Frame(abajo, style="Card.TFrame", padding=16)
        tarjeta_reducido.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ttk.Label(tarjeta_reducido, text="Muestra de PCA", style="Accent.TLabel").pack(anchor="w")
        texto_reducido = self._crear_texto_scrollable(tarjeta_reducido)
        poblar_texto(
            texto_reducido,
            "Primeras filas de la proyección:\n{}".format(formatear_matriz(self.X_reducido[:8])),
        )

        tarjeta_autovalores = ttk.Frame(abajo, style="Card.TFrame", padding=16)
        tarjeta_autovalores.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(tarjeta_autovalores, text="Autovalores", style="Accent.TLabel").pack(anchor="w")
        texto_autovalores = self._crear_texto_scrollable(tarjeta_autovalores)
        poblar_texto(
            texto_autovalores,
            "Autovalores:\n{}\n\nAutovectores:\n{}".format(
                formatear_matriz(self.autovalores),
                formatear_matriz(self.autovectores),
            ),
        )

    def _crear_pestana_interpretacion(self):
        """Muestra una interpretación escrita de los resultados obtenidos."""
        pestaña = ttk.Frame(self.notebook)
        self.notebook.add(pestaña, text="Interpretación")

        contenedor = ttk.Frame(pestaña, padding=12)
        contenedor.pack(fill="both", expand=True)

        tarjeta = ttk.Frame(contenedor, style="Card.TFrame", padding=16)
        tarjeta.pack(fill="both", expand=True)

        ttk.Label(tarjeta, text="Lectura de resultados", style="Accent.TLabel").pack(anchor="w")
        texto = self._crear_texto_scrollable(tarjeta)
        texto.configure(wrap="word")
        poblar_texto(texto, crear_texto_interpretacion(self.resumen))

    def _crear_pestana_matrices(self):
        """Muestra matrices relevantes para comprobar el procedimiento PCA."""
        pestaña = ttk.Frame(self.notebook)
        self.notebook.add(pestaña, text="Matrices")

        contenedor = ttk.Frame(pestaña, padding=12)
        contenedor.pack(fill="both", expand=True)

        superior = ttk.Frame(contenedor)
        superior.pack(fill="both", expand=True)

        tarjetas = [
            ("Datos centrados", self.X_centrado),
            ("Matriz de covarianza", self.covarianza),
            ("Autovectores", self.autovectores),
        ]

        for indice, (titulo, matriz) in enumerate(tarjetas):
            tarjeta = ttk.Frame(superior, style="Card.TFrame", padding=14)
            tarjeta.grid(row=0, column=indice, sticky="nsew", padx=8)
            superior.grid_columnconfigure(indice, weight=1)

            ttk.Label(tarjeta, text=titulo, style="Accent.TLabel").pack(anchor="w")
            texto = self._crear_texto_scrollable(tarjeta)
            poblar_texto(texto, formatear_matriz(matriz))

    def _crear_pestana_graficos(self):
        """Muestra los gráficos principales dentro de una zona con scroll."""
        pestaña = ttk.Frame(self.notebook)
        self.notebook.add(pestaña, text="Gráficos")

        # El canvas con scrollbar evita que los graficos se corten en pantallas pequeñas.
        canvas_scroll = tk.Canvas(pestaña, bg="#0F172A", highlightthickness=0)
        scroll_y = ttk.Scrollbar(pestaña, orient="vertical", command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scroll_y.set)
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

        contenedor = ttk.Frame(canvas_scroll, padding=12)
        ventana = canvas_scroll.create_window((0, 0), window=contenedor, anchor="nw")

        def ajustar_scroll(event):
            # Actualiza el area desplazable cada vez que cambia el contenido.
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

        def ajustar_ancho(event):
            # Hace que las tarjetas usen todo el ancho disponible.
            canvas_scroll.itemconfigure(ventana, width=event.width)

        contenedor.bind("<Configure>", ajustar_scroll)
        canvas_scroll.bind("<Configure>", ajustar_ancho)

        figuras = [
            ("Datos originales", crear_figura_original(self.X, self.y, self.nombres, self.variables)),
            ("PCA", crear_figura_pca(self.X_reducido, self.y, self.nombres)),
            ("Varianza explicada", crear_figura_varianza(self.varianza)),
            ("Varianza acumulada", crear_figura_varianza_acumulada(self.varianza)),
        ]
        self.figuras.extend(figura for _, figura in figuras)

        for indice, (titulo, figura) in enumerate(figuras):
            tarjeta = ttk.Frame(contenedor, style="Card.TFrame", padding=12)
            tarjeta.grid(row=indice, column=0, sticky="nsew", pady=8)
            contenedor.grid_rowconfigure(indice, weight=1)
            contenedor.grid_columnconfigure(0, weight=1)

            ttk.Label(tarjeta, text=titulo, style="Accent.TLabel").pack(anchor="w", padx=8, pady=(6, 4))

            # FigureCanvasTkAgg convierte una figura Matplotlib en widget de Tkinter.
            canvas = FigureCanvasTkAgg(figura, master=tarjeta)
            self.canvases.append(canvas)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            canvas_widget.configure(width=1000, height=410)


def main():
    app = AppPCA()
    app.mainloop()
