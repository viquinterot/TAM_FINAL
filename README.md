# Exploración Interactiva Acelerada de Volúmenes Médicos con UMAP en GPU


> Repositorio del proyecto que demuestra un pipeline computacional para la exploración interactiva de volúmenes médicos. El sistema utiliza un embedding de características con UMAP, acelerado en GPU con NVIDIA RAPIDS, y una interfaz de visualización 3D construida con Vedo.

**Autores:**
*   **Autor Principal:** Víctor Germán Quintero Toro (`viquinterot@unal.edu.co`)
*   **Asesor:** Andrés Marino Álvarez Meza, PhD

**Afiliación:** Departamento de Ingeniería Eléctrica, Electrónica y Computación, Universidad Nacional de Colombia - Sede Manizales.

---

## Tabla de Contenidos
* [Acerca del Proyecto](#acerca-del-proyecto)
* [Características Principales](#características-principales)
* [Tecnologías Utilizadas](#tecnologías-utilizadas)
* [Metodología del Pipeline](#metodología-del-pipeline)
* [Cómo Empezar](#cómo-empezar)
  * [Prerrequisitos](#prerrequisitos)
  * [Instalación](#instalación)
* [Uso](#uso)
* [Trabajo Futuro](#trabajo-futuro)
* [Licencia](#licencia)
* [Agradecimientos](#agradecimientos)

## Acerca del Proyecto

La interpretación de imágenes médicas 3D, como las de Tomografía Computarizada (TC), es una tarea fundamental pero compleja. La **Renderización Directa de Volumen (DVR)** es una técnica potente para visualizar estos datos, pero su eficacia depende críticamente del diseño de una **Función de Transferencia (TF)**, que asigna propiedades ópticas (color y opacidad) a cada vóxel.

Tradicionalmente, el diseño de TFs es un proceso manual y poco intuitivo. Los enfoques modernos que usan espacios de características de alta dimensionalidad (gradiente, Laplaciano, etc.) son más robustos pero computacionalmente costosos y difíciles de explorar para un humano.

Este proyecto presenta una solución a este problema mediante un **pipeline acelerado por GPU** que permite una exploración fluida y en tiempo real. La idea central es:
1.  Extraer un conjunto de características de cada vóxel.
2.  Utilizar **UMAP (Uniform Manifold Approximation and Projection)**, acelerado con la librería `cuML` de NVIDIA RAPIDS, para proyectar estas características a un espacio 2D manejable.
3.  Vincular este espacio 2D con el render 3D en una **interfaz interactiva de doble panel**. El usuario puede seleccionar clústeres de vóxeles en el mapa 2D, y el sistema actualiza la TF en tiempo real para aislar y resaltar las estructuras anatómicas correspondientes en la visualización 3D.

Este enfoque supera los cuellos de botella de los métodos basados en CPU y transforma el análisis de características en una parte integral y dinámica del bucle de visualización.

## Características Principales

*   **Rendimiento en Tiempo Real:** El cálculo del embedding UMAP para millones de vóxeles se completa en segundos gracias a la aceleración en GPU.
*   **Actualización Instantánea:** La Función de Transferencia (TF) se actualiza de forma inmediata al interactuar con el espacio de características.
*   **Separación Clara de Estructuras:** UMAP logra una excelente separación de estructuras anatómicas (hueso, tejido blando, aire) en clústeres visualmente definidos.
*   **Exploración Intuitiva:** Permite a los usuarios (clínicos, investigadores) aislar estructuras de interés simplemente haciendo clic en los clústeres correspondientes, facilitando la segmentación visual y el diagnóstico.

## Tecnologías Utilizadas

Este proyecto se construyó utilizando las siguientes tecnologías y librerías:

*   **[Python](https://www.python.org/)**: Lenguaje principal de desarrollo.
*   **[NVIDIA RAPIDS](https://rapids.ai/)**: Suite de librerías para ciencia de datos acelerada por GPU.
    *   **[cuML](https://github.com/rapidsai/cuml)**: Para la implementación de UMAP y `StandardScaler` en GPU.
    *   **[CuPy](https://cupy.dev/)**: Para la gestión de arreglos de datos en la memoria de la GPU.
*   **[Vedo](https://vedo.embl.es/)**: Para la creación de la visualización 3D y la interfaz interactiva de doble panel.
*   **[Scikit-image](https://scikit-image.org/)**: Para la extracción de características de las imágenes (Gradiente Gaussiano, Laplaciano).
*   **[Tifffile](https://pypi.org/project/tifffile/)**: Para la carga de secuencias de imágenes TIF.

## Metodología del Pipeline

El flujo de trabajo computacional se divide en tres etapas clave:

1.  **Extracción de Características**: Para cada vóxel del volumen 3D, se extrae un vector de 3 características para describir su valor y contexto local:
    *   **Intensidad original**: El valor escalar del vóxel.
    *   **Magnitud del Gradiente Gaussiano**: Estima la tasa de cambio de intensidad, útil para detectar bordes.
    *   **Laplaciano de Gaussiano (LoG)**: Detecta regiones de cambio rápido, útil para encontrar texturas.

2.  **Reducción de Dimensionalidad con UMAP en GPU**:
    *   Las características se transfieren a la memoria de la GPU como un arreglo `CuPy`.
    *   Se aplica un escalado estándar (`cuml.StandardScaler`).
    *   Se ejecuta `cuml.UMAP` sobre los datos escalados para generar el embedding 2D, preservando la estructura de los datos.

3.  **Visualización y Diseño de TF Interactivo**:
    *   Se utiliza `Vedo` para crear una aplicación de dos paneles.
    *   Una función de _callback_ se activa al hacer clic en el espacio UMAP, definiendo una **Región de Interés (ROI)**.
    *   El sistema identifica los vóxeles dentro del ROI y actualiza en tiempo real la TF del volumen 3D, asignando un color y opacidad distintivos a la selección.
Para ejecutar la aplicación, utiliza el siguiente comando. Asegúrate de tener un conjunto de datos (por ejemplo, una serie de archivos TIF de una TC) en una carpeta accesible.

```sh
python main_script.py --path_a_tus_datos
