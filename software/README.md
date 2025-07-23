# TAM - Medical Image 3D UMAP

**Análisis de Imágenes Médicas 3D con Reducción de Dimensionalidad UMAP**

Una aplicación completa para el análisis y visualización de imágenes médicas 3D con capacidades avanzadas de procesamiento de características y reducción de dimensionalidad usando UMAP.

## 🏥 Características Principales

- **Visualización 3D Interactiva**: Renderizado volumétrico con VTK
- **Procesamiento de Características**: Gradiente, Laplaciano, Curvatura, Textura y más
- **Reducción de Dimensionalidad**: Implementación optimizada de UMAP
- **Interfaz Gráfica Intuitiva**: Desarrollada con PyQt5
- **Soporte Multi-formato**: NIfTI (.nii, .nii.gz), MHA/MHD
- **Funciones de Transferencia**: Personalizables para optimizar visualización
- **Procesamiento por Lotes**: Análisis de múltiples volúmenes
- **Integración con Napari**: Visualización científica avanzada

## 📋 Requisitos del Sistema

### Requisitos Mínimos
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **RAM**: Mínimo 8GB (recomendado 16GB para volúmenes grandes)
- **Espacio en Disco**: 2GB libres

### Requisitos Recomendados
- **RAM**: 16GB o más
- **GPU**: NVIDIA GPU con soporte CUDA (opcional, para aceleración)
- **CPU**: Procesador multi-core (4+ cores)

## 🚀 Instalación

### Paso 1: Verificar Python

Asegúrate de tener Python 3.8 o superior instalado:

```bash
python --version
```

Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/)

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv medico3d_env

# Activar entorno virtual
# En Windows:
medico3d_env\Scripts\activate

# En Linux/macOS:
source medico3d_env/bin/activate
```

### Paso 3: Instalación Automática

Ejecuta el script de instalación que configurará todas las dependencias:

```bash
python setup.py
```

### Paso 4: Instalación Manual (Alternativa)

Si prefieres instalar manualmente:

```bash
pip install -r requirements.txt
```

## 🎯 Uso de la Aplicación

### Ejecutar la Aplicación

```bash
python main.py
```

### Flujo de Trabajo Básico

1. **Cargar Imagen Médica**
   - Archivo → Abrir Volumen
   - Selecciona archivo NIfTI (.nii, .nii.gz) o MHA/MHD
   - La imagen se cargará automáticamente en el visualizador 3D

2. **Configurar Visualización**
   - Ajusta la función de transferencia en la pestaña "Función de Transferencia"
   - Modifica opacidad y colores para optimizar la visualización
   - Usa los controles de cámara para navegar en 3D

3. **Procesar Características**
   - Ve a la pestaña "Procesamiento"
   - Selecciona las características a calcular:
     - Gradiente 3D
     - Laplaciano
     - Curvatura Media/Gaussiana
     - Características de Textura (LBP, GLCM)
     - Estadísticas Locales
   - Haz clic en "Procesar Todas las Características"

4. **Aplicar UMAP**
   - En la pestaña "UMAP", configura los parámetros:
     - Número de vecinos (n_neighbors): 15-50
     - Distancia mínima (min_dist): 0.1-0.5
     - Componentes: 2 o 3
   - Haz clic en "Ejecutar UMAP"
   - Los resultados se mostrarán en una nueva ventana

5. **Exportar Resultados**
   - Archivo → Exportar Resultados
   - Los archivos se guardarán en la carpeta `output/`

### Datos de Ejemplo

La aplicación incluye un volumen de muestra (`sample_volume.nii.gz`) para pruebas iniciales.

## 📁 Estructura del Proyecto

```
TAM - Medical Image 3d UMAP/
├── main.py                 # Aplicación principal
├── config.py              # Gestión de configuración
├── transfer_functions.py  # Funciones de transferencia
├── image_processor.py     # Procesamiento de imágenes
├── napari_widget.py       # Integración con Napari
├── setup.py               # Script de instalación
├── requirements.txt       # Dependencias Python
├── sample_volume.nii.gz   # Volumen de muestra
├── README.md             # Este archivo
├── config/               # Archivos de configuración
├── output/               # Resultados de procesamiento
├── temp/                 # Archivos temporales
└── logs/                 # Archivos de log
```

## ⚙️ Configuración Avanzada

### Optimización de Memoria

Para volúmenes grandes, ajusta estos parámetros en `config.py`:

```python
# Configuración de memoria
MAX_VOLUME_SIZE = 512  # Tamaño máximo por dimensión
CHUNK_SIZE = 64        # Tamaño de chunk para procesamiento
USE_MEMORY_MAPPING = True  # Usar mapeo de memoria
```

### Aceleración GPU

Para habilitar aceleración GPU (requiere NVIDIA GPU):

```bash
# Instalar CuPy para aceleración CUDA
pip install cupy-cuda11x  # Para CUDA 11.x
# o
pip install cupy-cuda12x  # Para CUDA 12.x
```

### Configuración de Calidad de Renderizado

Ajusta la calidad de renderizado según tu hardware:

```python
# En config.py
RENDER_QUALITY = "high"    # "low", "medium", "high"
MAX_RENDER_POINTS = 100000 # Máximo número de puntos para render
```

## 🔧 Solución de Problemas

### Problemas Comunes

1. **"Error cargando archivo"**
   - Verifica que el archivo sea un formato soportado (NIfTI, MHA/MHD)
   - Comprueba que el archivo no esté corrupto
   - Asegúrate de tener permisos de lectura

2. **"Error en visualización VTK"**
   - Actualiza los drivers de tu GPU
   - Reduce la calidad de renderizado
   - Verifica que VTK esté correctamente instalado

3. **"Memoria insuficiente"**
   - Reduce el tamaño del volumen usando submuestreo
   - Cierra otras aplicaciones que consuman RAM
   - Ajusta los parámetros de memoria en configuración

4. **"Error en UMAP"**
   - Reduce el número de características procesadas
   - Ajusta los parámetros de UMAP (menos vecinos)
   - Verifica que scikit-learn esté actualizado

### Logs de Depuración

Los logs se guardan automáticamente en la carpeta `logs/`:

```bash
# Ver logs recientes
tail -f logs/medico3d.log
```

### Reinstalación Limpia

Si experimentas problemas persistentes:

```bash
# Desactivar entorno virtual
deactivate

# Eliminar entorno virtual
rm -rf medico3d_env  # Linux/macOS
rmdir /s medico3d_env  # Windows

# Crear nuevo entorno y reinstalar
python -m venv medico3d_env
medico3d_env\Scripts\activate  # Windows
python setup.py
```

## 📊 Formatos de Archivo Soportados

### Entrada
- **NIfTI**: `.nii`, `.nii.gz` (recomendado)
- **MHA/MHD**: `.mha`, `.mhd`
- **DICOM**: Soporte básico (requiere conversión previa)

### Salida
- **Características**: `.npy`, `.nii.gz`
- **Resultados UMAP**: `.npy`, `.csv`
- **Visualizaciones**: `.png`, `.jpg`
- **Reportes**: `.html`, `.json`

## 🤝 Contribuciones

Este proyecto es una migración y optimización de la versión original en C#/.NET. 

### Características Implementadas
- ✅ Interfaz gráfica completa con PyQt5
- ✅ Visualización 3D con VTK
- ✅ Procesamiento de características avanzado
- ✅ Integración UMAP optimizada
- ✅ Gestión de memoria mejorada
- ✅ Soporte multi-formato

### Mejoras Futuras
- 🔄 Soporte para más formatos de imagen
- 🔄 Algoritmos de segmentación automática
- 🔄 Análisis estadístico avanzado
- 🔄 Exportación a formatos 3D estándar



## 📞 Soporte

Para reportar problemas o solicitar características:

1. Verifica la sección de solución de problemas
2. Revisa los logs en la carpeta `logs/`
3. Incluye información del sistema y pasos para reproducir el problema

---

**Desarrollado con ❤️ para la materia TAM de la Unal Manizales**