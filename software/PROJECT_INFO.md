# Información del Proyecto

## TAM - Medical Image 3D UMAP

**Versión**: 1.0.0  
**Fecha de Creación**: Diciembre 2024  
**Tipo**: Aplicación de Análisis de Imágenes Médicas 3D  

### Descripción
Esta es una versión portable y optimizada del proyecto Medico3D, migrada de C#/.NET a Python. La aplicación se enfoca en el análisis de imágenes médicas 3D con capacidades avanzadas de procesamiento de características y reducción de dimensionalidad usando UMAP.

### Archivos Incluidos
- `main.py` - Aplicación principal con interfaz PyQt5
- `config.py` - Gestión de configuración y parámetros
- `transfer_functions.py` - Funciones de transferencia para visualización
- `image_processor.py` - Procesamiento de características de imagen
- `napari_widget.py` - Integración con Napari para visualización científica
- `setup.py` - Script de instalación automática
- `requirements.txt` - Lista de dependencias Python
- `sample_volume.nii.gz` - Volumen de muestra para pruebas

### Estructura de Carpetas
- `config/` - Archivos de configuración
- `output/` - Resultados de procesamiento
- `temp/` - Archivos temporales
- `logs/` - Archivos de registro

### Scripts de Inicio
- `start.bat` - Script de inicio para Windows
- `start.sh` - Script de inicio para Linux/macOS

### Características Principales
1. **Visualización 3D**: Renderizado volumétrico interactivo con VTK
2. **Procesamiento de Características**: 
   - Gradiente 3D
   - Laplaciano
   - Curvatura (Media y Gaussiana)
   - Características de Textura (LBP, GLCM)
   - Estadísticas Locales
3. **Reducción de Dimensionalidad**: Implementación optimizada de UMAP
4. **Interfaz Intuitiva**: Desarrollada con PyQt5
5. **Soporte Multi-formato**: NIfTI, MHA/MHD
6. **Procesamiento por Lotes**: Análisis de múltiples volúmenes

### Requisitos del Sistema
- Python 3.8 o superior
- 8GB RAM mínimo (16GB recomendado)
- Windows 10/11, Linux, o macOS
- GPU NVIDIA opcional para aceleración

### Inicio Rápido
1. Ejecutar `start.bat` (Windows) o `start.sh` (Linux/macOS)
2. La primera ejecución instalará automáticamente las dependencias
3. Cargar un volumen médico desde Archivo → Abrir Volumen
4. Procesar características en la pestaña "Procesamiento"
5. Aplicar UMAP en la pestaña correspondiente

### Migración desde Versión Original
Esta versión mantiene la funcionalidad completa de la aplicación original en C#/.NET, con las siguientes mejoras:
- Mejor gestión de memoria
- Interfaz más moderna
- Instalación simplificada
- Soporte multiplataforma mejorado
- Integración con herramientas científicas de Python

### Soporte
Para problemas o preguntas, consulta el archivo README.md o revisa los logs en la carpeta `logs/`.

---
*Desarrollado como migración Python del proyecto Medico3D original*