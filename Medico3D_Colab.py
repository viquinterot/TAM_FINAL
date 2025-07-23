# 🏥 Medico3D para Google Colab
# Análisis de Imágenes Médicas 3D con Aceleración GPU
# Versión optimizada para Google Colab

"""
Medico3D Colab - Análisis de imágenes médicas 3D en Google Colab
===============================================================

Esta versión está optimizada para ejecutarse en Google Colab con:
- Aceleración GPU automática (CUDA)
- Interfaz interactiva con widgets
- Visualización 3D integrada
- Procesamiento de características avanzadas
- Soporte para formatos médicos estándar

Autor: Medico3D Team
Versión: 2.0.0-Colab
"""

# ============================================================================
# CONFIGURACIÓN E INSTALACIÓN AUTOMÁTICA
# ============================================================================

import sys
import subprocess
import os
from pathlib import Path

def install_dependencies():
    """Instalar dependencias necesarias para Colab"""
    print("🔧 Instalando dependencias para Medico3D...")
    
    # Dependencias básicas
    packages = [
        "nibabel",           # Lectura de archivos médicos
        "SimpleITK",         # Procesamiento de imágenes médicas
        "vtk",              # Visualización 3D
        "plotly",           # Gráficos interactivos
        "ipywidgets",       # Widgets interactivos
        "scikit-image",     # Procesamiento de imágenes
        "umap-learn",       # Reducción de dimensionalidad
        "pandas",           # Manipulación de datos
        "seaborn",          # Visualización estadística
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
    
    # Verificar si CuPy ya está disponible
    try:
        import cupy as cp
        print("✅ CuPy ya está disponible")
        return True
    except ImportError:
        pass
    
    # Intentar instalar CuPy para GPU (opcional)
    try:
        print("\n🚀 Intentando instalar CuPy para aceleración GPU...")
        # Intentar CUDA 12.x primero (más común en Colab actual)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "cupy-cuda12x", "-q"
            ])
            print("✅ CuPy CUDA 12.x instalado")
            return True
        except:
            # Fallback a CUDA 11.x
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "cupy-cuda11x", "-q"
                ])
                print("✅ CuPy CUDA 11.x instalado")
                return True
            except:
                print("⚠️  CuPy no se pudo instalar, usando CPU")
                return False
    except:
        print("⚠️  Error en instalación de GPU, usando CPU")
        return False

# Instalar dependencias al importar
gpu_installation_attempted = install_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from scipy import stats as scipy_stats
from skimage import filters, measure, morphology
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration detection (mejorado)
GPU_ENABLED = False
CUPY_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUPY_AVAILABLE = True
    
    # Verificar que CuPy puede acceder a la GPU
    try:
        # Intentar crear un array pequeño en GPU
        test_array = cp.array([1, 2, 3])
        _ = cp.asnumpy(test_array)  # Convertir de vuelta a CPU
        GPU_ENABLED = True
        print("🚀 GPU acceleration enabled")
        print(f"🔧 GPU disponible: Sí")
        
        # Información adicional de GPU
        try:
            device = cp.cuda.Device()
            print(f"🎯 GPU: {device.attributes['Name'].decode()}")
        except:
            print("🎯 GPU: Detectada (nombre no disponible)")
            
    except Exception as e:
        print(f"⚠️  CuPy importado pero GPU no accesible: {e}")
        print("💻 Using CPU processing")
        print(f"🔧 GPU disponible: No")
        
except ImportError:
    print("💻 Using CPU processing")
    print(f"🔧 GPU disponible: No")

# Intentar importar cuML si está disponible
try:
    from cuml import UMAP as cuUMAP
    CUML_AVAILABLE = GPU_ENABLED and CUPY_AVAILABLE  # Solo usar cuML si GPU y CuPy están disponibles
except ImportError:
    CUML_AVAILABLE = False

# UMAP fallback
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ============================================================================
# CLASE PRINCIPAL MEDICO3D COLAB
# ============================================================================

class Medico3DColab:
    """
    Clase principal para análisis de imágenes médicas en Google Colab
    """
    
    def __init__(self):
        """Inicializar Medico3D para Colab"""
        self.volume = None
        self.volume_info = {}
        self.features = {}
        self.results = {}
        self.use_gpu = GPU_ENABLED
        
        print("🏥 Medico3D Colab inicializado")
        print(f"🔧 GPU: {'Habilitado' if self.use_gpu else 'Deshabilitado'}")
        
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
    
    # Propiedades para acceso directo a características calculadas
    @property
    def gradient(self):
        """Acceso directo al gradiente calculado"""
        return self.features.get('gradient', {}).get('magnitude', None)
    
    @property
    def laplacian(self):
        """Acceso directo al laplaciano calculado"""
        return self.features.get('laplacian', None)
    
    @property
    def mean_feature(self):
        """Acceso directo a la característica de media local"""
        return self.features.get('statistical', {}).get('mean', None)
    
    @property
    def std_feature(self):
        """Acceso directo a la característica de desviación estándar local"""
        return self.features.get('statistical', {}).get('std', None)
    
    @property
    def var_feature(self):
        """Acceso directo a la característica de varianza local"""
        return self.features.get('statistical', {}).get('variance', None)
    
    @property
    def gaussian_filtered(self):
        """Acceso directo al volumen filtrado con Gaussiano"""
        return self.features.get('gaussian_filtered', None)
    
    @property
    def umap_embedding(self):
        """Acceso directo al embedding UMAP"""
        return self.features.get('umap_embedding', None)

    def load_volume(self, file_path: str) -> bool:
        """
        Cargar volumen médico desde archivo
        
        Parámetros:
        -----------
        file_path : str
            Ruta al archivo de imagen médica
            
        Retorna:
        --------
        bool : True si se cargó exitosamente
        """
        try:
            print(f"📂 Cargando: {file_path}")
            
            # Detectar formato y cargar
            if file_path.endswith(('.nii', '.nii.gz')):
                img = nib.load(file_path)
                self.volume = img.get_fdata()
                header = img.header
                self.volume_info = {
                    'format': 'NIfTI',
                    'shape': self.volume.shape,
                    'voxel_size': header.get_zooms(),
                    'data_type': str(self.volume.dtype)
                }
            elif file_path.endswith(('.mha', '.mhd')):
                img = sitk.ReadImage(file_path)
                self.volume = sitk.GetArrayFromImage(img)
                self.volume_info = {
                    'format': 'MetaImage',
                    'shape': self.volume.shape,
                    'voxel_size': img.GetSpacing(),
                    'data_type': str(self.volume.dtype)
                }
            else:
                raise ValueError(f"Formato no soportado: {file_path}")
            
            # Información del volumen
            self.volume_info.update({
                'min_value': float(np.min(self.volume)),
                'max_value': float(np.max(self.volume)),
                'mean_value': float(np.mean(self.volume)),
                'std_value': float(np.std(self.volume)),
                'file_path': file_path
            })
            
            print(f"✅ Volumen cargado: {self.volume.shape}")
            print(f"📊 Rango: [{self.volume_info['min_value']:.2f}, {self.volume_info['max_value']:.2f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando volumen: {e}")
            return False
    
    def show_volume_info(self):
        """Mostrar información detallada del volumen"""
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return
        
        # Crear tabla de información
        info_df = pd.DataFrame([
            ['Formato', self.volume_info['format']],
            ['Dimensiones', f"{self.volume_info['shape']}"],
            ['Tipo de datos', self.volume_info['data_type']],
            ['Tamaño voxel', f"{self.volume_info['voxel_size']}"],
            ['Valor mínimo', f"{self.volume_info['min_value']:.4f}"],
            ['Valor máximo', f"{self.volume_info['max_value']:.4f}"],
            ['Valor medio', f"{self.volume_info['mean_value']:.4f}"],
            ['Desviación estándar', f"{self.volume_info['std_value']:.4f}"],
        ], columns=['Propiedad', 'Valor'])
        
        print("📋 INFORMACIÓN DEL VOLUMEN")
        print("=" * 40)
        display(info_df)
        
        # Histograma de intensidades
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histograma
        ax1.hist(self.volume.flatten(), bins=100, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Intensidad')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Histograma de Intensidades')
        ax1.grid(True, alpha=0.3)
        
        # Estadísticas por slice
        slice_means = [np.mean(self.volume[i]) for i in range(self.volume.shape[0])]
        ax2.plot(slice_means, color='orange', linewidth=2)
        ax2.set_xlabel('Slice')
        ax2.set_ylabel('Intensidad Media')
        ax2.set_title('Intensidad Media por Slice')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_slices(self, slice_indices: Optional[List[int]] = None, 
                        axis: int = 0, cmap: str = 'gray'):
        """
        Visualizar slices del volumen
        
        Parámetros:
        -----------
        slice_indices : List[int], opcional
            Índices de slices a mostrar
        axis : int
            Eje para extraer slices (0, 1, o 2)
        cmap : str
            Mapa de colores
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return
        
        # Slices por defecto
        if slice_indices is None:
            total_slices = self.volume.shape[axis]
            slice_indices = [
                total_slices // 4,
                total_slices // 2,
                3 * total_slices // 4
            ]
        
        n_slices = len(slice_indices)
        fig, axes = plt.subplots(1, n_slices, figsize=(4*n_slices, 4))
        
        if n_slices == 1:
            axes = [axes]
        
        axis_names = ['Sagital', 'Coronal', 'Axial']
        
        for i, slice_idx in enumerate(slice_indices):
            if axis == 0:
                slice_data = self.volume[slice_idx, :, :]
            elif axis == 1:
                slice_data = self.volume[:, slice_idx, :]
            else:
                slice_data = self.volume[:, :, slice_idx]
            
            im = axes[i].imshow(slice_data, cmap=cmap, origin='lower')
            axes[i].set_title(f'{axis_names[axis]} - Slice {slice_idx}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_viewer(self):
        """Crear visor interactivo con widgets"""
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return
        
        # Widgets de control
        axis_widget = widgets.Dropdown(
            options=[('Sagital (X)', 0), ('Coronal (Y)', 1), ('Axial (Z)', 2)],
            value=2,
            description='Eje:'
        )
        
        slice_widget = widgets.IntSlider(
            value=self.volume.shape[2] // 2,
            min=0,
            max=self.volume.shape[2] - 1,
            description='Slice:'
        )
        
        cmap_widget = widgets.Dropdown(
            options=['gray', 'viridis', 'plasma', 'inferno', 'hot', 'cool'],
            value='gray',
            description='Color:'
        )
        
        def update_slice(axis, slice_idx, cmap):
            with plt.ioff():
                fig, ax = plt.subplots(figsize=(8, 6))
                
                if axis == 0:
                    slice_data = self.volume[slice_idx, :, :]
                    slice_widget.max = self.volume.shape[0] - 1
                elif axis == 1:
                    slice_data = self.volume[:, slice_idx, :]
                    slice_widget.max = self.volume.shape[1] - 1
                else:
                    slice_data = self.volume[:, :, slice_idx]
                    slice_widget.max = self.volume.shape[2] - 1
                
                im = ax.imshow(slice_data, cmap=cmap, origin='lower')
                ax.set_title(f'Slice {slice_idx} - Eje {axis}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.show()
        
        # Actualizar rango del slider cuando cambia el eje
        def update_axis(change):
            axis = change['new']
            slice_widget.max = self.volume.shape[axis] - 1
            slice_widget.value = self.volume.shape[axis] // 2
        
        axis_widget.observe(update_axis, names='value')
        
        # Widget interactivo
        interactive_plot = widgets.interactive(
            update_slice,
            axis=axis_widget,
            slice_idx=slice_widget,
            cmap=cmap_widget
        )
        
        display(interactive_plot)
    
    def compute_gradient(self) -> Dict[str, np.ndarray]:
        """
        Calcular gradiente del volumen
        
        Retorna:
        --------
        Dict con componentes del gradiente
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return {}
        
        print("🔄 Calculando gradiente...")
        
        try:
            if self.use_gpu:
                # Procesamiento GPU
                volume_gpu = cp.asarray(self.volume)
                gx = cp_ndimage.sobel(volume_gpu, axis=0)
                gy = cp_ndimage.sobel(volume_gpu, axis=1)
                gz = cp_ndimage.sobel(volume_gpu, axis=2)
                magnitude = cp.sqrt(gx**2 + gy**2 + gz**2)
                
                result = {
                    'Gx': cp.asnumpy(gx),
                    'Gy': cp.asnumpy(gy),
                    'Gz': cp.asnumpy(gz),
                    'magnitude': cp.asnumpy(magnitude)
                }
            else:
                # Procesamiento CPU
                gx = ndimage.sobel(self.volume, axis=0)
                gy = ndimage.sobel(self.volume, axis=1)
                gz = ndimage.sobel(self.volume, axis=2)
                magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
                
                result = {
                    'Gx': gx,
                    'Gy': gy,
                    'Gz': gz,
                    'magnitude': magnitude
                }
            
            self.features['gradient'] = result
            print("✅ Gradiente calculado")
            return result
            
        except Exception as e:
            print(f"❌ Error calculando gradiente: {e}")
            return {}
    
    def compute_laplacian(self) -> np.ndarray:
        """
        Calcular Laplaciano del volumen
        
        Retorna:
        --------
        np.ndarray : Laplaciano del volumen
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return np.array([])
        
        print("🔄 Calculando Laplaciano...")
        
        try:
            if self.use_gpu:
                volume_gpu = cp.asarray(self.volume)
                laplacian_gpu = cp_ndimage.laplace(volume_gpu)
                laplacian = cp.asnumpy(laplacian_gpu)
            else:
                laplacian = ndimage.laplace(self.volume)
            
            self.features['laplacian'] = laplacian
            print("✅ Laplaciano calculado")
            return laplacian
            
        except Exception as e:
            print(f"❌ Error calculando Laplaciano: {e}")
            return np.array([])
    
    def compute_statistical_features(self, window_size: int = 3) -> Dict[str, np.ndarray]:
        """
        Calcular características estadísticas locales
        
        Parámetros:
        -----------
        window_size : int
            Tamaño de ventana para estadísticas locales
            
        Retorna:
        --------
        Dict con características estadísticas
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return {}
        
        print(f"🔄 Calculando características estadísticas (ventana: {window_size})...")
        
        try:
            # Media local
            mean_filter = ndimage.uniform_filter(self.volume, size=window_size)
            
            # Desviación estándar local
            sqr_filter = ndimage.uniform_filter(self.volume**2, size=window_size)
            std_filter = np.sqrt(sqr_filter - mean_filter**2)
            
            # Varianza local
            var_filter = std_filter**2
            
            result = {
                'mean': mean_filter,
                'std': std_filter,
                'variance': var_filter
            }
            
            self.features['statistical'] = result
            print("✅ Características estadísticas calculadas")
            return result
            
        except Exception as e:
            print(f"❌ Error calculando características estadísticas: {e}")
            return {}
    
    def apply_gaussian_filter(self, sigma: float = 1.0) -> np.ndarray:
        """
        Aplicar filtro Gaussiano
        
        Parámetros:
        -----------
        sigma : float
            Desviación estándar del kernel Gaussiano
            
        Retorna:
        --------
        np.ndarray : Volumen filtrado
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return np.array([])
        
        print(f"🔄 Aplicando filtro Gaussiano (σ={sigma})...")
        
        try:
            if self.use_gpu:
                volume_gpu = cp.asarray(self.volume)
                filtered_gpu = cp_ndimage.gaussian_filter(volume_gpu, sigma=sigma)
                filtered = cp.asnumpy(filtered_gpu)
            else:
                filtered = ndimage.gaussian_filter(self.volume, sigma=sigma)
            
            self.features['gaussian_filtered'] = filtered
            print("✅ Filtro Gaussiano aplicado")
            return filtered
            
        except Exception as e:
            print(f"❌ Error aplicando filtro Gaussiano: {e}")
            return np.array([])
    
    def compute_umap_embedding(self, n_components: int = 3, n_neighbors: int = 15, 
                              min_dist: float = 0.1, sample_ratio: float = 0.05) -> np.ndarray:
        """
        Calcular embedding UMAP para reducción de dimensionalidad
        OPTIMIZADO PARA GOOGLE COLAB CON DETECCIÓN AUTOMÁTICA DE VRAM
        
        Parámetros:
        -----------
        n_components : int
            Número de componentes del embedding
        n_neighbors : int
            Número de vecinos para UMAP (ajustado automáticamente según VRAM)
        min_dist : float
            Distancia mínima en el embedding
        sample_ratio : float
            Ratio de muestreo (ajustado automáticamente según VRAM disponible)
            
        Retorna:
        --------
        np.ndarray : Embedding UMAP
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return np.array([])
        
        if not UMAP_AVAILABLE:
            print("❌ UMAP no disponible")
            return np.array([])
        
        # Detectar VRAM disponible y optimizar parámetros automáticamente
        gpu_memory_gb = 0
        if self.use_gpu and CUML_AVAILABLE:
            try:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                gpu_memory_bytes = cp.cuda.Device().mem_info[1]  # Memoria total
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
                print(f"🔥 GPU detectada con {gpu_memory_gb:.1f} GB VRAM")
            except:
                gpu_memory_gb = 0
        
        # Detectar si estamos en Colab
        in_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        
        # OPTIMIZACIÓN INTELIGENTE BASADA EN VRAM DISPONIBLE
        if gpu_memory_gb >= 12:  # GPU de alta gama (12+ GB)
            print("🚀 GPU de alta gama detectada - Configuración agresiva")
            sample_ratio = min(sample_ratio, 0.25)  # Hasta 25% de los datos
            n_neighbors = min(n_neighbors, 50)      # Hasta 50 vecinos
            batch_size_multiplier = 4
        elif gpu_memory_gb >= 8:   # GPU de gama media (8-12 GB)
            print("⚡ GPU de gama media detectada - Configuración balanceada")
            sample_ratio = min(sample_ratio, 0.15)  # Hasta 15% de los datos
            n_neighbors = min(n_neighbors, 30)      # Hasta 30 vecinos
            batch_size_multiplier = 3
        elif gpu_memory_gb >= 4:   # GPU básica (4-8 GB)
            print("💻 GPU básica detectada - Configuración moderada")
            sample_ratio = min(sample_ratio, 0.08)  # Hasta 8% de los datos
            n_neighbors = min(n_neighbors, 20)      # Hasta 20 vecinos
            batch_size_multiplier = 2
        elif in_colab:
            print("🔧 Detectado Google Colab - Configuración conservadora")
            sample_ratio = min(sample_ratio, 0.02)  # Máximo 2% en Colab básico
            n_neighbors = min(n_neighbors, 10)      # Máximo 10 vecinos
            batch_size_multiplier = 1
        else:
            print("💾 Configuración CPU/GPU limitada")
            sample_ratio = min(sample_ratio, 0.05)  # 5% por defecto
            n_neighbors = min(n_neighbors, 15)      # 15 vecinos por defecto
            batch_size_multiplier = 1
        
        print(f"🔄 Calculando UMAP embedding...")
        print(f"   🎯 Configuración optimizada para {gpu_memory_gb:.1f} GB VRAM")
        print(f"   📊 Muestreo: {sample_ratio*100:.1f}%")
        print(f"   🔗 Vecinos: {n_neighbors}")
        print(f"   📦 Componentes: {n_components}")
        
        # Información del volumen
        total_voxels = self.volume.size
        print(f"   📦 Volumen total: {total_voxels:,} voxels")
        
        try:
            import time
            start_time = time.time()
            
            # Preparar datos con mejoras para evitar errores CUML/RAFT
            print("📋 Preparando datos...")
            volume_flat = self.volume.reshape(-1, 1)
            
            # SOLUCIÓN PARA ERROR CUML/RAFT: Añadir ruido mínimo para evitar puntos idénticos
            print("🔧 Añadiendo ruido mínimo para evitar puntos idénticos...")
            noise_scale = np.std(volume_flat) * 1e-6  # Ruido muy pequeño (0.0001% del std)
            if noise_scale == 0:  # Si todos los valores son idénticos
                noise_scale = 1e-6
            
            # Añadir ruido gaussiano mínimo
            noise = np.random.normal(0, noise_scale, volume_flat.shape)
            volume_flat_noisy = volume_flat + noise
            
            # Verificar que no hay valores idénticos
            unique_values = len(np.unique(volume_flat_noisy))
            total_values = len(volume_flat_noisy)
            print(f"🔍 Valores únicos: {unique_values:,} de {total_values:,} ({unique_values/total_values*100:.2f}%)")
            
            # Si aún hay muchos valores idénticos, añadir más diversidad
            if unique_values < total_values * 0.1:  # Menos del 10% son únicos
                print("⚠️  Detectados muchos valores idénticos, añadiendo más diversidad...")
                # Añadir índices normalizados para garantizar unicidad
                index_noise = np.arange(len(volume_flat_noisy)).reshape(-1, 1) * noise_scale * 0.1
                volume_flat_noisy = volume_flat_noisy + index_noise
                
                unique_values_new = len(np.unique(volume_flat_noisy))
                print(f"✅ Valores únicos después de diversificación: {unique_values_new:,}")
            
            volume_flat = volume_flat_noisy
            
            # Muestreo inteligente para acelerar
            if sample_ratio < 1.0:
                n_samples = int(len(volume_flat) * sample_ratio)
                print(f"🎯 Muestreando {n_samples:,} de {len(volume_flat):,} voxels")
                
                # Muestreo estratificado para mejor representación
                # Dividir en bins de intensidad y muestrear proporcionalmente
                volume_1d = volume_flat.flatten()
                hist, bin_edges = np.histogram(volume_1d, bins=20)
                
                indices = []
                for i in range(len(hist)):
                    if hist[i] > 0:
                        bin_mask = (volume_1d >= bin_edges[i]) & (volume_1d < bin_edges[i+1])
                        bin_indices = np.where(bin_mask)[0]
                        if len(bin_indices) > 0:
                            # Muestrear proporcionalmente de cada bin
                            n_from_bin = max(1, int(len(bin_indices) * sample_ratio))
                            selected = np.random.choice(bin_indices, 
                                                      min(n_from_bin, len(bin_indices)), 
                                                      replace=False)
                            indices.extend(selected)
                
                indices = np.array(indices)
                if len(indices) > n_samples:
                    indices = np.random.choice(indices, n_samples, replace=False)
                
                volume_sample = volume_flat[indices]
                print(f"✅ Muestreo estratificado completado: {len(indices):,} muestras")
            else:
                volume_sample = volume_flat
                indices = np.arange(len(volume_flat))
                print("📊 Usando volumen completo (sin muestreo)")
            
            prep_time = time.time() - start_time
            print(f"⏱️  Preparación: {prep_time:.1f}s")
            
            # UMAP con progreso y validaciones mejoradas
            umap_start = time.time()
            print("🚀 Ejecutando UMAP...")
            
            # Validaciones adicionales antes de UMAP
            print("🔍 Validando datos para UMAP...")
            
            # Verificar que tenemos suficientes muestras para n_neighbors
            n_samples_available = len(volume_sample)
            if n_neighbors >= n_samples_available:
                n_neighbors_adjusted = max(2, n_samples_available - 1)
                print(f"⚠️  Ajustando n_neighbors de {n_neighbors} a {n_neighbors_adjusted} (muestras disponibles: {n_samples_available})")
                n_neighbors = n_neighbors_adjusted
            
            # Verificar varianza en los datos
            data_variance = np.var(volume_sample)
            if data_variance < 1e-10:
                print("⚠️  Varianza muy baja detectada, añadiendo más diversidad...")
                additional_noise = np.random.normal(0, 1e-5, volume_sample.shape)
                volume_sample = volume_sample + additional_noise
                print(f"✅ Varianza después de ajuste: {np.var(volume_sample):.2e}")
            
            # Normalizar datos para mejor estabilidad
            print("📊 Normalizando datos...")
            volume_mean = np.mean(volume_sample)
            volume_std = np.std(volume_sample)
            if volume_std > 0:
                volume_sample_normalized = (volume_sample - volume_mean) / volume_std
            else:
                volume_sample_normalized = volume_sample
            
            print(f"📈 Estadísticas normalizadas - Media: {np.mean(volume_sample_normalized):.3f}, Std: {np.std(volume_sample_normalized):.3f}")
            
            if self.use_gpu and CUML_AVAILABLE:
                print("🔥 Usando cuML GPU para UMAP")
                try:
                    # Convertir a GPU
                    volume_gpu = cp.asarray(volume_sample_normalized)
                    
                    # Configuración de parámetros basada en VRAM disponible
                    if gpu_memory_gb >= 12:
                        # GPU de alta gama - parámetros agresivos
                        n_epochs = 1000 if n_samples_available < 50000 else 1500
                        learning_rate = 1.5
                        spread = 2.0
                        print("🚀 Usando configuración agresiva para GPU de alta gama")
                    elif gpu_memory_gb >= 8:
                        # GPU de gama media - parámetros balanceados
                        n_epochs = 800 if n_samples_available < 30000 else 1200
                        learning_rate = 1.2
                        spread = 1.5
                        print("⚡ Usando configuración balanceada para GPU de gama media")
                    elif gpu_memory_gb >= 4:
                        # GPU básica - parámetros moderados
                        n_epochs = 500 if n_samples_available < 20000 else 800
                        learning_rate = 1.0
                        spread = 1.0
                        print("💻 Usando configuración moderada para GPU básica")
                    else:
                        # Configuración conservadora
                        n_epochs = 200 if n_samples_available < 10000 else 500
                        learning_rate = 1.0
                        spread = 1.0
                        print("🔧 Usando configuración conservadora")
                    
                    # Parámetros optimizados para cuML
                    reducer = cuUMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=max(min_dist, 0.01),
                        random_state=42,
                        verbose=True,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs,
                        spread=spread,
                        # Parámetros adicionales para mejor calidad con más VRAM
                        negative_sample_rate=5 if gpu_memory_gb >= 8 else 3,
                        transform_queue_size=8.0 if gpu_memory_gb >= 12 else 4.0,
                        # Optimizaciones de memoria
                        low_memory=False if gpu_memory_gb >= 8 else True
                    )
                    
                    print(f"🎯 Parámetros cuML: epochs={n_epochs}, lr={learning_rate}, spread={spread}")
                    embedding_sample = reducer.fit_transform(volume_gpu)
                    embedding_sample = cp.asnumpy(embedding_sample)
                    print("✅ UMAP GPU completado con configuración optimizada")
                    
                except Exception as gpu_error:
                    error_msg = str(gpu_error)
                    print(f"⚠️  Error en GPU: {error_msg[:200]}...")
                    
                    # Detectar si es error RAFT específico
                    is_raft_error = "RAFT failure" in error_msg or "non-zero distance" in error_msg
                    
                    if is_raft_error:
                        print("🔍 Error RAFT detectado - Aplicando estrategia de muestreo inteligente")
                        # Para errores RAFT, usar muestreo más agresivo desde el inicio
                        available_ram_gb = 12
                        raft_max_samples = min(25000, int(available_ram_gb * 2000))  # Más conservador para RAFT
                        
                        if len(volume_sample_normalized) > raft_max_samples:
                            print(f"🔧 Muestreo RAFT: {len(volume_sample_normalized):,} → {raft_max_samples:,} muestras")
                            raft_indices = np.random.choice(len(volume_sample_normalized), 
                                                          raft_max_samples, 
                                                          replace=False)
                            volume_raft_sample = volume_sample_normalized[raft_indices]
                        else:
                            volume_raft_sample = volume_sample_normalized
                            raft_indices = np.arange(len(volume_sample_normalized))
                    else:
                        volume_raft_sample = volume_sample_normalized
                        raft_indices = np.arange(len(volume_sample_normalized))
                    
                    print("🔄 Intentando fallback a CPU optimizado...")
                    
                    try:
                        # Parámetros optimizados para CPU con memoria limitada
                        cpu_n_neighbors = min(6, max(3, len(volume_raft_sample) // 1500))
                        
                        reducer = umap.UMAP(
                            n_components=n_components,
                            n_neighbors=cpu_n_neighbors,
                            min_dist=min_dist,
                            random_state=42,
                            verbose=True,
                            learning_rate=0.8,
                            n_epochs=150,
                            low_memory=True,
                            n_jobs=1
                        )
                        embedding_raft_sample = reducer.fit_transform(volume_raft_sample)
                        print("✅ UMAP CPU (fallback optimizado) completado")
                        
                        # Expandir embedding si se hizo muestreo RAFT
                        if len(volume_raft_sample) < len(volume_sample_normalized):
                            print("🔄 Expandiendo embedding desde muestreo RAFT...")
                            embedding_sample = np.zeros((len(volume_sample_normalized), n_components))
                            embedding_sample[raft_indices] = embedding_raft_sample
                            
                            # Interpolar valores faltantes
                            missing_mask = np.ones(len(volume_sample_normalized), dtype=bool)
                            missing_mask[raft_indices] = False
                            missing_indices = np.where(missing_mask)[0]
                            
                            if len(missing_indices) > 0:
                                for i in range(n_components):
                                    embedding_sample[missing_indices, i] = np.mean(embedding_raft_sample[:, i])
                        else:
                            embedding_sample = embedding_raft_sample
                            
                    except Exception as cpu_error:
                        print(f"❌ Error también en CPU: {str(cpu_error)[:200]}...")
                        print("🔧 Intentando con parámetros ultra-conservadores...")
                        
                        # Muestreo extremadamente agresivo para casos críticos
                        ultra_max_samples = min(8000, len(volume_raft_sample))
                        if len(volume_raft_sample) > ultra_max_samples:
                            ultra_indices = np.random.choice(len(volume_raft_sample), 
                                                           ultra_max_samples, 
                                                           replace=False)
                            volume_ultra_sample = volume_raft_sample[ultra_indices]
                        else:
                            volume_ultra_sample = volume_raft_sample
                            ultra_indices = np.arange(len(volume_raft_sample))
                        
                        print(f"📊 Muestras ultra-conservadoras: {len(volume_ultra_sample):,}")
                        
                        # Último intento con parámetros muy conservadores
                        reducer = umap.UMAP(
                            n_components=n_components,
                            n_neighbors=min(3, max(2, len(volume_ultra_sample) // 2500)),
                            min_dist=0.2,
                            random_state=42,
                            verbose=False,
                            learning_rate=0.5,
                            n_epochs=50,
                            low_memory=True,
                            n_jobs=1
                        )
                        embedding_ultra_sample = reducer.fit_transform(volume_ultra_sample)
                        print("✅ UMAP ultra-conservador completado")
                        
                        # Expandir embedding ultra-conservador con doble nivel
                        embedding_sample = np.zeros((len(volume_sample_normalized), n_components))
                        
                        if len(volume_ultra_sample) < len(volume_raft_sample):
                            # Primero expandir de ultra a raft
                            embedding_raft_expanded = np.zeros((len(volume_raft_sample), n_components))
                            embedding_raft_expanded[ultra_indices] = embedding_ultra_sample
                            
                            # Interpolar faltantes en nivel raft
                            missing_raft_mask = np.ones(len(volume_raft_sample), dtype=bool)
                            missing_raft_mask[ultra_indices] = False
                            missing_raft_indices = np.where(missing_raft_mask)[0]
                            
                            if len(missing_raft_indices) > 0:
                                for i in range(n_components):
                                    embedding_raft_expanded[missing_raft_indices, i] = np.mean(embedding_ultra_sample[:, i])
                            
                            # Luego expandir de raft a sample completo
                            if len(volume_raft_sample) < len(volume_sample_normalized):
                                embedding_sample[raft_indices] = embedding_raft_expanded
                                
                                missing_sample_mask = np.ones(len(volume_sample_normalized), dtype=bool)
                                missing_sample_mask[raft_indices] = False
                                missing_sample_indices = np.where(missing_sample_mask)[0]
                                
                                if len(missing_sample_indices) > 0:
                                    for i in range(n_components):
                                        embedding_sample[missing_sample_indices, i] = np.mean(embedding_raft_expanded[:, i])
                            else:
                                embedding_sample = embedding_raft_expanded
                        else:
                            embedding_sample = embedding_ultra_sample
            else:
                print("💻 Usando UMAP CPU con optimización de memoria")
                
                # Calcular muestreo agresivo para CPU para evitar agotamiento de RAM
                available_ram_gb = 12  # RAM disponible estimada
                max_samples_for_ram = min(50000, int(available_ram_gb * 3000))  # ~3000 muestras por GB
                
                if len(volume_sample_normalized) > max_samples_for_ram:
                    print(f"🔧 Reduciendo muestras de {len(volume_sample_normalized):,} a {max_samples_for_ram:,} para optimizar RAM")
                    # Muestreo adicional para CPU
                    cpu_indices = np.random.choice(len(volume_sample_normalized), 
                                                 max_samples_for_ram, 
                                                 replace=False)
                    volume_cpu_sample = volume_sample_normalized[cpu_indices]
                    cpu_sample_indices = indices[cpu_indices]  # Mantener referencia a índices originales
                else:
                    volume_cpu_sample = volume_sample_normalized
                    cpu_sample_indices = indices
                
                print(f"📊 Muestras para CPU: {len(volume_cpu_sample):,}")
                
                try:
                    # Parámetros optimizados para CPU con memoria limitada
                    cpu_n_neighbors = min(8, max(3, len(volume_cpu_sample) // 1000))
                    
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=cpu_n_neighbors,
                        min_dist=min_dist,
                        random_state=42,
                        verbose=True,
                        learning_rate=0.8,
                        n_epochs=150,
                        low_memory=True,  # Activar modo de baja memoria
                        n_jobs=1  # Un solo hilo para controlar memoria
                    )
                    embedding_cpu_sample = reducer.fit_transform(volume_cpu_sample)
                    print("✅ UMAP CPU completado")
                    
                    # Si se hizo muestreo adicional, expandir el embedding
                    if len(volume_cpu_sample) < len(volume_sample_normalized):
                        print("🔄 Expandiendo embedding CPU...")
                        embedding_sample = np.zeros((len(volume_sample_normalized), n_components))
                        embedding_sample[cpu_indices] = embedding_cpu_sample
                        
                        # Interpolar valores faltantes con promedio
                        missing_mask = np.ones(len(volume_sample_normalized), dtype=bool)
                        missing_mask[cpu_indices] = False
                        missing_indices = np.where(missing_mask)[0]
                        
                        if len(missing_indices) > 0:
                            for i in range(n_components):
                                embedding_sample[missing_indices, i] = np.mean(embedding_cpu_sample[:, i])
                    else:
                        embedding_sample = embedding_cpu_sample
                        
                except Exception as cpu_error:
                    print(f"⚠️  Error en CPU: {str(cpu_error)[:200]}...")
                    print("🔧 Intentando con parámetros ultra-conservadores...")
                    
                    # Muestreo extremadamente agresivo para casos críticos
                    ultra_max_samples = min(10000, len(volume_cpu_sample))
                    if len(volume_cpu_sample) > ultra_max_samples:
                        ultra_indices = np.random.choice(len(volume_cpu_sample), 
                                                       ultra_max_samples, 
                                                       replace=False)
                        volume_ultra_sample = volume_cpu_sample[ultra_indices]
                    else:
                        volume_ultra_sample = volume_cpu_sample
                        ultra_indices = np.arange(len(volume_cpu_sample))
                    
                    print(f"📊 Muestras ultra-conservadoras: {len(volume_ultra_sample):,}")
                    
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=min(3, max(2, len(volume_ultra_sample) // 2000)),
                        min_dist=0.2,
                        random_state=42,
                        verbose=False,
                        learning_rate=0.5,
                        n_epochs=50,
                        low_memory=True,
                        n_jobs=1
                    )
                    embedding_ultra_sample = reducer.fit_transform(volume_ultra_sample)
                    
                    # Expandir embedding ultra-conservador
                    embedding_sample = np.zeros((len(volume_sample_normalized), n_components))
                    if len(volume_cpu_sample) > ultra_max_samples:
                        # Primero expandir de ultra a cpu
                        embedding_cpu_expanded = np.zeros((len(volume_cpu_sample), n_components))
                        embedding_cpu_expanded[ultra_indices] = embedding_ultra_sample
                        
                        # Interpolar faltantes en nivel CPU
                        missing_cpu_mask = np.ones(len(volume_cpu_sample), dtype=bool)
                        missing_cpu_mask[ultra_indices] = False
                        missing_cpu_indices = np.where(missing_cpu_mask)[0]
                        
                        if len(missing_cpu_indices) > 0:
                            for i in range(n_components):
                                embedding_cpu_expanded[missing_cpu_indices, i] = np.mean(embedding_ultra_sample[:, i])
                        
                        # Luego expandir de cpu a sample completo
                        if len(volume_cpu_sample) < len(volume_sample_normalized):
                            embedding_sample[cpu_indices] = embedding_cpu_expanded
                            
                            missing_sample_mask = np.ones(len(volume_sample_normalized), dtype=bool)
                            missing_sample_mask[cpu_indices] = False
                            missing_sample_indices = np.where(missing_sample_mask)[0]
                            
                            if len(missing_sample_indices) > 0:
                                for i in range(n_components):
                                    embedding_sample[missing_sample_indices, i] = np.mean(embedding_cpu_expanded[:, i])
                        else:
                            embedding_sample = embedding_cpu_expanded
                    else:
                        embedding_sample = embedding_ultra_sample
                    
                    print("✅ UMAP ultra-conservador completado")
            
            umap_time = time.time() - umap_start
            print(f"⏱️  UMAP: {umap_time:.1f}s")
            
            # 🚨 GESTIÓN CRÍTICA DE MEMORIA - LIBERAR INMEDIATAMENTE
            print("🧹 Liberando memoria intermedia...")
            
            # Liberar variables grandes inmediatamente
            del volume_flat_noisy, volume_sample_normalized
            if 'volume_gpu' in locals():
                del volume_gpu
            if 'volume_raft_sample' in locals():
                del volume_raft_sample
            if 'volume_cpu_sample' in locals():
                del volume_cpu_sample
            if 'volume_ultra_sample' in locals():
                del volume_ultra_sample
            
            # Forzar garbage collection
            import gc
            gc.collect()
            
            # Si hay GPU, limpiar memoria GPU también
            if self.use_gpu and CUML_AVAILABLE:
                try:
                    import cupy as cp
                    mempool = cp.get_default_memory_pool()
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool.free_all_blocks()
                    print("🔥 Memoria GPU liberada")
                except:
                    pass
            
            # Verificar memoria disponible
            try:
                import psutil
                available_ram = psutil.virtual_memory().available / (1024**3)
                used_ram = psutil.virtual_memory().used / (1024**3)
                print(f"💾 RAM disponible: {available_ram:.1f} GB, usada: {used_ram:.1f} GB")
                
                # Si queda poca RAM, usar estrategia ultra-conservadora
                if available_ram < 2.0:  # Menos de 2 GB disponibles
                    print("🚨 MEMORIA CRÍTICA - Usando estrategia de emergencia")
                    # Guardar solo el embedding muestreado, sin reconstruir completo
                    self.umap_embedding = embedding_sample
                    self.umap_indices = indices if sample_ratio < 1.0 else None
                    
                    total_time = time.time() - start_time
                    print(f"⏱️  Tiempo total: {total_time:.1f}s")
                    print("⚠️  Embedding guardado en modo conservador (solo muestras)")
                    return embedding_sample
                    
            except ImportError:
                print("⚠️  psutil no disponible, continuando con precaución")
            
            # Reconstruir embedding completo solo si hay suficiente memoria
            recon_start = time.time()
            print("🔄 Reconstruyendo embedding completo...")
            
            if sample_ratio < 1.0:
                # Usar estrategia de memoria eficiente para reconstrucción
                print("🔗 Interpolando valores faltantes (modo eficiente)...")
                
                # Crear embedding completo por chunks para evitar picos de memoria
                chunk_size = 100000  # Procesar en chunks de 100k voxels
                embedding_full = np.zeros((len(volume_flat), n_components))
                
                # Asignar valores muestreados
                embedding_full[indices] = embedding_sample
                
                # Para interpolación, usar promedio simple en lugar de vecinos cercanos
                # (más eficiente en memoria)
                missing_mask = np.ones(len(volume_flat), dtype=bool)
                missing_mask[indices] = False
                missing_indices = np.where(missing_mask)[0]
                
                if len(missing_indices) > 0:
                    print(f"📊 Interpolando {len(missing_indices):,} valores faltantes...")
                    # Usar promedio por componente (muy eficiente en memoria)
                    for i in range(n_components):
                        embedding_full[missing_indices, i] = np.mean(embedding_sample[:, i])
                
                # Liberar memoria del embedding muestreado
                del embedding_sample
                gc.collect()
                
                embedding = embedding_full
            else:
                embedding = embedding_sample
            
            # Liberar memoria del volumen original procesado
            del volume_flat
            gc.collect()
            
            # Reshape al formato original
            embedding_volume = embedding.reshape(self.volume.shape + (n_components,))
            
            # Liberar embedding temporal
            del embedding
            gc.collect()
            
            recon_time = time.time() - recon_start
            total_time = time.time() - start_time
            
            print(f"⏱️  Reconstrucción: {recon_time:.1f}s")
            print(f"⏱️  Tiempo total: {total_time:.1f}s")
            
            # Verificar memoria final y usar función de limpieza
            self._force_memory_cleanup()
            
            self.features['umap_embedding'] = embedding_volume
            print("✅ UMAP embedding calculado exitosamente")
            print(f"📊 Forma final: {embedding_volume.shape}")
            
            return embedding_volume
            
        except Exception as e:
            print(f"❌ Error calculando UMAP: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    def visualize_features(self, feature_name: str, slice_idx: Optional[int] = None):
        """
        Visualizar características calculadas
        
        Parámetros:
        -----------
        feature_name : str
            Nombre de la característica a visualizar
        slice_idx : int, opcional
            Índice del slice a mostrar
        """
        if feature_name not in self.features:
            print(f"❌ Característica '{feature_name}' no encontrada")
            print(f"Disponibles: {list(self.features.keys())}")
            return
        
        feature_data = self.features[feature_name]
        
        if slice_idx is None:
            slice_idx = feature_data.shape[2] // 2 if len(feature_data.shape) >= 3 else 0
        
        if isinstance(feature_data, dict):
            # Múltiples componentes (ej: gradiente)
            n_components = len(feature_data)
            fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 4))
            
            if n_components == 1:
                axes = [axes]
            
            for i, (comp_name, comp_data) in enumerate(feature_data.items()):
                if len(comp_data.shape) >= 3:
                    slice_data = comp_data[:, :, slice_idx]
                else:
                    slice_data = comp_data
                
                im = axes[i].imshow(slice_data, cmap='viridis', origin='lower')
                axes[i].set_title(f'{comp_name}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        else:
            # Componente único
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if len(feature_data.shape) >= 3:
                slice_data = feature_data[:, :, slice_idx]
            else:
                slice_data = feature_data
            
            im = ax.imshow(slice_data, cmap='viridis', origin='lower')
            ax.set_title(f'{feature_name} - Slice {slice_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    def create_3d_visualization(self, feature_name: str = 'original', 
                               threshold_percentile: float = 90,
                               render_type: str = 'scatter',
                               max_points: int = 15000):
        """
        Crear visualización 3D interactiva con Plotly
        OPTIMIZADO PARA EVITAR AGOTAMIENTO DE MEMORIA
        
        Parámetros:
        -----------
        feature_name : str
            Característica a visualizar ('original' para volumen original)
        threshold_percentile : float
            Percentil para umbralización
        render_type : str
            Tipo de render ('scatter', 'volume', 'isosurface')
        max_points : int
            Número máximo de puntos para render scatter
        """
        if self.volume is None:
            print("❌ No hay volumen cargado")
            return
        
        # 🚨 VERIFICAR MEMORIA DISPONIBLE ANTES DE CONTINUAR
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024**3)
            print(f"💾 RAM disponible antes del render: {available_ram:.1f} GB")
            
            if available_ram < 1.5:  # Menos de 1.5 GB disponibles
                print("🚨 MEMORIA INSUFICIENTE - Usando configuración de emergencia")
                max_points = min(max_points, 5000)  # Reducir puntos drásticamente
                render_type = 'scatter'  # Forzar scatter (más eficiente)
                
        except ImportError:
            print("⚠️  psutil no disponible, continuando con precaución")
        
        # Seleccionar datos con gestión de memoria
        if feature_name == 'original':
            data = self.volume
            title_suffix = "Volumen Original"
        elif feature_name in self.features:
            feature_data = self.features[feature_name]
            if isinstance(feature_data, dict):
                if 'magnitude' in feature_data:
                    data = feature_data['magnitude']  # Usar magnitud si es gradiente
                    title_suffix = f"{feature_name.title()} (Magnitud)"
                else:
                    # Usar el primer componente disponible
                    first_key = list(feature_data.keys())[0]
                    data = feature_data[first_key]
                    title_suffix = f"{feature_name.title()} ({first_key})"
            else:
                data = feature_data
                title_suffix = feature_name.title()
        else:
            print(f"❌ Característica '{feature_name}' no encontrada")
            return
        
        print(f"🔄 Creando visualización 3D de '{title_suffix}'...")
        
        # 🧹 Gestión de memoria: Crear copia mínima necesaria
        import gc
        
        if render_type == 'volume':
            # Render volumétrico usando go.Volume
            try:
                # Submuestrear agresivamente para rendimiento y memoria
                step = max(2, data.shape[0] // 32)  # Más agresivo
                subsampled = data[::step, ::step, ::step].copy()
                
                # Liberar referencia original inmediatamente
                del data
                gc.collect()
                
                threshold = np.percentile(subsampled, threshold_percentile)
                
                fig = go.Figure(data=go.Volume(
                    x=np.arange(0, subsampled.shape[0]),
                    y=np.arange(0, subsampled.shape[1]),
                    z=np.arange(0, subsampled.shape[2]),
                    value=subsampled.flatten(),
                    isomin=threshold,
                    isomax=subsampled.max(),
                    opacity=0.1,
                    surface_count=10,  # Reducido para memoria
                    colorscale='Viridis'
                ))
                
                # Liberar subsampled inmediatamente
                del subsampled
                gc.collect()
                
                fig.update_layout(
                    title=f'Render Volumétrico 3D: {title_suffix}',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                        bgcolor='black'
                    ),
                    width=900,
                    height=700
                )
                
            except Exception as e:
                print(f"⚠️ Error en render volumétrico: {e}")
                print("🔄 Cambiando a render scatter...")
                render_type = 'scatter'
        
        if render_type == 'isosurface':
            # Render de isosuperficie
            try:
                # Submuestrear para memoria
                if data.size > 1000000:  # Si es muy grande
                    step = max(1, data.shape[0] // 64)
                    data_iso = data[::step, ::step, ::step].copy()
                    del data
                    gc.collect()
                else:
                    data_iso = data
                
                threshold = np.percentile(data_iso, threshold_percentile)
                
                fig = go.Figure(data=go.Isosurface(
                    x=np.arange(data_iso.shape[0]),
                    y=np.arange(data_iso.shape[1]),
                    z=np.arange(data_iso.shape[2]),
                    value=data_iso,
                    isomin=threshold,
                    isomax=data_iso.max(),
                    surface_count=2,  # Reducido para memoria
                    colorscale='Viridis',
                    caps=dict(x_show=False, y_show=False, z_show=False)
                ))
                
                # Liberar data_iso inmediatamente
                del data_iso
                gc.collect()
                
                fig.update_layout(
                    title=f'Isosuperficie 3D: {title_suffix}',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        bgcolor='black'
                    ),
                    width=900,
                    height=700
                )
                
            except Exception as e:
                print(f"⚠️ Error en isosuperficie: {e}")
                print("🔄 Cambiando a render scatter...")
                render_type = 'scatter'
        
        if render_type == 'scatter':
            # Render scatter (por defecto) - OPTIMIZADO PARA MEMORIA
            
            # Umbralización para reducir datos ANTES de extraer coordenadas
            threshold = np.percentile(data, threshold_percentile)
            mask = data > threshold
            
            # Contar puntos válidos antes de extraer
            n_valid_points = np.sum(mask)
            print(f"📊 Puntos válidos encontrados: {n_valid_points:,}")
            
            if n_valid_points == 0:
                print("⚠️ No hay puntos válidos con el umbral especificado")
                return
            
            # Limitar desde el principio si hay demasiados puntos
            if n_valid_points > max_points:
                print(f"🔧 Limitando a {max_points:,} puntos para optimizar memoria")
                # Muestreo directo de la máscara para evitar crear arrays grandes
                valid_indices = np.where(mask)
                selected_indices = np.random.choice(len(valid_indices[0]), max_points, replace=False)
                
                coords = (
                    valid_indices[0][selected_indices],
                    valid_indices[1][selected_indices], 
                    valid_indices[2][selected_indices]
                )
                values = data[coords]
                
                # Liberar variables intermedias
                del valid_indices, selected_indices, mask
                gc.collect()
            else:
                # Obtener coordenadas de voxels significativos
                coords = np.where(mask)
                values = data[mask]
                
                # Liberar mask inmediatamente
                del mask
                gc.collect()
            
            # Liberar data original
            del data
            gc.collect()
            
            # Crear gráfico 3D scatter con configuración optimizada
            fig = go.Figure(data=go.Scatter3d(
                x=coords[0],
                y=coords[1],
                z=coords[2],
                mode='markers',
                marker=dict(
                    size=2,  # Tamaño reducido para memoria
                    color=values,
                    colorscale='Viridis',
                    opacity=0.7,  # Reducido para rendimiento
                    colorbar=dict(title=title_suffix),
                    line=dict(width=0)
                ),
                # Simplificar hover para memoria
                hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
            ))
            
            # Liberar coords y values
            del coords, values
            gc.collect()
            
            fig.update_layout(
                title=f'Visualización 3D: {title_suffix}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    bgcolor='black'
                ),
                width=900,
                height=700
            )
        
        # Verificar memoria antes de mostrar
        try:
            import psutil
            available_ram_after = psutil.virtual_memory().available / (1024**3)
            print(f"💾 RAM disponible después del procesamiento: {available_ram_after:.1f} GB")
        except:
            pass
        
        # Mostrar figura
        fig.show()
        print(f"✅ Visualización 3D creada ({render_type})")
        print(f"📊 Puntos mostrados: {max_points if render_type == 'scatter' and n_valid_points > max_points else (n_valid_points if render_type == 'scatter' else 'Volumétrico')}")
        print(f"🎯 Umbral: {threshold_percentile}% percentil")
        
        # Liberar figura de memoria local y forzar limpieza
        del fig
        gc.collect()
        
        # Usar función auxiliar de limpieza después del render
        self._force_memory_cleanup()
    
    def _force_memory_cleanup(self):
        """
        Función auxiliar para forzar limpieza de memoria
        """
        import gc
        
        # Forzar garbage collection múltiple
        for _ in range(3):
            gc.collect()
        
        # Limpiar cache de GPU si está disponible
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            print("🧹 Cache GPU liberado")
        except:
            pass
        
        # Verificar memoria disponible
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024**3)
            print(f"💾 RAM disponible después de limpieza: {available_ram:.1f} GB")
        except:
            pass
    
    def generate_report(self) -> str:
        """
        Generar reporte completo del análisis
        
        Retorna:
        --------
        str : Reporte en formato HTML
        """
        if self.volume is None:
            return "<p>❌ No hay volumen cargado</p>"
        
        html_report = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px;">
            <h1>🏥 Reporte Medico3D</h1>
            <hr>
            
            <h2>📋 Información del Volumen</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td><b>Formato:</b></td><td>{self.volume_info.get('format', 'N/A')}</td></tr>
                <tr><td><b>Dimensiones:</b></td><td>{self.volume_info.get('shape', 'N/A')}</td></tr>
                <tr><td><b>Tipo de datos:</b></td><td>{self.volume_info.get('data_type', 'N/A')}</td></tr>
                <tr><td><b>Rango de valores:</b></td><td>[{self.volume_info.get('min_value', 0):.4f}, {self.volume_info.get('max_value', 0):.4f}]</td></tr>
                <tr><td><b>Valor medio:</b></td><td>{self.volume_info.get('mean_value', 0):.4f}</td></tr>
                <tr><td><b>Desviación estándar:</b></td><td>{self.volume_info.get('std_value', 0):.4f}</td></tr>
            </table>
            
            <h2>🔬 Características Calculadas</h2>
            <ul>
        """
        
        for feature_name, feature_data in self.features.items():
            if isinstance(feature_data, dict):
                components = list(feature_data.keys())
                html_report += f"<li><b>{feature_name}:</b> {len(components)} componentes ({', '.join(components)})</li>"
            else:
                html_report += f"<li><b>{feature_name}:</b> {feature_data.shape}</li>"
        
        html_report += """
            </ul>
            
            <h2>⚙️ Configuración</h2>
            <ul>
                <li><b>Procesamiento:</b> """ + ("GPU (CUDA)" if self.use_gpu else "CPU") + """</li>
                <li><b>Versión:</b> Medico3D Colab 2.0.0</li>
            </ul>
            
            <hr>
            <p><i>Reporte generado automáticamente por Medico3D</i></p>
        </div>
        """
        
        return html_report
    
    def compare_features(self):
        """
        Comparar diferentes características calculadas
        """
        if not self.features:
            print("❌ No hay características calculadas para comparar")
            print("💡 Ejecuta primero los métodos de cálculo de características")
            return
        
        print("📊 COMPARACIÓN DE CARACTERÍSTICAS")
        print("=" * 50)
        
        # Crear figura con subplots
        available_features = []
        feature_data = []
        
        # Recopilar características disponibles
        if 'gradient' in self.features and 'magnitude' in self.features['gradient']:
            available_features.append('Gradiente (Magnitud)')
            feature_data.append(self.features['gradient']['magnitude'])
        
        if 'laplacian' in self.features:
            available_features.append('Laplaciano')
            feature_data.append(self.features['laplacian'])
        
        if 'statistical' in self.features:
            if 'mean' in self.features['statistical']:
                available_features.append('Media Local')
                feature_data.append(self.features['statistical']['mean'])
            if 'std' in self.features['statistical']:
                available_features.append('Desviación Estándar')
                feature_data.append(self.features['statistical']['std'])
        
        if 'gaussian_filtered' in self.features:
            available_features.append('Filtro Gaussiano')
            feature_data.append(self.features['gaussian_filtered'])
        
        if not available_features:
            print("❌ No hay características válidas para comparar")
            return
        
        # Crear visualización comparativa
        n_features = len(available_features)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Normalizar axes para que siempre sea indexable
        if n_features == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = axes.flatten()
        elif rows > 1 and cols > 1:
            axes = axes.flatten()
        
        # Slice central para visualización
        if self.volume is not None:
            central_slice = self.volume.shape[2] // 2
        else:
            central_slice = feature_data[0].shape[2] // 2
        
        for i, (name, data) in enumerate(zip(available_features, feature_data)):
            ax = axes[i]
            
            # Mostrar slice central
            slice_data = data[:, :, central_slice]
            im = ax.imshow(slice_data, cmap='viridis', origin='lower')
            ax.set_title(f'{name}\nSlice {central_slice}')
            ax.axis('off')
            
            # Añadir colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Ocultar axes vacíos
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas comparativas
        print("\n📈 ESTADÍSTICAS COMPARATIVAS")
        print("-" * 50)
        
        stats_df = []
        for name, data in zip(available_features, feature_data):
            stats_df.append({
                'Característica': name,
                'Forma': str(data.shape),
                'Mínimo': f"{np.min(data):.6f}",
                'Máximo': f"{np.max(data):.6f}",
                'Media': f"{np.mean(data):.6f}",
                'Std': f"{np.std(data):.6f}"
            })
        
        stats_table = pd.DataFrame(stats_df)
        display(stats_table)
        
        # Histogramas comparativos
        print("\n📊 DISTRIBUCIÓN DE VALORES")
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for name, data in zip(available_features, feature_data):
            # Muestrear datos para histograma (evitar sobrecarga)
            sample_size = min(50000, data.size)
            sample_data = np.random.choice(data.flatten(), sample_size, replace=False)
            
            ax.hist(sample_data, bins=50, alpha=0.6, label=name, density=True)
        
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidad')
        ax.set_title('Distribución de Valores por Característica')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Comparación de características completada")
    
    def statistical_analysis(self):
        """
        Realizar análisis estadístico completo de las características
        """
        if not self.features:
            print("❌ No hay características calculadas para analizar")
            print("💡 Ejecuta primero los métodos de cálculo de características")
            return
        
        print("📈 ANÁLISIS ESTADÍSTICO COMPLETO")
        print("=" * 50)
        
        # Análisis del volumen original
        if self.volume is not None:
            print("\n🔍 ANÁLISIS DEL VOLUMEN ORIGINAL")
            print("-" * 30)
            
            vol_stats = {
                'Dimensiones': self.volume.shape,
                'Tamaño total': self.volume.size,
                'Tipo de datos': str(self.volume.dtype),
                'Memoria (MB)': f"{self.volume.nbytes / (1024**2):.2f}",
                'Valor mínimo': f"{np.min(self.volume):.6f}",
                'Valor máximo': f"{np.max(self.volume):.6f}",
                'Media': f"{np.mean(self.volume):.6f}",
                'Mediana': f"{np.median(self.volume):.6f}",
                'Desviación estándar': f"{np.std(self.volume):.6f}",
                'Varianza': f"{np.var(self.volume):.6f}",
                'Asimetría': f"{scipy_stats.skew(self.volume.flatten()):.6f}",
                 'Curtosis': f"{scipy_stats.kurtosis(self.volume.flatten()):.6f}"
            }
            
            for key, value in vol_stats.items():
                print(f"  {key}: {value}")
        
        # Análisis por característica
        print("\n🔬 ANÁLISIS POR CARACTERÍSTICA")
        print("-" * 30)
        
        all_stats = []
        
        for feature_name, feature_data in self.features.items():
            print(f"\n📊 {feature_name.upper()}")
            
            if isinstance(feature_data, dict):
                # Característica con múltiples componentes
                for comp_name, comp_data in feature_data.items():
                    print(f"  └─ {comp_name}:")
                    
                    stats = self._calculate_detailed_stats(comp_data, f"{feature_name}_{comp_name}")
                    all_stats.append(stats)
                    
                    for key, value in stats.items():
                        if key != 'Nombre':
                            print(f"     {key}: {value}")
            else:
                # Característica simple
                print(f"  Datos:")
                
                stats = self._calculate_detailed_stats(feature_data, feature_name)
                all_stats.append(stats)
                
                for key, value in stats.items():
                    if key != 'Nombre':
                        print(f"     {key}: {value}")
        
        # Tabla resumen
        if all_stats:
            print("\n📋 TABLA RESUMEN")
            print("-" * 30)
            
            summary_df = pd.DataFrame(all_stats)
            display(summary_df)
        
        # Análisis de correlaciones (si hay múltiples características)
        if len(self.features) > 1:
            print("\n🔗 ANÁLISIS DE CORRELACIONES")
            print("-" * 30)
            
            self._correlation_analysis()
        
        # Análisis de distribuciones
        print("\n📊 ANÁLISIS DE DISTRIBUCIONES")
        print("-" * 30)
        
        self._distribution_analysis()
        
        print("\n✅ Análisis estadístico completado")
    
    def _calculate_detailed_stats(self, data: np.ndarray, name: str) -> Dict[str, Any]:
        """Calcular estadísticas detalladas para un array"""
        flat_data = data.flatten()
        
        # Muestrear si es muy grande
        if flat_data.size > 1000000:
            sample_size = 1000000
            flat_data = np.random.choice(flat_data, sample_size, replace=False)
        
        return {
            'Nombre': name,
            'Forma': str(data.shape),
            'Tamaño': data.size,
            'Memoria (MB)': f"{data.nbytes / (1024**2):.2f}",
            'Mínimo': f"{np.min(flat_data):.6f}",
            'Máximo': f"{np.max(flat_data):.6f}",
            'Media': f"{np.mean(flat_data):.6f}",
            'Mediana': f"{np.median(flat_data):.6f}",
            'Std': f"{np.std(flat_data):.6f}",
            'Varianza': f"{np.var(flat_data):.6f}",
            'Percentil 25': f"{np.percentile(flat_data, 25):.6f}",
            'Percentil 75': f"{np.percentile(flat_data, 75):.6f}",
            'Asimetría': f"{scipy_stats.skew(flat_data):.6f}",
             'Curtosis': f"{scipy_stats.kurtosis(flat_data):.6f}"
        }
    
    def _correlation_analysis(self):
        """Análisis de correlaciones entre características"""
        try:
            # Recopilar datos para correlación
            correlation_data = {}
            
            # Volumen original
            if self.volume is not None:
                # Muestrear volumen original
                sample_size = min(10000, self.volume.size)
                vol_sample = np.random.choice(self.volume.flatten(), sample_size, replace=False)
                correlation_data['Original'] = vol_sample
            
            # Características
            for feature_name, feature_data in self.features.items():
                if isinstance(feature_data, dict):
                    for comp_name, comp_data in feature_data.items():
                        sample_size = min(10000, comp_data.size)
                        comp_sample = np.random.choice(comp_data.flatten(), sample_size, replace=False)
                        correlation_data[f"{feature_name}_{comp_name}"] = comp_sample
                else:
                    sample_size = min(10000, feature_data.size)
                    feat_sample = np.random.choice(feature_data.flatten(), sample_size, replace=False)
                    correlation_data[feature_name] = feat_sample
            
            if len(correlation_data) > 1:
                # Crear DataFrame para correlaciones
                min_length = min(len(data) for data in correlation_data.values())
                corr_df = pd.DataFrame({
                    name: data[:min_length] for name, data in correlation_data.items()
                })
                
                # Calcular matriz de correlación
                corr_matrix = corr_df.corr()
                
                # Visualizar matriz de correlación
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.3f')
                plt.title('Matriz de Correlación entre Características')
                plt.tight_layout()
                plt.show()
                
                # Mostrar correlaciones más altas
                print("🔗 Correlaciones más significativas:")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.1:  # Solo correlaciones significativas
                            corr_pairs.append({
                                'Característica 1': corr_matrix.columns[i],
                                'Característica 2': corr_matrix.columns[j],
                                'Correlación': f"{corr_val:.3f}"
                            })
                
                if corr_pairs:
                    corr_df_display = pd.DataFrame(corr_pairs)
                    corr_df_display = corr_df_display.sort_values('Correlación', key=lambda x: abs(x.astype(float)), ascending=False)
                    display(corr_df_display.head(10))
                else:
                    print("  No se encontraron correlaciones significativas (>0.1)")
        
        except Exception as e:
            print(f"❌ Error en análisis de correlaciones: {e}")
    
    def _distribution_analysis(self):
        """Análisis de distribuciones de las características"""
        try:
            # Crear gráficos de distribución
            available_features = []
            feature_data = []
            
            # Recopilar características para análisis
            for feature_name, data in self.features.items():
                if isinstance(data, dict):
                    for comp_name, comp_data in data.items():
                        available_features.append(f"{feature_name}_{comp_name}")
                        # Muestrear datos
                        sample_size = min(50000, comp_data.size)
                        sample_data = np.random.choice(comp_data.flatten(), sample_size, replace=False)
                        feature_data.append(sample_data)
                else:
                    available_features.append(feature_name)
                    # Muestrear datos
                    sample_size = min(50000, data.size)
                    sample_data = np.random.choice(data.flatten(), sample_size, replace=False)
                    feature_data.append(sample_data)
            
            if not available_features:
                return
            
            # Crear subplots para distribuciones
            n_features = len(available_features)
            cols = min(3, n_features)
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            
            # Normalizar axes para que siempre sea indexable
            if n_features == 1:
                axes = [axes]
            elif rows == 1 and cols > 1:
                axes = axes.flatten()
            elif rows > 1 and cols > 1:
                axes = axes.flatten()
            
            for i, (name, data) in enumerate(zip(available_features, feature_data)):
                ax = axes[i]
                
                # Histograma
                ax.hist(data, bins=50, alpha=0.7, density=True, color='skyblue')
                ax.set_title(f'Distribución: {name}')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Densidad')
                ax.grid(True, alpha=0.3)
                
                # Añadir estadísticas básicas
                mean_val = np.mean(data)
                std_val = np.std(data)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ')
                ax.legend(fontsize=8)
            
            # Ocultar axes vacíos
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error en análisis de distribuciones: {e}")
    
    def save_features(self, output_dir: str = None):
        """
        Guardar características calculadas
        
        Parámetros:
        -----------
        output_dir : str
            Directorio de salida (por defecto: /content/medico3d_output en Colab, 
            ./medico3d_output en local)
        """
        if not self.features:
            print("❌ No hay características para guardar")
            return
        
        # Determinar directorio de salida
        if output_dir is None:
            if os.path.exists("/content"):
                # Estamos en Google Colab
                output_dir = "/content/medico3d_output"
            else:
                # Estamos en entorno local
                output_dir = os.path.join(os.getcwd(), "medico3d_output")
        
        # Crear directorio
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Guardando características en: {output_dir}")
        
        for feature_name, feature_data in self.features.items():
            try:
                if isinstance(feature_data, dict):
                    # Múltiples componentes
                    for comp_name, comp_data in feature_data.items():
                        filename = f"{feature_name}_{comp_name}.npy"
                        filepath = os.path.join(output_dir, filename)
                        np.save(filepath, comp_data)
                        print(f"✅ {filename}")
                else:
                    # Componente único
                    filename = f"{feature_name}.npy"
                    filepath = os.path.join(output_dir, filename)
                    np.save(filepath, feature_data)
                    print(f"✅ {filename}")
            
            except Exception as e:
                print(f"❌ Error guardando {feature_name}: {e}")
        
        # Guardar reporte
        try:
            report_html = self.generate_report()
            report_path = os.path.join(output_dir, "reporte.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            print(f"✅ reporte.html")
        except Exception as e:
            print(f"❌ Error guardando reporte: {e}")

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_sample_data():
    """Crear datos de ejemplo para pruebas"""
    print("🔧 Creando datos de ejemplo...")
    
    # Crear volumen sintético
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, 64),
        np.linspace(-2, 2, 64),
        np.linspace(-2, 2, 64)
    )
    
    # Esferas con diferentes intensidades
    sphere1 = np.exp(-(x**2 + y**2 + z**2))
    sphere2 = 0.5 * np.exp(-((x-1)**2 + (y-1)**2 + (z-1)**2) * 2)
    sphere3 = 0.3 * np.exp(-((x+1)**2 + (y+1)**2 + (z+1)**2) * 3)
    
    volume = sphere1 + sphere2 + sphere3
    
    # Añadir ruido
    volume += 0.1 * np.random.random(volume.shape)
    
    # Determinar ruta de salida (Colab vs local)
    if os.path.exists("/content"):
        # Estamos en Google Colab
        output_path = "/content/sample_volume.nii.gz"
    else:
        # Estamos en entorno local
        output_path = os.path.join(os.getcwd(), "sample_volume.nii.gz")
    
    # Guardar como archivo NIfTI
    img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(img, output_path)
    
    print(f"✅ Datos de ejemplo creados: {output_path}")
    return output_path

def quick_demo():
    """Demostración rápida de Medico3D"""
    print("🚀 DEMO RÁPIDO DE MEDICO3D")
    print("=" * 40)
    
    # Crear instancia
    medico = Medico3DColab()
    
    # Crear datos de ejemplo
    sample_file = create_sample_data()
    
    # Cargar volumen
    medico.load_volume(sample_file)
    
    # Mostrar información
    medico.show_volume_info()
    
    # Calcular características
    print("\n🔬 Calculando características...")
    medico.compute_gradient()
    medico.compute_laplacian()
    medico.compute_statistical_features()
    
    # Visualizar
    print("\n📊 Creando visualizaciones...")
    medico.visualize_slices()
    medico.visualize_features('gradient')
    
    # Crear visualización 3D
    medico.create_3d_visualization('original')
    
    # Mostrar reporte
    print("\n📋 Generando reporte...")
    report = medico.generate_report()
    display(HTML(report))
    
    # Guardar resultados
    medico.save_features()
    
    print("\n✅ Demo completado!")
    return medico

# ============================================================================
# INTERFAZ INTERACTIVA
# ============================================================================

def create_interactive_interface():
    """Crear interfaz interactiva completa"""
    
    # Crear instancia global
    global medico3d_instance
    medico3d_instance = Medico3DColab()
    
    # Widgets de control
    file_upload = widgets.FileUpload(
        accept='.nii,.nii.gz,.mha,.mhd',
        multiple=False,
        description='Cargar archivo'
    )
    
    load_sample_btn = widgets.Button(
        description='Usar datos de ejemplo',
        button_style='info'
    )
    
    info_btn = widgets.Button(
        description='Mostrar información',
        button_style='primary'
    )
    
    # Widgets de procesamiento
    gradient_btn = widgets.Button(description='Calcular Gradiente', button_style='success')
    laplacian_btn = widgets.Button(description='Calcular Laplaciano', button_style='success')
    stats_btn = widgets.Button(description='Características Estadísticas', button_style='success')
    gaussian_btn = widgets.Button(description='Filtro Gaussiano', button_style='success')
    umap_btn = widgets.Button(description='UMAP Embedding', button_style='success')
    
    # Widgets de visualización
    view_slices_btn = widgets.Button(description='Ver Slices', button_style='warning')
    view_3d_btn = widgets.Button(description='Vista 3D', button_style='warning')
    interactive_viewer_btn = widgets.Button(description='Visor Interactivo', button_style='warning')
    
    # Widgets de salida
    generate_report_btn = widgets.Button(description='Generar Reporte', button_style='danger')
    save_features_btn = widgets.Button(description='Guardar Características', button_style='danger')
    
    output = widgets.Output()
    
    # Funciones de callback
    def on_load_sample(b):
        with output:
            clear_output()
            sample_file = create_sample_data()
            medico3d_instance.load_volume(sample_file)
    
    def on_show_info(b):
        with output:
            clear_output()
            medico3d_instance.show_volume_info()
    
    def on_gradient(b):
        with output:
            clear_output()
            medico3d_instance.compute_gradient()
    
    def on_laplacian(b):
        with output:
            clear_output()
            medico3d_instance.compute_laplacian()
    
    def on_stats(b):
        with output:
            clear_output()
            medico3d_instance.compute_statistical_features()
    
    def on_gaussian(b):
        with output:
            clear_output()
            medico3d_instance.apply_gaussian_filter()
    
    def on_umap(b):
        with output:
            clear_output()
            medico3d_instance.compute_umap_embedding()
    
    def on_view_slices(b):
        with output:
            clear_output()
            medico3d_instance.visualize_slices()
    
    def on_view_3d(b):
        with output:
            clear_output()
            medico3d_instance.create_3d_visualization()
    
    def on_interactive_viewer(b):
        with output:
            clear_output()
            medico3d_instance.create_interactive_viewer()
    
    def on_generate_report(b):
        with output:
            clear_output()
            report = medico3d_instance.generate_report()
            display(HTML(report))
    
    def on_save_features(b):
        with output:
            clear_output()
            medico3d_instance.save_features()
    
    # Conectar callbacks
    load_sample_btn.on_click(on_load_sample)
    info_btn.on_click(on_show_info)
    gradient_btn.on_click(on_gradient)
    laplacian_btn.on_click(on_laplacian)
    stats_btn.on_click(on_stats)
    gaussian_btn.on_click(on_gaussian)
    umap_btn.on_click(on_umap)
    view_slices_btn.on_click(on_view_slices)
    view_3d_btn.on_click(on_view_3d)
    interactive_viewer_btn.on_click(on_interactive_viewer)
    generate_report_btn.on_click(on_generate_report)
    save_features_btn.on_click(on_save_features)
    
    # Layout
    load_section = widgets.VBox([
        widgets.HTML("<h3>📂 Cargar Datos</h3>"),
        file_upload,
        load_sample_btn,
        info_btn
    ])
    
    process_section = widgets.VBox([
        widgets.HTML("<h3>🔬 Procesamiento</h3>"),
        widgets.HBox([gradient_btn, laplacian_btn]),
        widgets.HBox([stats_btn, gaussian_btn]),
        umap_btn
    ])
    
    viz_section = widgets.VBox([
        widgets.HTML("<h3>📊 Visualización</h3>"),
        widgets.HBox([view_slices_btn, view_3d_btn]),
        interactive_viewer_btn
    ])
    
    output_section = widgets.VBox([
        widgets.HTML("<h3>💾 Salida</h3>"),
        widgets.HBox([generate_report_btn, save_features_btn])
    ])
    
    interface = widgets.VBox([
        widgets.HTML("<h1>🏥 Medico3D - Análisis de Imágenes Médicas 3D</h1>"),
        widgets.HBox([
            widgets.VBox([load_section, process_section]),
            widgets.VBox([viz_section, output_section])
        ]),
        output
    ])
    
    return interface

# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """Función principal para ejecutar en Colab"""
    print("🏥 MEDICO3D PARA GOOGLE COLAB")
    print("=" * 50)
    print("Análisis de Imágenes Médicas 3D con Aceleración GPU")
    print("Versión: 2.0.0-Colab")
    print("=" * 50)
    
    # Mostrar información del sistema
    print(f"🔧 GPU disponible: {'Sí' if GPU_ENABLED else 'No'}")
    print(f"🔧 UMAP disponible: {'Sí' if UMAP_AVAILABLE else 'No'}")
    
    # Crear y mostrar interfaz
    interface = create_interactive_interface()
    display(interface)
    
    print("\n📖 INSTRUCCIONES:")
    print("1. Cargar archivo médico o usar datos de ejemplo")
    print("2. Mostrar información del volumen")
    print("3. Calcular características deseadas")
    print("4. Visualizar resultados")
    print("5. Generar reporte y guardar")

# Ejecutar automáticamente si se importa
if __name__ == "__main__":
    main()

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

"""
# EJEMPLO 1: Uso básico
medico = Medico3DColab()
sample_file = create_sample_data()
medico.load_volume(sample_file)
medico.show_volume_info()
medico.compute_gradient()
medico.visualize_features('gradient')

# EJEMPLO 2: Análisis completo
medico = Medico3DColab()
medico.load_volume('mi_archivo.nii.gz')
medico.compute_gradient()
medico.compute_laplacian()
medico.compute_statistical_features()
medico.create_3d_visualization('gradient')
medico.save_features()

# EJEMPLO 3: Demo rápido
demo_instance = quick_demo()

# EJEMPLO 4: Interfaz interactiva
main()
"""