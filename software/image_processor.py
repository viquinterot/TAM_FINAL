"""
Módulo de procesamiento de imágenes médicas para Medico3D
Integración mejorada de los scripts Python existentes
"""

import os
import sys
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# GPU acceleration
try:
    from gpu_processor import GPUImageProcessor, GPU_AVAILABLE
    GPU_PROCESSOR_AVAILABLE = True
except ImportError:
    GPU_PROCESSOR_AVAILABLE = False
    GPU_AVAILABLE = False

# Importar módulos de procesamiento existentes
try:
    # Añadir ruta de scripts al path
    scripts_path = Path(__file__).parent.parent / 'ConseilAnalizador3D' / 'ScriptsPython'
    if scripts_path.exists():
        sys.path.insert(0, str(scripts_path))
    
    from MainScript import generarCaracteristicas
    from GradientFunction import GraFunct
    from LH_Main import LHvFunct
    from StaticTF_Function import StaticFunction
    from TF_Laplacian import SecDerivate
    from TF_Curvature import Curvature, Hess
    from PruebaUMAP import TFUMAP, TFTSNE  # TFTSNE is now a wrapper for UMAP
    
    PROCESSING_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: Módulos de procesamiento no disponibles: {e}")
    PROCESSING_MODULES_AVAILABLE = False


class ProcessingResult:
    """Clase para encapsular resultados de procesamiento"""
    
    def __init__(self):
        self.success = False
        self.error_message = ""
        self.processing_time = 0.0
        self.features = {}
        self.output_files = []
        self.metadata = {}
    
    def add_feature(self, name: str, data: np.ndarray, description: str = ""):
        """Añadir característica procesada"""
        self.features[name] = {
            'data': data,
            'description': description,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data))
        }
    
    def add_output_file(self, file_path: str, file_type: str = ""):
        """Añadir archivo de salida generado"""
        self.output_files.append({
            'path': file_path,
            'type': file_type,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir resultado a diccionario"""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'features': {
                name: {
                    'description': info['description'],
                    'shape': info['shape'],
                    'dtype': info['dtype'],
                    'min_value': info['min_value'],
                    'max_value': info['max_value'],
                    'mean_value': info['mean_value']
                } for name, info in self.features.items()
            },
            'output_files': self.output_files,
            'metadata': self.metadata
        }


class ImageProcessor:
    """Procesador principal de imágenes médicas"""
    
    def __init__(self, progress_callback: Optional[Callable[[int, str], None]] = None, 
                 use_gpu: bool = True, gpu_memory_pool: Optional[int] = None):
        self.progress_callback = progress_callback
        self.current_volume = None
        self.current_file_path = ""
        
        # Initialize GPU processor if available
        self.use_gpu = use_gpu and GPU_PROCESSOR_AVAILABLE
        if self.use_gpu:
            self.gpu_processor = GPUImageProcessor(use_gpu=True, gpu_memory_pool=gpu_memory_pool)
            self._update_progress(0, f"GPU acceleration enabled: {self.gpu_processor.device_info}")
        else:
            self.gpu_processor = None
            if use_gpu and not GPU_PROCESSOR_AVAILABLE:
                self._update_progress(0, "GPU acceleration requested but not available, using CPU")
            else:
                self._update_progress(0, "Using CPU processing")
        
    def _update_progress(self, percentage: int, message: str = ""):
        """Actualizar progreso si hay callback disponible"""
        if self.progress_callback:
            self.progress_callback(percentage, message)
    
    def load_volume(self, file_path: str) -> Optional[np.ndarray]:
        """Cargar volumen desde archivo"""
        try:
            self._update_progress(5, "Cargando archivo...")
            
            if file_path.endswith(('.nii', '.nii.gz')):
                img = nib.load(file_path)
                volume = img.get_fdata()
            elif file_path.endswith(('.mha', '.mhd')):
                # Para archivos MHA/MHD, usar SimpleITK si está disponible
                try:
                    import SimpleITK as sitk
                    img = sitk.ReadImage(file_path)
                    volume = sitk.GetArrayFromImage(img)
                except ImportError:
                    # Fallback: intentar con nibabel
                    img = nib.load(file_path)
                    volume = img.get_fdata()
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_path}")
            
            self.current_volume = volume
            self.current_file_path = file_path
            
            self._update_progress(10, f"Archivo cargado: {volume.shape}")
            return volume
            
        except Exception as e:
            raise Exception(f"Error cargando archivo: {str(e)}")
    
    def process_all_features(self, file_path: str, parameters: Dict[str, Any]) -> ProcessingResult:
        """Procesar todas las características de una imagen"""
        result = ProcessingResult()
        start_time = time.time()
        
        try:
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            self._update_progress(0, "Iniciando procesamiento...")
            
            # Cargar volumen
            volume = self.load_volume(file_path)
            if volume is None:
                raise Exception("No se pudo cargar el volumen")
            
            # Extraer parámetros
            f = parameters.get('f', volume.shape[0])
            c = parameters.get('c', volume.shape[1])
            s = parameters.get('s', volume.shape[2])
            e = parameters.get('e', 0.5)
            Thr = parameters.get('Thr', 0.1)
            ThrI = parameters.get('ThrI', 0.1)
            Devs = parameters.get('Devs', -1)
            
            self._update_progress(15, "Procesando características...")
            
            # Llamar a la función principal de procesamiento
            processing_result = generarCaracteristicas(
                file_path, f, c, s, e, Thr, ThrI, Devs
            )
            
            self._update_progress(90, "Finalizando procesamiento...")
            
            # Procesar resultados (esto depende de lo que retorne generarCaracteristicas)
            if isinstance(processing_result, dict):
                for feature_name, feature_data in processing_result.items():
                    if isinstance(feature_data, np.ndarray):
                        result.add_feature(feature_name, feature_data)
            
            result.success = True
            result.processing_time = time.time() - start_time
            
            self._update_progress(100, "Procesamiento completado")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
        
        return result
    
    def process_gradient(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Procesar gradiente del volumen con aceleración GPU opcional"""
        try:
            self._update_progress(20, "Calculando gradiente...")
            
            # Try GPU acceleration first
            if self.use_gpu and self.gpu_processor:
                try:
                    self._update_progress(25, "Calculando gradiente en GPU...")
                    result = self.gpu_processor.compute_gradient_gpu(volume)
                    self._update_progress(30, "Gradiente calculado en GPU")
                    return result
                except Exception as e:
                    self._update_progress(25, f"GPU falló, usando CPU: {str(e)}")
            
            # Fallback to original CPU processing
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            # Aplanar volumen para la función
            vol_flat = volume.flatten()
            f, c, s = volume.shape
            
            # Calcular gradiente
            Gx, Gy, Gz, Gnorm = GraFunct(vol_flat, f, c, s)
            
            # Reshape a dimensiones originales
            Gx = Gx.reshape(volume.shape)
            Gy = Gy.reshape(volume.shape)
            Gz = Gz.reshape(volume.shape)
            Gnorm = Gnorm.reshape(volume.shape)
            
            return {
                'Gx': Gx,
                'Gy': Gy,
                'Gz': Gz,
                'Gnorm': Gnorm
            }
            
        except Exception as e:
            raise Exception(f"Error procesando gradiente: {str(e)}")
    
    def process_laplacian(self, volume: np.ndarray) -> np.ndarray:
        """Procesar Laplaciano del volumen con aceleración GPU opcional"""
        try:
            self._update_progress(30, "Calculando Laplaciano...")
            
            # Try GPU acceleration first
            if self.use_gpu and self.gpu_processor:
                try:
                    self._update_progress(35, "Calculando Laplaciano en GPU...")
                    result = self.gpu_processor.compute_laplacian_gpu(volume)
                    self._update_progress(40, "Laplaciano calculado en GPU")
                    return result
                except Exception as e:
                    self._update_progress(35, f"GPU falló, usando CPU: {str(e)}")
            
            # Fallback to original CPU processing
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            vol_flat = volume.flatten()
            f, c, s = volume.shape
            
            laplacian = SecDerivate(vol_flat, f, c, s)
            return laplacian.reshape(volume.shape)
            
        except Exception as e:
            raise Exception(f"Error procesando Laplaciano: {str(e)}")
    
    def process_curvature(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Procesar curvatura del volumen"""
        try:
            self._update_progress(40, "Calculando curvatura...")
            
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            vol_flat = volume.flatten()
            f, c, s = volume.shape
            
            k1, k2 = Curvature(vol_flat, f, c, s)
            
            return {
                'k1': k1.reshape(volume.shape),
                'k2': k2.reshape(volume.shape)
            }
            
        except Exception as e:
            raise Exception(f"Error procesando curvatura: {str(e)}")
    
    def process_lh_function(self, volume: np.ndarray, e: float, Thr: float) -> Dict[str, np.ndarray]:
        """Procesar función Low/High"""
        try:
            self._update_progress(50, "Calculando función LH...")
            
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            vol_flat = volume.flatten()
            f, c, s = volume.shape
            
            lh_result = LHvFunct(vol_flat, e, Thr, f, c, s)
            
            # El resultado puede variar según la implementación
            if isinstance(lh_result, tuple):
                lh_values, gradient_values = lh_result
                return {
                    'LH_values': lh_values.reshape(volume.shape),
                    'LH_gradient': gradient_values.reshape(volume.shape)
                }
            else:
                return {
                    'LH_values': lh_result.reshape(volume.shape)
                }
            
        except Exception as e:
            raise Exception(f"Error procesando función LH: {str(e)}")
    
    def process_statistical_features(self, volume: np.ndarray, parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Procesar características estadísticas con aceleración GPU opcional"""
        try:
            self._update_progress(60, "Calculando características estadísticas...")
            
            # Try GPU acceleration first
            if self.use_gpu and self.gpu_processor:
                try:
                    self._update_progress(65, "Calculando estadísticas en GPU...")
                    result = self.gpu_processor.compute_statistical_features_gpu(volume)
                    self._update_progress(70, "Estadísticas calculadas en GPU")
                    return result
                except Exception as e:
                    self._update_progress(65, f"GPU falló, usando CPU: {str(e)}")
            
            # Fallback to original CPU processing
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            vol_flat = volume.flatten()
            f, c, s = volume.shape
            
            # Parámetros para función estadística
            ThrI = parameters.get('ThrI', 0.1)
            Devs = parameters.get('Devs', -1)
            
            stat_result = StaticFunction(vol_flat, f, c, s, ThrI, Devs)
            
            # Procesar resultado estadístico
            if isinstance(stat_result, dict):
                processed_result = {}
                for key, value in stat_result.items():
                    if isinstance(value, np.ndarray):
                        processed_result[key] = value.reshape(volume.shape)
                    else:
                        processed_result[key] = value
                return processed_result
            else:
                return {'statistical_features': stat_result.reshape(volume.shape)}
            
        except Exception as e:
            raise Exception(f"Error procesando características estadísticas: {str(e)}")
    
    def process_umap_reduction(self, features: Dict[str, np.ndarray], parameters: Dict[str, Any]) -> np.ndarray:
        """Procesar reducción de dimensionalidad UMAP con aceleración GPU opcional"""
        try:
            self._update_progress(70, "Aplicando reducción UMAP...")
            
            # Preparar datos para UMAP
            feature_arrays = []
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, dict) and 'data' in feature_data:
                    feature_arrays.append(feature_data['data'].flatten())
                elif isinstance(feature_data, np.ndarray):
                    feature_arrays.append(feature_data.flatten())
            
            if not feature_arrays:
                raise Exception("No hay características disponibles para UMAP")
            
            # Combinar características
            combined_features = np.column_stack(feature_arrays)
            
            # Try GPU acceleration first
            if self.use_gpu and self.gpu_processor:
                try:
                    self._update_progress(75, "Calculando UMAP en GPU...")
                    result = self.gpu_processor.compute_umap_gpu(
                        combined_features, 
                        n_components=parameters.get('n_components', 2),
                        n_neighbors=parameters.get('n_neighbors', 15),
                        min_dist=parameters.get('min_dist', 0.1)
                    )
                    self._update_progress(80, "UMAP calculado en GPU")
                    return result
                except Exception as e:
                    self._update_progress(75, f"GPU falló, usando CPU: {str(e)}")
            
            # Fallback to original CPU processing
            if not PROCESSING_MODULES_AVAILABLE:
                raise Exception("Módulos de procesamiento no disponibles")
            
            # Aplicar UMAP
            umap_result = TFUMAP(combined_features)
            
            return umap_result
            
        except Exception as e:
            raise Exception(f"Error en reducción UMAP: {str(e)}")
    
    def save_results(self, result: ProcessingResult, output_dir: str) -> List[str]:
        """Guardar resultados de procesamiento"""
        try:
            self._update_progress(80, "Guardando resultados...")
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            base_name = Path(self.current_file_path).stem
            saved_files = []
            
            # Guardar cada característica como archivo MHA
            for feature_name, feature_info in result.features.items():
                feature_data = feature_info['data']
                output_file = output_path / f"{base_name}_{feature_name}.mha"
                
                try:
                    import SimpleITK as sitk
                    
                    # Convertir a SimpleITK Image
                    sitk_image = sitk.GetImageFromArray(feature_data)
                    sitk.WriteImage(sitk_image, str(output_file))
                    
                    saved_files.append(str(output_file))
                    result.add_output_file(str(output_file), feature_name)
                    
                except ImportError:
                    # Fallback: guardar como numpy array
                    np_file = output_path / f"{base_name}_{feature_name}.npy"
                    np.save(np_file, feature_data)
                    saved_files.append(str(np_file))
                    result.add_output_file(str(np_file), feature_name)
            
            # Guardar metadata como JSON
            metadata_file = output_path / f"{base_name}_metadata.json"
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            saved_files.append(str(metadata_file))
            
            return saved_files
            
        except Exception as e:
            raise Exception(f"Error guardando resultados: {str(e)}")


class BatchProcessor:
    """Procesador por lotes para múltiples archivos"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.results = []
    
    def process_files(self, file_paths: List[str], parameters: Dict[str, Any], 
                     progress_callback: Optional[Callable[[int, str], None]] = None) -> List[ProcessingResult]:
        """Procesar múltiples archivos en paralelo"""
        
        def process_single_file(file_path: str) -> ProcessingResult:
            processor = ImageProcessor()
            return processor.process_all_features(file_path, parameters)
        
        results = []
        total_files = len(file_paths)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_file, fp) for fp in file_paths]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress = int((i + 1) / total_files * 100)
                        progress_callback(progress, f"Procesado {i + 1}/{total_files} archivos")
                        
                except Exception as e:
                    error_result = ProcessingResult()
                    error_result.success = False
                    error_result.error_message = str(e)
                    results.append(error_result)
        
        return results