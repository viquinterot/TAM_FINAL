#!/usr/bin/env python3
"""
TAM - Medical Image 3D UMAP
Aplicación para análisis de imágenes médicas 3D con reducción de dimensionalidad UMAP
Versión Python con interfaz PyQt5 y visualización VTK

Autor: Víctor Quintero
Fecha: Julio 2025
"""

import sys
import os
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar, QStatusBar,
    QFileDialog, QMessageBox, QDialog, QListWidget, QSplitter,
    QAction, QToolBar, QMenuBar, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, QSettings, QByteArray, Qt
from PyQt5.QtGui import QPixmap, QImage

# VTK imports
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Scientific computing imports
import numpy as np

# Importar módulos locales
from config import Medico3DConfig
from transfer_functions import TransferFunctionManager, TransferFunction
from image_processor import ImageProcessor, ProcessingResult
from napari_widget import create_napari_widget

# Inicializar configuración global
config = Medico3DConfig()


class ProcessingThread(QThread):
    """Hilo para procesamiento en segundo plano"""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, volume: np.ndarray, params: dict, processor: ImageProcessor):
        super().__init__()
        self.volume = volume
        self.params = params
        self.processor = processor
    
    def run(self):
        """Ejecutar procesamiento"""
        try:
            def progress_callback(progress: int):
                self.progress.emit(progress)
            
            # Procesar todas las características
            result = self.processor.process_all_features(
                self.volume,
                self.params,
                progress_callback=progress_callback
            )
            
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = f"Error en procesamiento: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class VTKWidget(QWidget):
    """Widget para visualización 3D con VTK"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume_data = None
        self.volume_mapper = None
        self.volume_actor = None
        self.init_vtk()
    
    def init_vtk(self):
        """Inicializar componentes VTK"""
        layout = QVBoxLayout()
        
        # Crear widget VTK
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        self.setLayout(layout)
        
        # Configurar renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.2)  # Fondo azul oscuro
        
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        # Configurar interactor
        self.interactor = self.render_window.GetInteractor()
        
        # Agregar ejes de coordenadas
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetTotalLength(50, 50, 50)
        self.renderer.AddActor(self.axes_actor)
        
        # Inicializar interactor
        self.interactor.Initialize()
        self.interactor.Start()
    
    def set_volume(self, volume_data: np.ndarray):
        """Establecer datos de volumen para visualización"""
        try:
            print(f"Configurando volumen: shape={volume_data.shape}, dtype={volume_data.dtype}")
            self.volume_data = volume_data
            
            # Normalizar datos si es necesario
            if volume_data.dtype != np.float32:
                volume_data = volume_data.astype(np.float32)
            
            # Asegurar que los datos estén en el rango correcto
            vmin, vmax = volume_data.min(), volume_data.max()
            print(f"Rango de datos: {vmin} - {vmax}")
            
            # Convertir numpy array a VTK
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(volume_data.shape[2], volume_data.shape[1], volume_data.shape[0])  # VTK usa orden diferente
            vtk_data.SetSpacing(1.0, 1.0, 1.0)
            vtk_data.SetOrigin(0.0, 0.0, 0.0)
            
            # Convertir datos - usar orden C para VTK
            flat_data = volume_data.flatten(order='C')
            vtk_array = vtk.vtkFloatArray()
            vtk_array.SetNumberOfTuples(flat_data.size)
            vtk_array.SetNumberOfComponents(1)
            
            # Copiar datos en lugar de usar SetVoidArray para evitar problemas de memoria
            for i in range(flat_data.size):
                vtk_array.SetValue(i, float(flat_data[i]))
            
            vtk_data.GetPointData().SetScalars(vtk_array)
            
            # Crear mapper de volumen - usar CPU mapper como fallback
            try:
                self.volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
                self.volume_mapper.SetInputData(vtk_data)
                print("Usando GPU Volume Ray Cast Mapper")
            except:
                print("GPU mapper falló, usando CPU mapper")
                self.volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
                self.volume_mapper.SetInputData(vtk_data)
            
            # Crear propiedades de volumen
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetInterpolationTypeToLinear()
            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            volume_property.SetDiffuse(0.6)
            volume_property.SetSpecular(0.2)
            
            # Función de transferencia por defecto basada en el rango real de datos (escala de grises)
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(vmin + (vmax-vmin)*0.3, 0.3, 0.3, 0.3)
            color_func.AddRGBPoint(vmin + (vmax-vmin)*0.6, 0.7, 0.7, 0.7)
            color_func.AddRGBPoint(vmax, 1.0, 1.0, 1.0)
            
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(vmin, 0.0)
            opacity_func.AddPoint(vmin + (vmax-vmin)*0.2, 0.0)
            opacity_func.AddPoint(vmin + (vmax-vmin)*0.5, 0.3)
            opacity_func.AddPoint(vmax, 0.8)
            
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            
            # Crear actor de volumen
            if self.volume_actor:
                self.renderer.RemoveActor(self.volume_actor)
            
            self.volume_actor = vtk.vtkVolume()
            self.volume_actor.SetMapper(self.volume_mapper)
            self.volume_actor.SetProperty(volume_property)
            
            self.renderer.AddActor(self.volume_actor)
            self.renderer.ResetCamera()
            self.render_window.Render()
            
            print("Volumen configurado exitosamente")
            
        except Exception as e:
            print(f"Error configurando volumen VTK: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_camera(self):
        """Resetear vista de cámara"""
        if self.renderer:
            self.renderer.ResetCamera()
            self.render_window.Render()
    
    def toggle_axes(self):
        """Mostrar/ocultar ejes de coordenadas"""
        if self.axes_actor:
            visible = self.axes_actor.GetVisibility()
            self.axes_actor.SetVisibility(not visible)
            self.render_window.Render()
    
    def save_screenshot(self, filename: str):
        """Guardar captura de pantalla"""
        try:
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(self.render_window)
            window_to_image.Update()
            
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
        except Exception as e:
            raise Exception(f"Error guardando captura: {e}")


class TransferFunctionWidget(QWidget):
    """Widget para editar funciones de transferencia"""
    
    def __init__(self, tf_manager=None, main_window=None):
        super().__init__(main_window)
        self.tf_manager = tf_manager if tf_manager else TransferFunctionManager()
        self.main_window = main_window
        self.current_tf = None
        self.init_ui()
        self.load_transfer_functions()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Selector de función de transferencia predefinida
        tf_group = QGroupBox("Funciones de Transferencia")
        tf_layout = QVBoxLayout()
        
        self.tf_combo = QComboBox()
        self.tf_combo.currentTextChanged.connect(self.on_tf_changed)
        tf_layout.addWidget(QLabel("Función predefinida:"))
        tf_layout.addWidget(self.tf_combo)
        
        # Controles de color
        color_group = QGroupBox("Puntos de Color")
        color_layout = QGridLayout()
        
        self.color_table = QTableWidget(5, 4)
        self.color_table.setHorizontalHeaderLabels(["Punto", "Rojo", "Verde", "Azul"])
        self.color_table.cellChanged.connect(self.on_color_table_changed)
        color_layout.addWidget(self.color_table, 0, 0, 1, 4)
        
        # Controles de opacidad
        opacity_group = QGroupBox("Puntos de Opacidad")
        opacity_layout = QGridLayout()
        
        self.opacity_table = QTableWidget(5, 2)
        self.opacity_table.setHorizontalHeaderLabels(["Punto", "Opacidad"])
        self.opacity_table.cellChanged.connect(self.on_opacity_table_changed)
        opacity_layout.addWidget(self.opacity_table, 0, 0, 1, 2)
        
        # Botones
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Cargar TF")
        self.save_btn = QPushButton("Guardar TF")
        self.apply_btn = QPushButton("Aplicar")
        
        self.load_btn.clicked.connect(self.load_tf_from_file)
        self.save_btn.clicked.connect(self.save_tf_to_file)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.apply_btn)
        
        # Ensamblar layout
        tf_group.setLayout(tf_layout)
        color_group.setLayout(color_layout)
        opacity_group.setLayout(opacity_layout)
        
        layout.addWidget(tf_group)
        layout.addWidget(color_group)
        layout.addWidget(opacity_group)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def load_transfer_functions(self):
        """Cargar funciones de transferencia disponibles"""
        # Cargar TF desde directorio legacy si existe
        legacy_tf_path = config.get_transfer_functions_path()
        if legacy_tf_path:
            self.tf_manager.load_legacy_xml_directory(legacy_tf_path)
        
        # Poblar combo
        tf_names = self.tf_manager.get_function_names()
        self.tf_combo.addItems(tf_names)
        
        # Seleccionar TF por defecto
        default_tf = config.get('visualization.default_transfer_function', 'Por defecto')
        if default_tf in tf_names:
            self.tf_combo.setCurrentText(default_tf)
    
    def on_tf_changed(self, tf_name: str):
        """Callback cuando cambia la función de transferencia seleccionada"""
        self.current_tf = self.tf_manager.get_transfer_function(tf_name)
        if self.current_tf:
            self.update_tables()
            # Aplicar automáticamente la función de transferencia si hay un volumen cargado
            if self.main_window and hasattr(self.main_window, 'current_volume') and self.main_window.current_volume is not None:
                self.main_window.apply_transfer_function()
    
    def update_tables(self):
        """Actualizar tablas con datos de la TF actual"""
        if not self.current_tf:
            return
        
        # Actualizar tabla de colores
        color_points = self.current_tf.get_color_points()
        self.color_table.setRowCount(max(5, len(color_points)))
        
        for i, (point, r, g, b) in enumerate(color_points):
            self.color_table.setItem(i, 0, QTableWidgetItem(str(point)))
            self.color_table.setItem(i, 1, QTableWidgetItem(f"{r:.3f}"))
            self.color_table.setItem(i, 2, QTableWidgetItem(f"{g:.3f}"))
            self.color_table.setItem(i, 3, QTableWidgetItem(f"{b:.3f}"))
        
        # Actualizar tabla de opacidad
        opacity_points = self.current_tf.get_opacity_points()
        self.opacity_table.setRowCount(max(5, len(opacity_points)))
        
        for i, (point, opacity) in enumerate(opacity_points):
            self.opacity_table.setItem(i, 0, QTableWidgetItem(str(point)))
            self.opacity_table.setItem(i, 1, QTableWidgetItem(f"{opacity:.3f}"))
    
    def on_color_table_changed(self):
        """Callback cuando cambia la tabla de colores"""
        if not self.current_tf:
            return
        self.update_tf_from_tables()
    
    def on_opacity_table_changed(self):
        """Callback cuando cambia la tabla de opacidad"""
        if not self.current_tf:
            return
        self.update_tf_from_tables()
    
    def update_tf_from_tables(self):
        """Actualizar TF actual desde las tablas"""
        if not self.current_tf:
            return
        
        try:
            # Limpiar puntos actuales
            self.current_tf.color_points.clear()
            self.current_tf.opacity_points.clear()
            
            # Leer tabla de colores
            for row in range(self.color_table.rowCount()):
                point_item = self.color_table.item(row, 0)
                r_item = self.color_table.item(row, 1)
                g_item = self.color_table.item(row, 2)
                b_item = self.color_table.item(row, 3)
                
                if all(item and item.text().strip() for item in [point_item, r_item, g_item, b_item]):
                    point = float(point_item.text())
                    r = float(r_item.text())
                    g = float(g_item.text())
                    b = float(b_item.text())
                    self.current_tf.add_color_point(point, r, g, b)
            
            # Leer tabla de opacidad
            for row in range(self.opacity_table.rowCount()):
                point_item = self.opacity_table.item(row, 0)
                opacity_item = self.opacity_table.item(row, 1)
                
                if point_item and opacity_item and point_item.text().strip() and opacity_item.text().strip():
                    point = float(point_item.text())
                    opacity = float(opacity_item.text())
                    self.current_tf.add_opacity_point(point, opacity)
                    
        except ValueError as e:
            print(f"Error actualizando TF: {e}")
    
    def get_current_transfer_function(self) -> Optional[TransferFunction]:
        """Obtener función de transferencia actual"""
        return self.current_tf
    
    def load_tf_from_file(self):
        """Cargar TF desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Función de Transferencia",
            "", "Archivos JSON (*.json);;Archivos XML (*.xml)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    self.tf_manager.load_from_json(file_path)
                else:
                    self.tf_manager.load_from_xml(file_path)
                
                # Actualizar combo
                self.tf_combo.clear()
                self.tf_combo.addItems(self.tf_manager.get_function_names())
                
                QMessageBox.information(self, "Éxito", "Función de transferencia cargada")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando TF: {e}")
    
    def save_tf_to_file(self):
        """Guardar TF actual a archivo"""
        if not self.current_tf:
            QMessageBox.warning(self, "Advertencia", "No hay función de transferencia seleccionada")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Función de Transferencia",
            f"{self.current_tf.name}.json", "Archivos JSON (*.json)"
        )
        
        if file_path:
            try:
                # Crear manager temporal con solo esta TF
                temp_manager = TransferFunctionManager()
                temp_manager.transfer_functions = {self.current_tf.name: self.current_tf}
                temp_manager.save_to_json(file_path)
                
                QMessageBox.information(self, "Éxito", "Función de transferencia guardada")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando TF: {e}")


class ParametersWidget(QWidget):
    """Widget para configurar parámetros de procesamiento"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_parameters()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Parámetros de entrada
        input_group = QGroupBox("Parámetros de Entrada")
        input_layout = QFormLayout()
        
        self.input_file_edit = QLineEdit()
        self.input_browse_btn = QPushButton("Examinar...")
        self.input_browse_btn.clicked.connect(self.browse_input_file)
        
        input_file_layout = QHBoxLayout()
        input_file_layout.addWidget(self.input_file_edit)
        input_file_layout.addWidget(self.input_browse_btn)
        
        input_layout.addRow("Archivo de entrada:", input_file_layout)
        
        # Dimensiones
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 2048)
        self.rows_spin.setValue(config.get('processing.default_rows', 256))
        
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 2048)
        self.cols_spin.setValue(config.get('processing.default_cols', 256))
        
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 2048)
        self.layers_spin.setValue(config.get('processing.default_layers', 256))
        
        input_layout.addRow("Filas:", self.rows_spin)
        input_layout.addRow("Columnas:", self.cols_spin)
        input_layout.addRow("Capas:", self.layers_spin)
        
        input_group.setLayout(input_layout)
        
        # Parámetros de procesamiento
        processing_group = QGroupBox("Parámetros de Procesamiento")
        processing_layout = QFormLayout()
        
        self.gradient_threshold_spin = QDoubleSpinBox()
        self.gradient_threshold_spin.setRange(0.0, 1.0)
        self.gradient_threshold_spin.setSingleStep(0.01)
        self.gradient_threshold_spin.setDecimals(3)
        self.gradient_threshold_spin.setValue(config.get('processing.gradient_threshold', 0.1))
        
        self.intensity_threshold_spin = QDoubleSpinBox()
        self.intensity_threshold_spin.setRange(0.0, 1.0)
        self.intensity_threshold_spin.setSingleStep(0.01)
        self.intensity_threshold_spin.setDecimals(3)
        self.intensity_threshold_spin.setValue(config.get('processing.intensity_threshold', 0.1))
        
        processing_layout.addRow("Umbral de gradiente:", self.gradient_threshold_spin)
        processing_layout.addRow("Umbral de intensidad:", self.intensity_threshold_spin)
        
        # Parámetros de UMAP
        self.umap_neighbors_spin = QSpinBox()
        self.umap_neighbors_spin.setRange(5, 200)
        self.umap_neighbors_spin.setValue(config.get('processing.umap_neighbors', 15))
        
        self.umap_min_dist_spin = QDoubleSpinBox()
        self.umap_min_dist_spin.setRange(0.0, 1.0)
        self.umap_min_dist_spin.setSingleStep(0.01)
        self.umap_min_dist_spin.setValue(config.get('processing.umap_min_dist', 0.1))
        
        processing_layout.addRow("UMAP Vecinos:", self.umap_neighbors_spin)
        processing_layout.addRow("UMAP Distancia Mín:", self.umap_min_dist_spin)
        
        processing_group.setLayout(processing_layout)
        
        # Parámetros de salida
        output_group = QGroupBox("Parámetros de Salida")
        output_layout = QFormLayout()
        
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Examinar...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_browse_btn)
        
        output_layout.addRow("Directorio de salida:", output_dir_layout)
        
        # Formato de salida
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["MHA", "NIfTI", "Ambos"])
        self.output_format_combo.setCurrentText(config.get('processing.output_format', 'MHA'))
        
        output_layout.addRow("Formato de salida:", self.output_format_combo)
        
        output_group.setLayout(output_layout)
        
        # Características a procesar
        features_group = QGroupBox("Características a Procesar")
        features_layout = QGridLayout()
        
        self.feature_checkboxes = {}
        features = [
            ('intensity', 'Intensidad'),
            ('gradient', 'Gradiente'),
            ('laplacian', 'Laplaciano'),
            ('curvature', 'Curvatura'),
            ('mean', 'Media'),
            ('std', 'Desviación Estándar'),
            ('lh', 'Función LH'),
            ('umap', 'UMAP')
        ]
        
        for i, (key, label) in enumerate(features):
            checkbox = QCheckBox(label)
            checkbox.setChecked(config.get(f'processing.features.{key}', True))
            self.feature_checkboxes[key] = checkbox
            features_layout.addWidget(checkbox, i // 2, i % 2)
        
        features_group.setLayout(features_layout)
        
        # Botones
        btn_layout = QHBoxLayout()
        self.load_params_btn = QPushButton("Cargar Parámetros")
        self.save_params_btn = QPushButton("Guardar Parámetros")
        self.reset_params_btn = QPushButton("Restablecer")
        
        self.load_params_btn.clicked.connect(self.load_parameters_from_file)
        self.save_params_btn.clicked.connect(self.save_parameters_to_file)
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        
        btn_layout.addWidget(self.load_params_btn)
        btn_layout.addWidget(self.save_params_btn)
        btn_layout.addWidget(self.reset_params_btn)
        
        # Ensamblar layout
        layout.addWidget(input_group)
        layout.addWidget(processing_group)
        layout.addWidget(output_group)
        layout.addWidget(features_group)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def load_parameters(self):
        """Cargar parámetros desde configuración"""
        # Directorio de salida por defecto
        default_output = config.get('paths.output_directory', '')
        if default_output:
            self.output_dir_edit.setText(default_output)
    
    def browse_input_file(self):
        """Examinar archivo de entrada"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de entrada",
            config.get('paths.last_input_directory', ''),
            "Archivos de imagen (*.nii *.nii.gz *.mha *.mhd);;Todos los archivos (*.*)"
        )
        
        if file_path:
            self.input_file_edit.setText(file_path)
            # Guardar directorio para próxima vez
            config.set('paths.last_input_directory', os.path.dirname(file_path))
    
    def browse_output_dir(self):
        """Examinar directorio de salida"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio de salida",
            config.get('paths.output_directory', '')
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            config.set('paths.output_directory', dir_path)
    
    def get_parameters(self) -> dict:
        """Obtener parámetros actuales como diccionario"""
        params = {
            'input_file': self.input_file_edit.text(),
            'output_directory': self.output_dir_edit.text(),
            'dimensions': {
                'rows': self.rows_spin.value(),
                'cols': self.cols_spin.value(),
                'layers': self.layers_spin.value()
            },
            'thresholds': {
                'gradient': self.gradient_threshold_spin.value(),
                'intensity': self.intensity_threshold_spin.value()
            },
            'umap': {
                'neighbors': self.umap_neighbors_spin.value(),
                'min_dist': self.umap_min_dist_spin.value()
            },
            'output_format': self.output_format_combo.currentText(),
            'features': {
                key: checkbox.isChecked() 
                for key, checkbox in self.feature_checkboxes.items()
            }
        }
        return params
    
    def set_parameters(self, params: dict):
        """Establecer parámetros desde diccionario"""
        if 'input_file' in params:
            self.input_file_edit.setText(params['input_file'])
        
        if 'output_directory' in params:
            self.output_dir_edit.setText(params['output_directory'])
        
        if 'dimensions' in params:
            dims = params['dimensions']
            self.rows_spin.setValue(dims.get('rows', 256))
            self.cols_spin.setValue(dims.get('cols', 256))
            self.layers_spin.setValue(dims.get('layers', 256))
        
        if 'thresholds' in params:
            thresh = params['thresholds']
            self.gradient_threshold_spin.setValue(thresh.get('gradient', 0.1))
            self.intensity_threshold_spin.setValue(thresh.get('intensity', 0.1))
        
        if 'umap' in params:
            umap_params = params['umap']
            self.umap_neighbors_spin.setValue(umap_params.get('neighbors', 15))
            self.umap_min_dist_spin.setValue(umap_params.get('min_dist', 0.1))
        
        if 'output_format' in params:
            self.output_format_combo.setCurrentText(params['output_format'])
        
        if 'features' in params:
            features = params['features']
            for key, checkbox in self.feature_checkboxes.items():
                checkbox.setChecked(features.get(key, True))
    
    def save_parameters_to_file(self):
        """Guardar parámetros a archivo"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Parámetros",
            "parametros.json", "Archivos JSON (*.json)"
        )
        
        if file_path:
            try:
                import json
                params = self.get_parameters()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Éxito", "Parámetros guardados correctamente")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando parámetros: {e}")
    
    def load_parameters_from_file(self):
        """Cargar parámetros desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Parámetros",
            "", "Archivos JSON (*.json);;Archivos XML (*.xml)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        params = json.load(f)
                else:
                    # Cargar desde XML legacy
                    params = self._load_legacy_xml_params(file_path)
                
                self.set_parameters(params)
                QMessageBox.information(self, "Éxito", "Parámetros cargados correctamente")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error cargando parámetros: {e}")
    
    def _load_legacy_xml_params(self, file_path: str) -> dict:
        """Cargar parámetros desde XML legacy"""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        params = {}
        
        # Buscar parámetros conocidos
        for param in root.findall('.//Parametro'):
            name = param.get('Nombre', '')
            value = param.get('Valor', '')
            
            if name == 'ArchivoEntrada':
                params['input_file'] = value
            elif name == 'Filas':
                params.setdefault('dimensions', {})['rows'] = int(value)
            elif name == 'Columnas':
                params.setdefault('dimensions', {})['cols'] = int(value)
            elif name == 'Capas':
                params.setdefault('dimensions', {})['layers'] = int(value)
            elif name == 'UmbralGradiente':
                params.setdefault('thresholds', {})['gradient'] = float(value)
            elif name == 'UmbralIntensidad':
                params.setdefault('thresholds', {})['intensity'] = float(value)
        
        return params
    
    def reset_parameters(self):
        """Restablecer parámetros a valores por defecto"""
        self.input_file_edit.clear()
        self.output_dir_edit.setText(config.get('paths.output_directory', ''))
        
        self.rows_spin.setValue(256)
        self.cols_spin.setValue(256)
        self.layers_spin.setValue(256)
        
        self.gradient_threshold_spin.setValue(0.1)
        self.intensity_threshold_spin.setValue(0.1)
        
        self.umap_neighbors_spin.setValue(15)
        self.umap_min_dist_spin.setValue(0.1)
        
        self.output_format_combo.setCurrentText('MHA')
        
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)


class BatchProcessDialog(QDialog):
    """Diálogo para procesamiento por lotes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Procesamiento por Lotes")
        self.setModal(True)
        self.resize(600, 400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Lista de archivos
        files_group = QGroupBox("Archivos a Procesar")
        files_layout = QVBoxLayout()
        
        self.files_list = QListWidget()
        files_layout.addWidget(self.files_list)
        
        # Botones para manejo de archivos
        files_btn_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Agregar Archivos...")
        self.add_dir_btn = QPushButton("Agregar Directorio...")
        self.remove_btn = QPushButton("Remover Seleccionados")
        self.clear_btn = QPushButton("Limpiar Lista")
        
        self.add_files_btn.clicked.connect(self.add_files)
        self.add_dir_btn.clicked.connect(self.add_directory)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn.clicked.connect(self.clear_list)
        
        files_btn_layout.addWidget(self.add_files_btn)
        files_btn_layout.addWidget(self.add_dir_btn)
        files_btn_layout.addWidget(self.remove_btn)
        files_btn_layout.addWidget(self.clear_btn)
        
        files_layout.addLayout(files_btn_layout)
        files_group.setLayout(files_layout)
        
        # Configuración de salida
        output_group = QGroupBox("Configuración de Salida")
        output_layout = QFormLayout()
        
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Examinar...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_browse_btn)
        
        output_layout.addRow("Directorio de salida:", output_dir_layout)
        
        self.parallel_check = QCheckBox("Procesamiento paralelo")
        self.parallel_check.setChecked(True)
        output_layout.addRow("", self.parallel_check)
        
        output_group.setLayout(output_layout)
        
        # Botones del diálogo
        button_layout = QHBoxLayout()
        self.process_btn = QPushButton("Procesar")
        self.cancel_btn = QPushButton("Cancelar")
        
        self.process_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # Ensamblar layout
        layout.addWidget(files_group)
        layout.addWidget(output_group)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_files(self):
        """Agregar archivos individuales"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar Archivos",
            "", "Archivos de imagen (*.nii *.nii.gz *.mha *.mhd)"
        )
        
        for file_path in files:
            self.files_list.addItem(file_path)
    
    def add_directory(self):
        """Agregar todos los archivos de un directorio"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio"
        )
        
        if dir_path:
            import glob
            
            # Buscar archivos de imagen en el directorio
            patterns = ['*.nii', '*.nii.gz', '*.mha', '*.mhd']
            for pattern in patterns:
                files = glob.glob(os.path.join(dir_path, pattern))
                for file_path in files:
                    self.files_list.addItem(file_path)
    
    def remove_selected(self):
        """Remover archivos seleccionados"""
        for item in self.files_list.selectedItems():
            self.files_list.takeItem(self.files_list.row(item))
    
    def clear_list(self):
        """Limpiar lista de archivos"""
        self.files_list.clear()
    
    def browse_output_dir(self):
        """Examinar directorio de salida"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio de Salida"
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)


class LayersNavigationWidget(QWidget):
    """Widget para navegación por capas del volumen"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume_data = None
        self.current_layer = 0
        self.total_layers = 0
        self.current_axis = 0  # 0=Z (axial), 1=Y (coronal), 2=X (sagital)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Información del volumen
        info_group = QGroupBox("Información del Volumen")
        info_layout = QFormLayout()
        
        self.volume_info_label = QLabel("No hay volumen cargado")
        info_layout.addRow("Estado:", self.volume_info_label)
        
        self.dimensions_label = QLabel("-")
        info_layout.addRow("Dimensiones:", self.dimensions_label)
        
        self.data_range_label = QLabel("-")
        info_layout.addRow("Rango de datos:", self.data_range_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Selección de eje
        axis_group = QGroupBox("Eje de Visualización")
        axis_layout = QVBoxLayout()
        
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagital (X)"])
        self.axis_combo.currentIndexChanged.connect(self.on_axis_changed)
        axis_layout.addWidget(self.axis_combo)
        
        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)
        
        # Controles de navegación por capas
        navigation_group = QGroupBox("Navegación por Capas")
        nav_layout = QVBoxLayout()
        
        # Slider para navegación
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(0)
        self.layer_slider.setValue(0)
        self.layer_slider.valueChanged.connect(self.on_layer_changed)
        nav_layout.addWidget(self.layer_slider)
        
        # Información de capa actual
        layer_info_layout = QHBoxLayout()
        
        self.layer_label = QLabel("Capa: 0 / 0")
        layer_info_layout.addWidget(self.layer_label)
        
        layer_info_layout.addStretch()
        
        # Botones de navegación
        self.prev_btn = QPushButton("◀ Anterior")
        self.next_btn = QPushButton("Siguiente ▶")
        self.prev_btn.clicked.connect(self.previous_layer)
        self.next_btn.clicked.connect(self.next_layer)
        
        layer_info_layout.addWidget(self.prev_btn)
        layer_info_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(layer_info_layout)
        
        # Input directo de número de capa
        direct_nav_layout = QHBoxLayout()
        direct_nav_layout.addWidget(QLabel("Ir a capa:"))
        
        self.layer_spinbox = QSpinBox()
        self.layer_spinbox.setMinimum(0)
        self.layer_spinbox.setMaximum(0)
        self.layer_spinbox.valueChanged.connect(self.on_spinbox_changed)
        direct_nav_layout.addWidget(self.layer_spinbox)
        
        direct_nav_layout.addStretch()
        nav_layout.addLayout(direct_nav_layout)
        
        navigation_group.setLayout(nav_layout)
        layout.addWidget(navigation_group)
        
        # Visualización de la capa actual
        display_group = QGroupBox("Vista de Capa Actual")
        display_layout = QVBoxLayout()
        
        # Widget para mostrar la imagen de la capa
        self.layer_display = QLabel()
        self.layer_display.setMinimumSize(300, 300)
        self.layer_display.setAlignment(Qt.AlignCenter)
        self.layer_display.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.layer_display.setText("No hay capa para mostrar")
        
        display_layout.addWidget(self.layer_display)
        
        # Controles de visualización de la capa
        display_controls_layout = QHBoxLayout()
        
        # Control de contraste
        display_controls_layout.addWidget(QLabel("Contraste:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_layer_display)
        display_controls_layout.addWidget(self.contrast_slider)
        
        # Control de brillo
        display_controls_layout.addWidget(QLabel("Brillo:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_layer_display)
        display_controls_layout.addWidget(self.brightness_slider)
        
        display_layout.addLayout(display_controls_layout)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Inicialmente deshabilitar controles
        self.set_controls_enabled(False)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_volume_data(self, volume_data):
        """Establecer datos del volumen"""
        self.volume_data = volume_data
        if volume_data is not None:
            # Actualizar información del volumen
            self.volume_info_label.setText("Volumen cargado")
            self.dimensions_label.setText(f"{volume_data.shape[0]} × {volume_data.shape[1]} × {volume_data.shape[2]}")
            self.data_range_label.setText(f"{volume_data.min():.1f} - {volume_data.max():.1f}")
            
            # Configurar controles para el eje actual
            self.update_axis_controls()
            self.set_controls_enabled(True)
            
            # Mostrar primera capa
            self.update_layer_display()
        else:
            self.volume_info_label.setText("No hay volumen cargado")
            self.dimensions_label.setText("-")
            self.data_range_label.setText("-")
            self.set_controls_enabled(False)
    
    def set_controls_enabled(self, enabled):
        """Habilitar/deshabilitar controles"""
        self.axis_combo.setEnabled(enabled)
        self.layer_slider.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.layer_spinbox.setEnabled(enabled)
        self.contrast_slider.setEnabled(enabled)
        self.brightness_slider.setEnabled(enabled)
    
    def update_axis_controls(self):
        """Actualizar controles según el eje seleccionado"""
        if self.volume_data is None:
            return
        
        # Obtener número de capas según el eje
        if self.current_axis == 0:  # Z (axial)
            self.total_layers = self.volume_data.shape[2]
        elif self.current_axis == 1:  # Y (coronal)
            self.total_layers = self.volume_data.shape[1]
        else:  # X (sagital)
            self.total_layers = self.volume_data.shape[0]
        
        # Actualizar controles
        self.layer_slider.setMaximum(self.total_layers - 1)
        self.layer_spinbox.setMaximum(self.total_layers - 1)
        
        # Resetear a primera capa
        self.current_layer = 0
        self.layer_slider.setValue(0)
        self.layer_spinbox.setValue(0)
        
        self.update_layer_info()
    
    def update_layer_info(self):
        """Actualizar información de la capa actual"""
        self.layer_label.setText(f"Capa: {self.current_layer + 1} / {self.total_layers}")
        
        # Habilitar/deshabilitar botones según la posición
        self.prev_btn.setEnabled(self.current_layer > 0)
        self.next_btn.setEnabled(self.current_layer < self.total_layers - 1)
    
    def on_axis_changed(self, index):
        """Manejar cambio de eje"""
        self.current_axis = index
        self.update_axis_controls()
        self.update_layer_display()
    
    def on_layer_changed(self, value):
        """Manejar cambio de capa desde slider"""
        self.current_layer = value
        self.layer_spinbox.blockSignals(True)
        self.layer_spinbox.setValue(value)
        self.layer_spinbox.blockSignals(False)
        self.update_layer_info()
        self.update_layer_display()
    
    def on_spinbox_changed(self, value):
        """Manejar cambio de capa desde spinbox"""
        self.current_layer = value
        self.layer_slider.blockSignals(True)
        self.layer_slider.setValue(value)
        self.layer_slider.blockSignals(False)
        self.update_layer_info()
        self.update_layer_display()
    
    def previous_layer(self):
        """Ir a la capa anterior"""
        if self.current_layer > 0:
            self.current_layer -= 1
            self.layer_slider.setValue(self.current_layer)
    
    def next_layer(self):
        """Ir a la siguiente capa"""
        if self.current_layer < self.total_layers - 1:
            self.current_layer += 1
            self.layer_slider.setValue(self.current_layer)
    
    def update_layer_display(self):
        """Actualizar visualización de la capa actual"""
        if self.volume_data is None:
            self.layer_display.setText("No hay capa para mostrar")
            return
        
        try:
            # Extraer la capa según el eje seleccionado
            if self.current_axis == 0:  # Z (axial)
                layer_data = self.volume_data[:, :, self.current_layer]
            elif self.current_axis == 1:  # Y (coronal)
                layer_data = self.volume_data[:, self.current_layer, :]
            else:  # X (sagital)
                layer_data = self.volume_data[self.current_layer, :, :]
            
            # Normalizar datos para visualización
            layer_min, layer_max = layer_data.min(), layer_data.max()
            if layer_max > layer_min:
                normalized = (layer_data - layer_min) / (layer_max - layer_min)
            else:
                normalized = np.zeros_like(layer_data)
            
            # Aplicar contraste y brillo
            contrast = self.contrast_slider.value() / 100.0
            brightness = self.brightness_slider.value() / 100.0
            
            adjusted = normalized * contrast + brightness
            adjusted = np.clip(adjusted, 0, 1)
            
            # Convertir a imagen de 8 bits
            image_data = (adjusted * 255).astype(np.uint8)
            
            # Crear QImage
            height, width = image_data.shape
            bytes_per_line = width
            
            q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Escalar imagen para ajustar al widget
            display_size = self.layer_display.size()
            scaled_image = q_image.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Mostrar imagen
            pixmap = QPixmap.fromImage(scaled_image)
            self.layer_display.setPixmap(pixmap)
            
        except Exception as e:
            self.layer_display.setText(f"Error mostrando capa: {e}")


class Medico3DMainWindow(QMainWindow):
    """Ventana principal de la aplicación Medico3D"""
    
    def __init__(self):
        super().__init__()
        
        # Inicializar procesador de imágenes con soporte GPU
        try:
            # Check if GPU is available
            from gpu_processor import GPU_AVAILABLE
            use_gpu = GPU_AVAILABLE
            if use_gpu:
                print("GPU detectada, habilitando aceleración GPU")
            else:
                print("GPU no disponible, usando CPU")
        except ImportError:
            use_gpu = False
            print("Módulo GPU no disponible, usando CPU")
        
        self.processor = ImageProcessor(use_gpu=use_gpu)
        self.tf_manager = TransferFunctionManager()
        self.processing_thread = None
        self.current_volume = None
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("TAM - Medical Image 3D UMAP")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout()
        
        # Panel izquierdo - Controles
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout()
        
        # Tabs para organizar controles
        control_tabs = QTabWidget()
        
        # Tab de parámetros
        self.params_widget = ParametersWidget()
        control_tabs.addTab(self.params_widget, "Parámetros")
        
        # Tab de funciones de transferencia
        self.tf_widget = TransferFunctionWidget(self.tf_manager, self)
        control_tabs.addTab(self.tf_widget, "Transfer Functions")
        
        left_layout.addWidget(control_tabs)
        
        # Botones de acción
        action_group = QGroupBox("Acciones")
        action_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Cargar Volumen")
        self.process_btn = QPushButton("Procesar")
        self.batch_btn = QPushButton("Procesamiento por Lotes")
        self.export_btn = QPushButton("Exportar Resultados")
        
        self.load_btn.clicked.connect(self.load_volume)
        self.process_btn.clicked.connect(self.process_volume)
        self.batch_btn.clicked.connect(self.batch_process)
        self.export_btn.clicked.connect(self.export_results)
        
        # Inicialmente deshabilitar botones que requieren datos
        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        
        action_layout.addWidget(self.load_btn)
        action_layout.addWidget(self.process_btn)
        action_layout.addWidget(self.batch_btn)
        action_layout.addWidget(self.export_btn)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Log de estado
        log_group = QGroupBox("Log de Estado")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        left_panel.setLayout(left_layout)
        
        # Panel derecho - Visualización con pestañas
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Crear pestañas para Vista 3D, Napari y Capas
        self.view_tabs = QTabWidget()
        
        # Pestaña Napari (Nueva visualización avanzada)
        napari_tab = QWidget()
        napari_layout = QVBoxLayout()
        
        # Widget Napari para visualización avanzada
        self.napari_widget = create_napari_widget()
        napari_layout.addWidget(self.napari_widget)
        
        napari_tab.setLayout(napari_layout)
        
        # Pestaña Vista 3D (Mantener para compatibilidad)
        view_3d_tab = QWidget()
        view_3d_layout = QVBoxLayout()
        
        # Widget VTK para visualización 3D
        self.vtk_widget = VTKWidget()
        view_3d_layout.addWidget(self.vtk_widget)
        
        # Controles de visualización 3D
        viz_controls = QGroupBox("Controles de Visualización")
        viz_layout = QHBoxLayout()
        
        self.reset_view_btn = QPushButton("Resetear Vista")
        self.screenshot_btn = QPushButton("Captura de Pantalla")
        self.toggle_axes_btn = QPushButton("Mostrar/Ocultar Ejes")
        
        self.reset_view_btn.clicked.connect(self.vtk_widget.reset_camera)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.toggle_axes_btn.clicked.connect(self.vtk_widget.toggle_axes)
        
        viz_layout.addWidget(self.reset_view_btn)
        viz_layout.addWidget(self.screenshot_btn)
        viz_layout.addWidget(self.toggle_axes_btn)
        viz_layout.addStretch()
        
        viz_controls.setLayout(viz_layout)
        view_3d_layout.addWidget(viz_controls)
        
        view_3d_tab.setLayout(view_3d_layout)
        
        # Pestaña Capas (Navegación tradicional)
        layers_tab = QWidget()
        layers_layout = QVBoxLayout()
        
        # Crear widget de navegación por capas
        self.layers_widget = LayersNavigationWidget()
        layers_layout.addWidget(self.layers_widget)
        
        layers_tab.setLayout(layers_layout)
        
        # Agregar pestañas (Napari primero como principal)
        self.view_tabs.addTab(napari_tab, "🔬 Napari")
        self.view_tabs.addTab(view_3d_tab, "🎯 Vista 3D")
        self.view_tabs.addTab(layers_tab, "📋 Capas")
        
        right_layout.addWidget(self.view_tabs)
        right_panel.setLayout(right_layout)
        
        # Ensamblar layout principal
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # El panel derecho se expande
        
        central_widget.setLayout(main_layout)
        
        # Crear menús
        self.create_menus()
        
        # Barra de estado
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Listo")
        
        # Conectar señal de TF para actualizar visualización
        self.tf_widget.apply_btn.clicked.connect(self.apply_transfer_function)
        
        self.log("Aplicación TAM - Medical Image 3D UMAP iniciada correctamente")
    
    def create_menus(self):
        """Crear menús de la aplicación"""
        menubar = self.menuBar()
        
        # Menú Archivo
        file_menu = menubar.addMenu('Archivo')
        
        load_action = QAction('Cargar Volumen...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_volume)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        save_config_action = QAction('Guardar Configuración...', self)
        save_config_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction('Cargar Configuración...', self)
        load_config_action.triggered.connect(self.load_configuration)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Salir', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menú Procesamiento
        process_menu = menubar.addMenu('Procesamiento')
        
        process_action = QAction('Procesar Volumen', self)
        process_action.setShortcut('Ctrl+P')
        process_action.triggered.connect(self.process_volume)
        process_menu.addAction(process_action)
        
        batch_action = QAction('Procesamiento por Lotes...', self)
        batch_action.triggered.connect(self.batch_process)
        process_menu.addAction(batch_action)
        
        # Menú Visualización
        view_menu = menubar.addMenu('Visualización')
        
        reset_view_action = QAction('Resetear Vista', self)
        reset_view_action.triggered.connect(self.vtk_widget.reset_camera)
        view_menu.addAction(reset_view_action)
        
        screenshot_action = QAction('Captura de Pantalla...', self)
        screenshot_action.triggered.connect(self.take_screenshot)
        view_menu.addAction(screenshot_action)
        
        # Menú Ayuda
        help_menu = menubar.addMenu('Ayuda')
        
        about_action = QAction('Acerca de...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_settings(self):
        """Cargar configuración de la aplicación"""
        try:
            # Restaurar geometría de ventana
            geometry = config.get('window.geometry')
            if geometry:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
            
            # Cargar último directorio usado
            last_dir = config.get('paths.last_input_directory', '')
            if last_dir:
                self.log(f"Último directorio: {last_dir}")
                
        except Exception as e:
            self.log(f"Error cargando configuración: {e}")
    
    def save_settings(self):
        """Guardar configuración de la aplicación"""
        try:
            # Guardar geometría de ventana
            geometry = self.saveGeometry().toBase64().data().decode()
            config.set('window.geometry', geometry)
            
            # Guardar configuración
            config.save()
            
        except Exception as e:
            self.log(f"Error guardando configuración: {e}")
    
    def log(self, message: str):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.status_bar.showMessage(message)
        
        # Auto-scroll al final
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
    
    def load_volume(self):
        """Cargar volumen desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Volumen",
            config.get('paths.last_input_directory', ''),
            "Archivos de imagen (*.nii *.nii.gz *.mha *.mhd);;Todos los archivos (*.*)"
        )
        
        if file_path:
            try:
                self.log(f"Cargando volumen: {file_path}")
                print(f"Intentando cargar: {file_path}")
                
                # Verificar que el archivo existe
                if not os.path.exists(file_path):
                    raise Exception(f"El archivo no existe: {file_path}")
                
                # Cargar volumen usando ImageProcessor
                self.current_volume = self.processor.load_volume(file_path)
                
                if self.current_volume is None:
                    raise Exception("No se pudo cargar el volumen")
                
                print(f"Volumen cargado exitosamente: shape={self.current_volume.shape}, dtype={self.current_volume.dtype}")
                print(f"Rango de valores: {self.current_volume.min()} - {self.current_volume.max()}")
                
                # Actualizar parámetros con dimensiones del volumen
                params = self.params_widget.get_parameters()
                params['input_file'] = file_path
                params['dimensions'] = {
                    'rows': self.current_volume.shape[0],
                    'cols': self.current_volume.shape[1],
                    'layers': self.current_volume.shape[2]
                }
                self.params_widget.set_parameters(params)
                
                # Visualizar volumen
                print("Configurando visualización...")
                self.vtk_widget.set_volume(self.current_volume)
                
                # Configurar widget de navegación por capas
                self.layers_widget.set_volume_data(self.current_volume)
                
                # Actualizar widget de Napari si está disponible
                if hasattr(self, 'napari_widget') and self.napari_widget:
                    try:
                        self.napari_widget.set_volume_data(self.current_volume)
                        self.log("Volumen cargado en Napari")
                    except Exception as e:
                        self.log(f"Error actualizando Napari: {e}")
                
                # Habilitar procesamiento
                self.process_btn.setEnabled(True)
                
                # Guardar directorio
                config.set('paths.last_input_directory', os.path.dirname(file_path))
                
                # Aplicar función de transferencia seleccionada
                self.apply_transfer_function()
                
                self.log(f"Volumen cargado: {self.current_volume.shape}")
                
            except Exception as e:
                error_msg = f"Error cargando volumen: {e}"
                self.log(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", error_msg)
    
    def process_volume(self):
        """Procesar volumen actual"""
        if self.current_volume is None:
            QMessageBox.warning(self, "Advertencia", "Primero debe cargar un volumen")
            return
        
        params = self.params_widget.get_parameters()
        
        if not params['output_directory']:
            QMessageBox.warning(self, "Advertencia", "Debe especificar un directorio de salida")
            return
        
        try:
            self.log("Iniciando procesamiento...")
            
            # Configurar y iniciar hilo de procesamiento
            self.processing_thread = ProcessingThread(
                volume=self.current_volume,
                params=params,
                processor=self.processor
            )
            
            self.processing_thread.progress.connect(self.update_progress)
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.error.connect(self.on_processing_error)
            
            # Mostrar barra de progreso
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Deshabilitar controles durante procesamiento
            self.process_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            
            self.processing_thread.start()
            
        except Exception as e:
            self.log(f"Error iniciando procesamiento: {e}")
            QMessageBox.critical(self, "Error", f"Error iniciando procesamiento:\n{e}")
    
    def update_progress(self, value: int):
        """Actualizar barra de progreso"""
        self.progress_bar.setValue(value)
    
    def on_processing_finished(self, result):
        """Callback cuando termina el procesamiento"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        if isinstance(result, ProcessingResult):
            self.log(f"Procesamiento completado en {result.processing_time:.2f}s")
            self.log(f"Características procesadas: {', '.join(result.features_processed)}")
            
            # Mostrar resultados en visualización si están disponibles
            if hasattr(result, 'volume_data') and result.volume_data is not None:
                self.vtk_widget.set_volume(result.volume_data)
                self.apply_transfer_function()
                
                # Actualizar widget de Napari si está disponible
                if hasattr(self, 'napari_widget') and self.napari_widget:
                    try:
                        self.napari_widget.set_volume_data(result.volume_data)
                        self.log("Resultados actualizados en Napari")
                    except Exception as e:
                        self.log(f"Error actualizando Napari con resultados: {e}")
        else:
            self.log("Procesamiento completado")
    
    def on_processing_error(self, error_msg: str):
        """Callback cuando hay error en procesamiento"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.log(f"Error en procesamiento: {error_msg}")
        QMessageBox.critical(self, "Error de Procesamiento", error_msg)
    
    def apply_transfer_function(self):
        """Aplicar función de transferencia actual"""
        current_tf = self.tf_widget.get_current_transfer_function()
        if current_tf and self.vtk_widget.volume_actor:
            try:
                # Obtener propiedades del volumen desde el actor
                volume_property = self.vtk_widget.volume_actor.GetProperty()
                color_func = volume_property.GetRGBTransferFunction()
                opacity_func = volume_property.GetScalarOpacity()
                
                # Limpiar funciones actuales
                color_func.RemoveAllPoints()
                opacity_func.RemoveAllPoints()
                
                # Agregar puntos de color
                for point, r, g, b in current_tf.get_color_points():
                    color_func.AddRGBPoint(point, r, g, b)
                
                # Agregar puntos de opacidad
                for point, opacity in current_tf.get_opacity_points():
                    opacity_func.AddPoint(point, opacity)
                
                # Aplicar propiedades adicionales de shading y lighting
                volume_property.SetShade(current_tf.shading)
                volume_property.SetAmbient(current_tf.ambient)
                volume_property.SetDiffuse(current_tf.diffuse)
                volume_property.SetSpecular(current_tf.specular)
                
                # Configurar interpolación
                if hasattr(current_tf, 'interpolation'):
                    if current_tf.interpolation.lower() == 'linear':
                        volume_property.SetInterpolationTypeToLinear()
                    elif current_tf.interpolation.lower() == 'nearest':
                        volume_property.SetInterpolationTypeToNearest()
                
                # Actualizar visualización
                self.vtk_widget.render_window.Render()
                
                self.log(f"Función de transferencia '{current_tf.name}' aplicada con shading={current_tf.shading}, ambient={current_tf.ambient}, diffuse={current_tf.diffuse}, specular={current_tf.specular}")
                
            except Exception as e:
                self.log(f"Error aplicando función de transferencia: {e}")
                import traceback
                traceback.print_exc()
    
    def batch_process(self):
        """Procesamiento por lotes"""
        dialog = BatchProcessDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # Implementar procesamiento por lotes
            self.log("Procesamiento por lotes iniciado")
    
    def export_results(self):
        """Exportar resultados"""
        if self.current_volume is None:
            QMessageBox.warning(self, "Advertencia", "No hay datos para exportar")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Resultados",
            "resultados.mha", "Archivos MHA (*.mha);;Archivos NIfTI (*.nii)"
        )
        
        if file_path:
            try:
                # Exportar usando ImageProcessor
                self.processor.save_volume(self.current_volume, file_path)
                self.log(f"Resultados exportados: {file_path}")
                QMessageBox.information(self, "Éxito", "Resultados exportados correctamente")
                
            except Exception as e:
                self.log(f"Error exportando: {e}")
                QMessageBox.critical(self, "Error", f"Error exportando resultados:\n{e}")
    
    def take_screenshot(self):
        """Tomar captura de pantalla de la visualización"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Captura",
            "captura.png", "Archivos PNG (*.png);;Archivos JPG (*.jpg)"
        )
        
        if file_path:
            try:
                self.vtk_widget.save_screenshot(file_path)
                self.log(f"Captura guardada: {file_path}")
                QMessageBox.information(self, "Éxito", "Captura guardada correctamente")
                
            except Exception as e:
                self.log(f"Error guardando captura: {e}")
                QMessageBox.critical(self, "Error", f"Error guardando captura:\n{e}")
    
    def save_configuration(self):
        """Guardar configuración actual"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Configuración",
            "configuracion.json", "Archivos JSON (*.json)"
        )
        
        if file_path:
            try:
                # Obtener configuración actual
                current_config = {
                    'parameters': self.params_widget.get_parameters(),
                    'transfer_function': self.tf_widget.tf_combo.currentText(),
                    'window_geometry': self.saveGeometry().toBase64().data().decode()
                }
                
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_config, f, indent=2, ensure_ascii=False)
                
                self.log(f"Configuración guardada: {file_path}")
                QMessageBox.information(self, "Éxito", "Configuración guardada correctamente")
                
            except Exception as e:
                self.log(f"Error guardando configuración: {e}")
                QMessageBox.critical(self, "Error", f"Error guardando configuración:\n{e}")
    
    def load_configuration(self):
        """Cargar configuración desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Configuración",
            "", "Archivos JSON (*.json)"
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                
                # Aplicar configuración
                if 'parameters' in saved_config:
                    self.params_widget.set_parameters(saved_config['parameters'])
                
                if 'transfer_function' in saved_config:
                    tf_name = saved_config['transfer_function']
                    index = self.tf_widget.tf_combo.findText(tf_name)
                    if index >= 0:
                        self.tf_widget.tf_combo.setCurrentIndex(index)
                
                if 'window_geometry' in saved_config:
                    geometry = QByteArray.fromBase64(saved_config['window_geometry'].encode())
                    self.restoreGeometry(geometry)
                
                self.log(f"Configuración cargada: {file_path}")
                QMessageBox.information(self, "Éxito", "Configuración cargada correctamente")
                
            except Exception as e:
                self.log(f"Error cargando configuración: {e}")
                QMessageBox.critical(self, "Error", f"Error cargando configuración:\n{e}")
    
    def show_about(self):
        """Mostrar diálogo Acerca de"""
        about_text = """
        <h2>TAM - Medical Image 3D UMAP</h2>
        <p><b>Versión:</b> 1.0.0</p>
        <p><b>Descripción:</b> Análisis de Imágenes Médicas 3D con Reducción de Dimensionalidad UMAP</p>
        <p><b>Características:</b></p>
        <ul>
            <li>Visualización 3D interactiva con VTK</li>
            <li>Procesamiento de características (Gradiente, Laplaciano, Curvatura, etc.)</li>
            <li>Funciones de transferencia personalizables</li>
            <li>Reducción de dimensionalidad con UMAP optimizado</li>
            <li>Integración con Napari para visualización científica</li>
            <li>Procesamiento por lotes</li>
            <li>Soporte para formatos NIfTI y MHA</li>
        </ul>
        <p><b>Desarrollado por:</b> Víctor Quintero para la asignatura TAM - Universidad Nacional de Colombia Sede Manizales</p>
        <p><b>Tecnologías:</b> Python, PyQt5, VTK, NumPy, SciPy, UMAP</p>
        <p><i>Migración optimizada del proyecto Medico3D original</i></p>
        """
        
        QMessageBox.about(self, "Acerca de TAM", about_text)
    
    def closeEvent(self, event):
        """Evento al cerrar la aplicación"""
        # Detener procesamiento si está en curso
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirmar Salida",
                "Hay un procesamiento en curso. ¿Desea salir de todas formas?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                self.processing_thread.terminate()
                self.processing_thread.wait()
        
        # Guardar configuración
        self.save_settings()
        
        event.accept()


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Configurar aplicación
    app.setApplicationName("TAM - Medical Image 3D UMAP")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TAM")
    
    try:
        # Crear y mostrar ventana principal
        window = Medico3DMainWindow()
        window.show()
        
        # Ejecutar aplicación
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
