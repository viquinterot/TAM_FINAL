#!/usr/bin/env python3
"""
Napari Widget para Medico3D
Widget de visualización avanzada usando Napari para imágenes médicas 3D

"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import warnings

# Suprimir warnings de Napari durante la importación
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    import napari
    from napari.qt import QtViewer
    from magicgui import magic_factory, magicgui
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
        QLabel, QSlider, QComboBox, QPushButton, QSpinBox,
        QCheckBox, QFormLayout, QTabWidget
    )
    from qtpy.QtCore import Qt, Signal
    NAPARI_AVAILABLE = True
except ImportError as e:
    print(f"Napari no disponible: {e}")
    NAPARI_AVAILABLE = False
    # Fallback imports
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
        QLabel, QSlider, QComboBox, QPushButton, QSpinBox,
        QCheckBox, QFormLayout, QTabWidget
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal


class NapariVisualizationWidget(QWidget):
    """Widget de visualización avanzada usando Napari"""
    
    # Señales
    layer_changed = Signal(int)
    volume_loaded = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = None
        self.volume_data = None
        self.feature_layers = {}
        self.current_colormap = 'gray'
        self.init_ui()
        
        if NAPARI_AVAILABLE:
            self.init_napari()
        else:
            self.init_fallback()
    
    def init_ui(self):
        """Inicializar interfaz de usuario"""
        layout = QVBoxLayout()
        
        # Controles superiores
        controls_group = QGroupBox("Controles de Visualización")
        controls_layout = QFormLayout()
        
        # Selector de colormap
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            'gray', 'viridis', 'plasma', 'inferno', 'magma',
            'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'
        ])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        controls_layout.addRow("Mapa de color:", self.colormap_combo)
        
        # Control de opacidad
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        controls_layout.addRow("Opacidad:", self.opacity_slider)
        
        # Control de contraste
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        controls_layout.addRow("Contraste:", self.contrast_slider)
        
        # Checkbox para mostrar/ocultar capas
        self.show_volume_cb = QCheckBox("Mostrar volumen principal")
        self.show_volume_cb.setChecked(True)
        self.show_volume_cb.toggled.connect(self.on_volume_visibility_changed)
        controls_layout.addRow(self.show_volume_cb)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Área principal para Napari viewer
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout()
        self.viewer_container.setLayout(self.viewer_layout)
        layout.addWidget(self.viewer_container, 1)  # Se expande
        
        # Controles inferiores - Gestión de capas
        layers_group = QGroupBox("Gestión de Capas")
        layers_layout = QVBoxLayout()
        
        # Botones de gestión de capas
        layer_buttons_layout = QHBoxLayout()
        
        self.clear_layers_btn = QPushButton("Limpiar Capas")
        self.export_layer_btn = QPushButton("Exportar Capa Actual")
        self.screenshot_btn = QPushButton("Captura de Pantalla")
        
        self.clear_layers_btn.clicked.connect(self.clear_feature_layers)
        self.export_layer_btn.clicked.connect(self.export_current_layer)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        layer_buttons_layout.addWidget(self.clear_layers_btn)
        layer_buttons_layout.addWidget(self.export_layer_btn)
        layer_buttons_layout.addWidget(self.screenshot_btn)
        layer_buttons_layout.addStretch()
        
        layers_layout.addLayout(layer_buttons_layout)
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)
        
        self.setLayout(layout)
    
    def init_napari(self):
        """Inicializar viewer de Napari"""
        try:
            # Crear viewer de Napari
            self.viewer = napari.Viewer(show=False)
            
            # Configurar el viewer
            self.viewer.theme = 'dark'
            self.viewer.axes.visible = True
            self.viewer.scale_bar.visible = True
            
            # Obtener el widget Qt del viewer
            napari_widget = self.viewer.window.qt_viewer
            
            # Agregar al layout
            self.viewer_layout.addWidget(napari_widget)
            
            # Conectar eventos
            self.viewer.layers.events.inserted.connect(self.on_layer_inserted)
            self.viewer.layers.events.removed.connect(self.on_layer_removed)
            
            print("Napari viewer inicializado correctamente")
            
        except Exception as e:
            print(f"Error inicializando Napari: {e}")
            self.init_fallback()
    
    def init_fallback(self):
        """Inicializar widget de fallback cuando Napari no está disponible"""
        fallback_label = QLabel("Napari no disponible\nUsando modo de compatibilidad")
        fallback_label.setAlignment(Qt.AlignCenter)
        fallback_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: white;
                font-size: 14px;
                padding: 20px;
                border: 2px dashed #555;
            }
        """)
        self.viewer_layout.addWidget(fallback_label)
    
    def set_volume_data(self, volume_data: np.ndarray, name: str = "Volume"):
        """Establecer datos del volumen principal"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            print("Napari no disponible para mostrar volumen")
            return
        
        try:
            self.volume_data = volume_data
            
            # Limpiar capas existentes del volumen principal
            layers_to_remove = [layer for layer in self.viewer.layers if layer.name == name]
            for layer in layers_to_remove:
                self.viewer.layers.remove(layer)
            
            # Agregar volumen como capa de imagen
            self.viewer.add_image(
                volume_data,
                name=name,
                colormap=self.current_colormap,
                opacity=self.opacity_slider.value() / 100.0,
                contrast_limits=self._calculate_contrast_limits(volume_data)
            )
            
            # Ajustar vista
            self.viewer.reset_view()
            
            # Emitir señal
            self.volume_loaded.emit(volume_data)
            
            print(f"Volumen cargado: {volume_data.shape}, dtype: {volume_data.dtype}")
            
        except Exception as e:
            print(f"Error cargando volumen en Napari: {e}")
    
    def add_feature_layer(self, feature_data: np.ndarray, name: str, 
                         colormap: str = 'viridis', opacity: float = 0.7):
        """Agregar capa de características"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            print("Napari no disponible para agregar capa")
            return
        
        try:
            # Remover capa existente con el mismo nombre
            if name in self.feature_layers:
                self.remove_feature_layer(name)
            
            # Agregar nueva capa
            layer = self.viewer.add_image(
                feature_data,
                name=name,
                colormap=colormap,
                opacity=opacity,
                contrast_limits=self._calculate_contrast_limits(feature_data),
                blending='additive'
            )
            
            # Guardar referencia
            self.feature_layers[name] = layer
            
            print(f"Capa de características agregada: {name}")
            
        except Exception as e:
            print(f"Error agregando capa {name}: {e}")
    
    def remove_feature_layer(self, name: str):
        """Remover capa de características"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        try:
            if name in self.feature_layers:
                layer = self.feature_layers[name]
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
                del self.feature_layers[name]
                print(f"Capa removida: {name}")
        except Exception as e:
            print(f"Error removiendo capa {name}: {e}")
    
    def clear_feature_layers(self):
        """Limpiar todas las capas de características"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        try:
            # Remover todas las capas excepto el volumen principal
            layers_to_remove = []
            for layer in self.viewer.layers:
                if layer.name != "Volume":
                    layers_to_remove.append(layer)
            
            for layer in layers_to_remove:
                self.viewer.layers.remove(layer)
            
            self.feature_layers.clear()
            print("Capas de características limpiadas")
            
        except Exception as e:
            print(f"Error limpiando capas: {e}")
    
    def update_umap_visualization(self, umap_result: np.ndarray, feature_names: List[str]):
        """Actualizar visualización con resultados de UMAP"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        try:
            # Limpiar capas UMAP anteriores
            umap_layers = [name for name in self.feature_layers.keys() if name.startswith('UMAP_')]
            for layer_name in umap_layers:
                self.remove_feature_layer(layer_name)
            
            # Agregar resultado de UMAP como capa
            if umap_result.ndim == 4 and umap_result.shape[-1] >= 2:
                # Usar las primeras dos componentes de UMAP para crear una visualización RGB
                umap_rgb = np.zeros((*umap_result.shape[:3], 3))
                umap_rgb[..., 0] = (umap_result[..., 0] - umap_result[..., 0].min()) / (umap_result[..., 0].max() - umap_result[..., 0].min())
                umap_rgb[..., 1] = (umap_result[..., 1] - umap_result[..., 1].min()) / (umap_result[..., 1].max() - umap_result[..., 1].min())
                
                self.viewer.add_image(
                    umap_rgb,
                    name="UMAP_Embedding",
                    opacity=0.6,
                    blending='additive'
                )
            
            # Agregar características individuales si están disponibles
            for i, feature_name in enumerate(feature_names[:min(len(feature_names), 5)]):  # Limitar a 5 características
                if i < umap_result.shape[-1]:
                    feature_data = umap_result[..., i]
                    self.add_feature_layer(
                        feature_data,
                        f"UMAP_{feature_name}",
                        colormap=['viridis', 'plasma', 'inferno', 'magma', 'hot'][i % 5],
                        opacity=0.5
                    )
            
            print(f"Visualización UMAP actualizada con {len(feature_names)} características")
            
        except Exception as e:
            print(f"Error actualizando visualización UMAP: {e}")
    
    def _calculate_contrast_limits(self, data: np.ndarray) -> Tuple[float, float]:
        """Calcular límites de contraste automáticamente"""
        try:
            # Usar percentiles para evitar outliers
            p1, p99 = np.percentile(data, [1, 99])
            return (float(p1), float(p99))
        except:
            return (float(data.min()), float(data.max()))
    
    def on_colormap_changed(self, colormap_name: str):
        """Manejar cambio de colormap"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        self.current_colormap = colormap_name
        
        # Actualizar colormap del volumen principal
        for layer in self.viewer.layers:
            if layer.name == "Volume":
                layer.colormap = colormap_name
                break
    
    def on_opacity_changed(self, value: int):
        """Manejar cambio de opacidad"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        opacity = value / 100.0
        
        # Actualizar opacidad del volumen principal
        for layer in self.viewer.layers:
            if layer.name == "Volume":
                layer.opacity = opacity
                break
    
    def on_contrast_changed(self, value: int):
        """Manejar cambio de contraste"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        contrast_factor = value / 100.0
        
        # Actualizar contraste del volumen principal
        for layer in self.viewer.layers:
            if layer.name == "Volume" and hasattr(layer, 'contrast_limits'):
                original_limits = self._calculate_contrast_limits(self.volume_data)
                range_val = original_limits[1] - original_limits[0]
                center = (original_limits[1] + original_limits[0]) / 2
                
                new_range = range_val / contrast_factor
                new_limits = (center - new_range/2, center + new_range/2)
                layer.contrast_limits = new_limits
                break
    
    def on_volume_visibility_changed(self, visible: bool):
        """Manejar cambio de visibilidad del volumen"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        for layer in self.viewer.layers:
            if layer.name == "Volume":
                layer.visible = visible
                break
    
    def on_layer_inserted(self, event):
        """Callback cuando se inserta una nueva capa"""
        print(f"Capa insertada: {event.value.name}")
    
    def on_layer_removed(self, event):
        """Callback cuando se remueve una capa"""
        print(f"Capa removida: {event.value.name}")
    
    def export_current_layer(self):
        """Exportar la capa actualmente seleccionada"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        try:
            if len(self.viewer.layers) > 0:
                current_layer = self.viewer.layers.selection.active
                if current_layer:
                    # Aquí se podría implementar la exportación
                    print(f"Exportando capa: {current_layer.name}")
                    # TODO: Implementar exportación real
        except Exception as e:
            print(f"Error exportando capa: {e}")
    
    def take_screenshot(self):
        """Tomar captura de pantalla del viewer"""
        if not NAPARI_AVAILABLE or self.viewer is None:
            return
        
        try:
            # Usar la funcionalidad nativa de Napari para capturas
            screenshot = self.viewer.screenshot(canvas_only=True)
            
            # Aquí se podría guardar la imagen
            print("Captura de pantalla tomada")
            # TODO: Implementar guardado de captura
            
            return screenshot
            
        except Exception as e:
            print(f"Error tomando captura: {e}")
    
    def reset_view(self):
        """Resetear vista del viewer"""
        if NAPARI_AVAILABLE and self.viewer is not None:
            self.viewer.reset_view()
    
    def get_viewer(self):
        """Obtener referencia al viewer de Napari"""
        return self.viewer
    
    def is_napari_available(self) -> bool:
        """Verificar si Napari está disponible"""
        return NAPARI_AVAILABLE and self.viewer is not None


# Widget de compatibilidad para cuando Napari no está disponible
class FallbackVisualizationWidget(QWidget):
    """Widget de fallback cuando Napari no está disponible"""
    
    volume_loaded = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        info_label = QLabel("""
        <h3>Modo de Compatibilidad</h3>
        <p>Napari no está disponible en este sistema.</p>
        <p>Para obtener funcionalidades avanzadas de visualización, instale Napari:</p>
        <code>pip install napari[all]</code>
        <p>Mientras tanto, puede usar la pestaña "Vista 3D" para visualización básica.</p>
        """)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        
        layout.addWidget(info_label)
        self.setLayout(layout)
    
    def set_volume_data(self, volume_data, name="Volume"):
        """Método stub para compatibilidad"""
        print(f"Volumen recibido en modo fallback: {volume_data.shape}")
        self.volume_loaded.emit(volume_data)
    
    def add_feature_layer(self, feature_data, name, colormap='viridis', opacity=0.7):
        """Método stub para compatibilidad"""
        print(f"Capa de características recibida en modo fallback: {name}")
    
    def clear_feature_layers(self):
        """Método stub para compatibilidad"""
        print("Limpieza de capas solicitada en modo fallback")
    
    def update_umap_visualization(self, umap_result, feature_names):
        """Método stub para compatibilidad"""
        print("Actualización UMAP solicitada en modo fallback")
    
    def reset_view(self):
        """Método stub para compatibilidad"""
        pass
    
    def is_napari_available(self):
        """Siempre retorna False para el widget de fallback"""
        return False


def create_napari_widget(parent=None) -> QWidget:
    """Factory function para crear el widget apropiado"""
    if NAPARI_AVAILABLE:
        return NapariVisualizationWidget(parent)
    else:
        return FallbackVisualizationWidget(parent)
