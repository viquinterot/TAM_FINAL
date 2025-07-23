"""
Módulo para manejo de funciones de transferencia
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import json


class TransferFunctionPoint:
    """Punto de función de transferencia"""
    
    def __init__(self, point: float, red: float = 0.0, green: float = 0.0, 
                 blue: float = 0.0, opacity: float = 1.0):
        self.point = point
        self.red = red
        self.green = green
        self.blue = blue
        self.opacity = opacity
    
    def to_dict(self) -> Dict:
        return {
            'point': self.point,
            'red': self.red,
            'green': self.green,
            'blue': self.blue,
            'opacity': self.opacity
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransferFunctionPoint':
        return cls(
            point=data['point'],
            red=data.get('red', 0.0),
            green=data.get('green', 0.0),
            blue=data.get('blue', 0.0),
            opacity=data.get('opacity', 1.0)
        )


class TransferFunction:
    """Clase para manejar funciones de transferencia"""
    
    def __init__(self, name: str = "Default", description: str = ""):
        self.name = name
        self.description = description
        self.color_points: List[TransferFunctionPoint] = []
        self.opacity_points: List[TransferFunctionPoint] = []
        
        # Propiedades adicionales del XML
        self.interpolation = "Linear"
        self.shading = True
        self.ambient = 0.1
        self.diffuse = 0.9
        self.specular = 0.2
        
    def add_color_point(self, point: float, red: float, green: float, blue: float):
        """Añadir punto de color"""
        tf_point = TransferFunctionPoint(point, red, green, blue)
        self.color_points.append(tf_point)
        self.color_points.sort(key=lambda p: p.point)
    
    def add_opacity_point(self, point: float, opacity: float):
        """Añadir punto de opacidad"""
        tf_point = TransferFunctionPoint(point, opacity=opacity)
        self.opacity_points.append(tf_point)
        self.opacity_points.sort(key=lambda p: p.point)
    
    def get_color_points(self) -> List[Tuple[float, float, float, float]]:
        """Obtener puntos de color como tuplas (point, r, g, b)"""
        return [(p.point, p.red, p.green, p.blue) for p in self.color_points]
    
    def get_opacity_points(self) -> List[Tuple[float, float]]:
        """Obtener puntos de opacidad como tuplas (point, opacity)"""
        return [(p.point, p.opacity) for p in self.opacity_points]
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario para serialización"""
        return {
            'name': self.name,
            'description': self.description,
            'color_points': [p.to_dict() for p in self.color_points],
            'opacity_points': [p.to_dict() for p in self.opacity_points],
            'interpolation': self.interpolation,
            'shading': self.shading,
            'ambient': self.ambient,
            'diffuse': self.diffuse,
            'specular': self.specular
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransferFunction':
        """Crear desde diccionario"""
        tf = cls(data['name'], data.get('description', ''))
        
        for point_data in data.get('color_points', []):
            point = TransferFunctionPoint.from_dict(point_data)
            tf.color_points.append(point)
        
        for point_data in data.get('opacity_points', []):
            point = TransferFunctionPoint.from_dict(point_data)
            tf.opacity_points.append(point)
        
        tf.interpolation = data.get('interpolation', 'Linear')
        tf.shading = data.get('shading', True)
        tf.ambient = data.get('ambient', 0.1)
        tf.diffuse = data.get('diffuse', 0.9)
        tf.specular = data.get('specular', 0.2)
        
        return tf


class TransferFunctionManager:
    """Gestor de funciones de transferencia"""
    
    def __init__(self):
        self.transfer_functions: Dict[str, TransferFunction] = {}
        self.load_default_functions()
    
    def load_default_functions(self):
        """Cargar funciones de transferencia por defecto"""
        
        # Función por defecto
        default_tf = TransferFunction("Por defecto", "Función de transferencia por defecto")
        default_tf.add_color_point(0, 0.0, 0.0, 0.0)
        default_tf.add_color_point(64, 0.5, 0.0, 0.0)
        default_tf.add_color_point(128, 1.0, 0.5, 0.0)
        default_tf.add_color_point(192, 1.0, 1.0, 0.5)
        default_tf.add_color_point(255, 1.0, 1.0, 1.0)
        
        default_tf.add_opacity_point(0, 0.0)
        default_tf.add_opacity_point(64, 0.1)
        default_tf.add_opacity_point(128, 0.3)
        default_tf.add_opacity_point(192, 0.6)
        default_tf.add_opacity_point(255, 0.8)
        
        self.transfer_functions["Por defecto"] = default_tf
        
        # Hot Metal
        hot_metal = TransferFunction("Hot Metal", "Función de transferencia Hot Metal")
        hot_metal.add_color_point(0, 0.0, 0.0, 0.0)
        hot_metal.add_color_point(85, 0.5, 0.0, 0.0)
        hot_metal.add_color_point(170, 1.0, 0.5, 0.0)
        hot_metal.add_color_point(255, 1.0, 1.0, 1.0)
        
        hot_metal.add_opacity_point(0, 0.0)
        hot_metal.add_opacity_point(85, 0.2)
        hot_metal.add_opacity_point(170, 0.6)
        hot_metal.add_opacity_point(255, 1.0)
        
        self.transfer_functions["Hot Metal"] = hot_metal
        
        # CT Grey Level
        ct_grey = TransferFunction("CT Grey Level", "Función de transferencia CT Grey Level")
        ct_grey.add_color_point(-1024, 0.0, 0.0, 0.0)
        ct_grey.add_color_point(-512, 0.25, 0.25, 0.25)
        ct_grey.add_color_point(0, 0.5, 0.5, 0.5)
        ct_grey.add_color_point(512, 0.75, 0.75, 0.75)
        ct_grey.add_color_point(1024, 1.0, 1.0, 1.0)
        
        ct_grey.add_opacity_point(-1024, 0.0)
        ct_grey.add_opacity_point(-512, 0.1)
        ct_grey.add_opacity_point(0, 0.3)
        ct_grey.add_opacity_point(512, 0.6)
        ct_grey.add_opacity_point(1024, 0.9)
        
        self.transfer_functions["CT Grey Level"] = ct_grey
        
        # Osirix Airways II
        airways = TransferFunction("Osirix Airways II", "Función de transferencia Osirix Airways II")
        airways.add_color_point(-742, 0.0, 0.6, 0.7)
        airways.add_color_point(-684, 1.0, 1.0, 0.0)
        airways.add_color_point(-400, 1.0, 0.5, 0.0)
        airways.add_color_point(0, 1.0, 0.0, 0.0)
        airways.add_color_point(400, 0.0, 1.0, 0.0)
        
        airways.add_opacity_point(-742, 0.0)
        airways.add_opacity_point(-684, 0.2)
        airways.add_opacity_point(-400, 0.4)
        airways.add_opacity_point(0, 0.6)
        airways.add_opacity_point(400, 0.8)
        
        self.transfer_functions["Osirix Airways II"] = airways
    
    def get_transfer_function(self, name: str) -> Optional[TransferFunction]:
        """Obtener función de transferencia por nombre"""
        return self.transfer_functions.get(name)
    
    def get_function_names(self) -> List[str]:
        """Obtener lista de nombres de funciones disponibles"""
        return list(self.transfer_functions.keys())
    
    def add_transfer_function(self, tf: TransferFunction):
        """Añadir nueva función de transferencia"""
        self.transfer_functions[tf.name] = tf
    
    def load_from_xml(self, xml_path: str):
        """Cargar funciones de transferencia desde XML (formato original)"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for tf_element in root.findall('.//TransferFunction'):
                # Intentar ambos formatos: 'n' y 'Name'
                name_element = tf_element.find('n')
                if name_element is None:
                    name_element = tf_element.find('Name')
                
                name = name_element.text if name_element is not None else "Unknown"
                
                description_element = tf_element.find('Description')
                description = description_element.text if description_element is not None else ""
                
                tf = TransferFunction(name, description)
                
                # Cargar propiedades adicionales del XML
                interpolation_elem = tf_element.find('Interpolation')
                if interpolation_elem is not None:
                    tf.interpolation = interpolation_elem.text
                
                shading_elem = tf_element.find('Shading')
                if shading_elem is not None:
                    tf.shading = shading_elem.text.lower() == 'true'
                
                ambient_elem = tf_element.find('Ambient')
                if ambient_elem is not None:
                    tf.ambient = float(ambient_elem.text)
                
                diffuse_elem = tf_element.find('Diffuse')
                if diffuse_elem is not None:
                    tf.diffuse = float(diffuse_elem.text)
                
                specular_elem = tf_element.find('Specular')
                if specular_elem is not None:
                    tf.specular = float(specular_elem.text)
                
                # Cargar puntos de color
                color_element = tf_element.find('Color')
                if color_element is not None:
                    for point_element in color_element.findall('TFPoint'):
                        point = float(point_element.find('p').text)
                        red = float(point_element.find('r').text)
                        green = float(point_element.find('g').text)
                        blue = float(point_element.find('b').text)
                        tf.add_color_point(point, red, green, blue)
                
                # Cargar puntos de opacidad
                opacity_element = tf_element.find('Opacity')
                if opacity_element is not None:
                    for point_element in opacity_element.findall('TFOpacity'):
                        point = float(point_element.find('p').text)
                        opacity = float(point_element.find('o').text)
                        tf.add_opacity_point(point, opacity)
                
                self.transfer_functions[name] = tf
                print(f"Cargada función de transferencia: {name}")
                
        except Exception as e:
            print(f"Error cargando XML {xml_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def save_to_json(self, json_path: str):
        """Guardar funciones de transferencia en formato JSON"""
        data = {
            'transfer_functions': {
                name: tf.to_dict() for name, tf in self.transfer_functions.items()
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, json_path: str):
        """Cargar funciones de transferencia desde JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, tf_data in data.get('transfer_functions', {}).items():
                tf = TransferFunction.from_dict(tf_data)
                self.transfer_functions[name] = tf
                
        except Exception as e:
            print(f"Error cargando JSON: {e}")
    
    def load_legacy_xml_directory(self, xml_dir: str):
        """Cargar todas las funciones de transferencia de un directorio XML"""
        if not os.path.exists(xml_dir):
            return
        
        for filename in os.listdir(xml_dir):
            if filename.endswith('.xml'):
                xml_path = os.path.join(xml_dir, filename)
                self.load_from_xml(xml_path)