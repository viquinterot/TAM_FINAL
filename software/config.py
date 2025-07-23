"""
Módulo de configuración para Medico3D
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from pathlib import Path


class Medico3DConfig:
    """Clase para manejar la configuración de Medico3D"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Usar directorio de configuración del usuario
            self.config_dir = Path.home() / '.medico3d'
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / 'config.json'
        self.legacy_ini_file = self.config_dir / 'ConseilAnalizador3D.ini'
        
        # Configuración por defecto
        self.default_config = {
            'database_type': 'JSON',
            'python_paths': {
                'python2.7': '',
                'python3.6': '',
                'current': ''
            },
            'paths': {
                'scripts_python': '',
                'transfer_functions': '',
                'data_directory': '',
                'output_directory': ''
            },
            'processing': {
                'default_dimensions': {
                    'f': 256,
                    'c': 256,
                    's': 256
                },
                'default_parameters': {
                    'e': 0.5,
                    'Thr': 0.1,
                    'ThrI': 0.1,
                    'Devs': -1
                },
                'parallel_processing': True,
                'max_workers': 4
            },
            'visualization': {
                'default_background_color': [0.1, 0.1, 0.2],
                'default_transfer_function': 'Por defecto',
                'auto_render': True,
                'quality': 'high'
            },
            'ui': {
                'theme': 'dark',
                'language': 'es',
                'window_geometry': None,
                'splitter_sizes': [350, 1050]
            },
            'recent_files': [],
            'max_recent_files': 10
        }
        
        self.config = self.default_config.copy()
        self.load_config()
    
    def load_config(self):
        """Cargar configuración desde archivo"""
        # Intentar cargar desde JSON primero
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                return
            except Exception as e:
                print(f"Error cargando configuración JSON: {e}")
        
        # Si no existe JSON, intentar migrar desde INI legacy
        self.migrate_from_legacy_ini()
    
    def migrate_from_legacy_ini(self):
        """Migrar configuración desde archivo INI legacy"""
        legacy_paths = [
            self.legacy_ini_file,
            Path('../ConseilAnalizador3D/ConseilAnalizador3D.ini'),
            Path('ConseilAnalizador3D.ini')
        ]
        
        for ini_path in legacy_paths:
            if ini_path.exists():
                try:
                    # Leer archivo INI manualmente ya que no tiene secciones
                    with open(ini_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    legacy_config = {}
                    for line in lines:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            legacy_config[key.strip()] = value.strip()
                    
                    # Migrar configuración básica
                    if 'BASEDATOS' in legacy_config:
                        self.config['database_type'] = legacy_config['BASEDATOS']
                    
                    if 'PYTHON2.7PATH' in legacy_config:
                        self.config['python_paths']['python2.7'] = legacy_config['PYTHON2.7PATH']
                    
                    if 'PYTHON3.6PATH' in legacy_config:
                        self.config['python_paths']['python3.6'] = legacy_config['PYTHON3.6PATH']
                    
                    print(f"Configuración migrada desde: {ini_path}")
                    self.save_config()
                    break
                    
                except Exception as e:
                    print(f"Error migrando configuración desde {ini_path}: {e}")
    
    def save_config(self):
        """Guardar configuración actual"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando configuración: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración usando notación de punto"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Establecer valor de configuración usando notación de punto"""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
        self.save_config()
    
    def get_python_path(self) -> str:
        """Obtener ruta de Python actual"""
        current = self.get('python_paths.current')
        if current and os.path.exists(current):
            return current
        
        # Intentar encontrar Python automáticamente
        import sys
        return sys.executable
    
    def get_scripts_path(self) -> str:
        """Obtener ruta de scripts Python"""
        scripts_path = self.get('paths.scripts_python')
        if scripts_path and os.path.exists(scripts_path):
            return scripts_path
        
        # Ruta por defecto relativa
        default_path = Path(__file__).parent.parent / 'ConseilAnalizador3D' / 'ScriptsPython'
        if default_path.exists():
            self.set('paths.scripts_python', str(default_path))
            return str(default_path)
        
        return ''
    
    def get_transfer_functions_path(self) -> str:
        """Obtener ruta de funciones de transferencia"""
        tf_path = self.get('paths.transfer_functions')
        if tf_path and os.path.exists(tf_path):
            return tf_path
        
        # Ruta por defecto relativa
        default_path = Path(__file__).parent.parent / 'ConseilAnalizador3D' / 'TransferFunctions'
        if default_path.exists():
            self.set('paths.transfer_functions', str(default_path))
            return str(default_path)
        
        return ''
    
    def add_recent_file(self, file_path: str):
        """Añadir archivo a la lista de recientes"""
        recent_files = self.get('recent_files', [])
        
        # Remover si ya existe
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Añadir al principio
        recent_files.insert(0, file_path)
        
        # Limitar número máximo
        max_files = self.get('max_recent_files', 10)
        recent_files = recent_files[:max_files]
        
        self.set('recent_files', recent_files)
    
    def get_recent_files(self) -> list:
        """Obtener lista de archivos recientes"""
        recent_files = self.get('recent_files', [])
        # Filtrar archivos que ya no existen
        existing_files = [f for f in recent_files if os.path.exists(f)]
        
        if len(existing_files) != len(recent_files):
            self.set('recent_files', existing_files)
        
        return existing_files
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Obtener parámetros por defecto para procesamiento"""
        return {
            'f': self.get('processing.default_dimensions.f', 256),
            'c': self.get('processing.default_dimensions.c', 256),
            's': self.get('processing.default_dimensions.s', 256),
            'e': self.get('processing.default_parameters.e', 0.5),
            'Thr': self.get('processing.default_parameters.Thr', 0.1),
            'ThrI': self.get('processing.default_parameters.ThrI', 0.1),
            'Devs': self.get('processing.default_parameters.Devs', -1)
        }
    
    def update_default_parameters(self, parameters: Dict[str, Any]):
        """Actualizar parámetros por defecto"""
        for key, value in parameters.items():
            if key in ['f', 'c', 's']:
                self.set(f'processing.default_dimensions.{key}', value)
            elif key in ['e', 'Thr', 'ThrI', 'Devs']:
                self.set(f'processing.default_parameters.{key}', value)
    
    def reset_to_defaults(self):
        """Resetear configuración a valores por defecto"""
        self.config = self.default_config.copy()
        self.save_config()
    
    def export_config(self, export_path: str):
        """Exportar configuración a archivo"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error exportando configuración: {e}")
    
    def import_config(self, import_path: str):
        """Importar configuración desde archivo"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
                self.config.update(imported_config)
                self.save_config()
        except Exception as e:
            print(f"Error importando configuración: {e}")


# Instancia global de configuración
config = Medico3DConfig()