"""
Script de instalación y configuración para Medico3D Python
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def install_requirements():
    """Instalar dependencias"""
    print("\n📦 Instalando dependencias...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: No se encontró requirements.txt")
        return False
    
    try:
        # Actualizar pip primero
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Instalar dependencias
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False


def check_vtk_installation():
    """Verificar instalación de VTK"""
    try:
        import vtk
        print(f"✅ VTK {vtk.vtkVersion.GetVTKVersion()} - OK")
        return True
    except ImportError:
        print("❌ Error: VTK no está instalado correctamente")
        return False


def check_pyqt_installation():
    """Verificar instalación de PyQt6"""
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 - OK")
        return True
    except ImportError:
        print("❌ Error: PyQt6 no está instalado correctamente")
        return False


def setup_directories():
    """Configurar directorios necesarios"""
    print("\n📁 Configurando directorios...")
    
    base_dir = Path(__file__).parent
    
    directories = [
        base_dir / "config",
        base_dir / "output",
        base_dir / "temp",
        base_dir / "transfer_functions",
        base_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"   📂 {directory.name}")
    
    print("✅ Directorios configurados")


def create_desktop_shortcut():
    """Crear acceso directo en el escritorio (Windows)"""
    if platform.system() != "Windows":
        return
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "Medico3D.lnk")
        target = sys.executable
        wDir = str(Path(__file__).parent)
        arguments = str(Path(__file__).parent / "main.py")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.Arguments = arguments
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = target
        shortcut.save()
        
        print("✅ Acceso directo creado en el escritorio")
        
    except ImportError:
        print("⚠️  No se pudo crear acceso directo (winshell no disponible)")
    except Exception as e:
        print(f"⚠️  Error creando acceso directo: {e}")


def test_installation():
    """Probar la instalación"""
    print("\n🧪 Probando instalación...")
    
    try:
        # Importar módulos principales
        from config import config
        from transfer_functions import TransferFunctionManager
        from image_processor import ImageProcessor
        
        print("✅ Módulos principales importados correctamente")
        
        # Probar configuración
        test_value = config.get('ui.theme', 'dark')
        print(f"✅ Configuración funcionando (tema: {test_value})")
        
        # Probar transfer functions
        tf_manager = TransferFunctionManager()
        tf_names = tf_manager.get_function_names()
        print(f"✅ Transfer Functions cargadas: {len(tf_names)} funciones")
        
        # Probar procesador de imágenes
        processor = ImageProcessor()
        print("✅ Procesador de imágenes inicializado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False


def main():
    """Función principal de instalación"""
    print("🏥 Medico3D - Instalación y Configuración")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_requirements():
        print("\n❌ Error en la instalación. Verifique los errores anteriores.")
        sys.exit(1)
    
    # Verificar instalaciones críticas
    vtk_ok = check_vtk_installation()
    pyqt_ok = check_pyqt_installation()
    
    if not (vtk_ok and pyqt_ok):
        print("\n❌ Error: Dependencias críticas no instaladas correctamente")
        sys.exit(1)
    
    # Configurar directorios
    setup_directories()
    
    # Crear acceso directo (Windows)
    if platform.system() == "Windows":
        create_desktop_shortcut()
    
    # Probar instalación
    if not test_installation():
        print("\n❌ Error en las pruebas de instalación")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 ¡Instalación completada exitosamente!")
    print("\nPara ejecutar Medico3D:")
    print(f"   python {Path(__file__).parent / 'main.py'}")
    print("\nO use el acceso directo del escritorio (Windows)")
    print("=" * 50)


if __name__ == "__main__":
    main()