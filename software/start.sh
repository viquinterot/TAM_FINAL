#!/bin/bash

echo "========================================"
echo "TAM - Medical Image 3D UMAP"
echo "========================================"
echo

# Verificar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no está instalado"
    echo "Por favor instala Python 3.8 o superior"
    exit 1
fi

# Verificar versión de Python
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Se requiere Python 3.8 o superior. Versión actual: $python_version"
    exit 1
fi

echo "Verificando entorno virtual..."
if [ ! -d "medico3d_env" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv medico3d_env
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudo crear el entorno virtual"
        exit 1
    fi
fi

echo "Activando entorno virtual..."
source medico3d_env/bin/activate

echo "Verificando dependencias..."
python -c "import PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalando dependencias..."
    python setup.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Falló la instalación de dependencias"
        exit 1
    fi
fi

echo
echo "Iniciando TAM - Medical Image 3D UMAP..."
echo
python main.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: La aplicación terminó con errores"
    echo "Revisa los logs en la carpeta 'logs' para más información"
    read -p "Presiona Enter para continuar..."
fi

deactivate