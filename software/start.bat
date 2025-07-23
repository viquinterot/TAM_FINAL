@echo off
echo ========================================
echo TAM - Medical Image 3D UMAP
echo ========================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no está instalado o no está en el PATH
    echo Por favor instala Python 3.8 o superior desde https://www.python.org/
    pause
    exit /b 1
)

echo Verificando entorno virtual...
if not exist "medico3d_env" (
    echo Creando entorno virtual...
    python -m venv medico3d_env
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
)

echo Activando entorno virtual...
call medico3d_env\Scripts\activate.bat

echo Verificando dependencias...
python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias...
    python setup.py
    if errorlevel 1 (
        echo ERROR: Falló la instalación de dependencias
        pause
        exit /b 1
    )
)

echo.
echo Iniciando TAM - Medical Image 3D UMAP...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo ERROR: La aplicación terminó con errores
    echo Revisa los logs en la carpeta 'logs' para más información
    pause
)

deactivate