import os
import sys

def get_host_installed_python_versions():
    python_versions = {}

    # Получаем путь к исполняемому файлу Python
    python_executable_path = sys.executable
    print("python_executable_path: ",python_executable_path)

    # Получаем каталог, содержащий исполняемый файл Python
    python_installation_path = os.path.dirname(os.path.dirname(python_executable_path))
    print("python_installation_path: ",python_installation_path)

    # Получаем список всех папок в каталоге установки Python
    python_folders = os.listdir(python_installation_path)

    for folder in python_folders:
        # Проверяем, является ли папка версией Python
        if folder.startswith('Python'):
            python_version = folder.replace('Python', '')
            if len(python_version)>1:
                python_version = python_version[0]+'.'+python_version[1:]
            python_version_path = os.path.join(python_installation_path, folder)
            python_executable = os.path.join(python_version_path, 'python.exe')

            if os.path.isfile(python_executable):
                python_versions[python_version] = python_executable

    return python_versions

# Пример использования:
installed_python_versions = get_host_installed_python_versions()
print("Установленные версии Python:")
print(installed_python_versions)
