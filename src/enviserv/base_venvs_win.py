import subprocess
import re

from .dictan import DictAnalyzer
from .mylog import MyLogger

envilog = MyLogger('base_venvs_win','INFO')
da = DictAnalyzer(logger=envilog,llev=30)

errors = {}

# Функция для выполнения команды в оболочке и получения результатов
def run_command(command):
    process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    output, error = process.communicate()
    output = output.decode("utf-8")
    if error:
        errors[str(command)] = error
    return output

# Получаем список путей к интерпретаторам Python
command = ["where", "python"]
python_paths = run_command(command).split('\n')

# Фильтруем пути, исключая те, что содержат 'WindowsApps'
python_paths = [path.strip() for path in python_paths if "WindowsApps" not in path]
#print(python_paths)
# Создаем пустой словарь для хранения информации о зависимостях
venvs_base = {}

# Регулярное выражение для извлечения версии Python из пути
version_pattern = re.compile(r"Python(\d+)(\d+)\\python.exe")

for python_path in python_paths:
    print("Получаем зависимости для питона на пути:")
    print(python_path)
    # Извлекаем версию Python из пути
    version_match = version_pattern.search(python_path)
    #print("version_match: ",version_match)
    if version_match:
        python_version = f"Python{version_match.group(1)[0]}.{version_match.group(1)[1:]}{version_match.group(2)}"
        #python_version = f"Python{version_match.group(1)}.{version_match.group(2)}"
    else:
        continue

    # Получаем список зависимостей для данного интерпретатора Python
    try:
        #command = f'"{python_path}" -m pip list --format=freeze'
        command = [python_path, '-m', 'pip', 'list', '--format=freeze']
        #print(f"Attempting run command={command}")
        output = run_command(command)
        #print(f"command executed")
        #print(output)
        dependencies_list = output.split('\n')
        dependencies = {item.split('==')[0]: item.split('==')[1] for item in dependencies_list if '==' in item}
    except Exception as e:
        errors['deps_gettig_for'+str(command)]=e
        #print(f"Ошибка при получении списка зависимостей для {python_path}: {e}")
        continue

    # Обновляем словарь с информацией о зависимостях
    venvs_base[python_version] = dependencies

# Выводим полученный словарь
da.print_dict(venvs_base)
