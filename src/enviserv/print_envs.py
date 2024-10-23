import os
import subprocess
import json

conda_virtualenvs_directory="C:\\Users\\user\\.virtualenvs"

def check_conda_envs():
    try:
        # Проверяем, установлена ли Conda
        conda_check = subprocess.run(\
            ["conda", "--version"], capture_output=True, text=True)
        
        if conda_check.returncode != 0:
            print("Conda is not installed.")
            return []
        # Выполняем команду "conda env list --json" и декодируем вывод
        envs_info = subprocess.check_output(\
            ["conda", "env", "list", "--json"]).decode("utf-8")
        # Преобразуем JSON-данные в словарь
        envs_data = json.loads(envs_info)
        result = []
        
        for env in envs_data["envs"]:
            # Извлекаем имя окружения из пути
            env_name = os.path.basename(env)
            # Проверяем, активно ли текущее окружение
            is_active = "Yes" if os.environ.get(\
                "CONDA_DEFAULT_ENV") == env_name else "No"
            # Добавляем информацию об окружении в результат
            result.append({
                "Environment": env_name,
                "Manager": "Conda",
                "IsActive": is_active
            })
        
        return result
    except Exception as e:
        return []

def check_venv_env():
    # Получаем путь к текущему виртуальному окружению
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        # Извлекаем имя окружения из пути
        env_name = os.path.basename(venv_path)
        # Определяем, активно ли текущее окружение
        is_active = "Yes"
        return [{
            "Environment": env_name,
            "Manager": "venv",
            "IsActive": is_active
        }]
    return []

def get_conda_dependencies(env_name):
    try:
        # Сохраняем текущее активное окружение
        current_env = os.environ.get("CONDA_DEFAULT_ENV")
        print(f"Getting Conda Dependencies: {current_env}")
        
        # Получаем информацию о зависимостях указанного окружения
        result = subprocess.run(["conda", "list", "--name", env_name], check=True, stdout=subprocess.PIPE, text=True)
        dependencies_info = result.stdout
        
        # Разделяем зависимости и объединяем их в одну строку с разделителем
        dependencies = dependencies_info.split("\n")
        return "\r\n".join(dependencies)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return ""

def get_venv_dependencies(env_path):
    try:
        # Сохраняем текущее активное виртуальное окружение
        current_venv = os.environ.get("VIRTUAL_ENV")
        print(f"Getting Venv Dependencies: {current_venv}")
        
        # Активируем виртуальное окружение
        activate_path = os.path.join(env_path, "Scripts", "Activate")
        subprocess.run([activate_path], check=True)
        
        # Получаем информацию о зависимостях с использованием pip
        dependencies_info = subprocess.check_output(["pip", "list", "--format=freeze"]).decode("utf-8")
        
        # Восстанавливаем предыдущее активное виртуальное окружение
        if current_venv:
            activate_path_current = os.path.join(current_venv, "Scripts", "Activate")
            subprocess.run([activate_path_current], check=True)
        else:
            subprocess.run(["deactivate"], check=True)
        
        # Разделяем зависимости и объединяем их в одну строку с разделителем
        dependencies = dependencies_info.split("\n")
        return "\r\n".join(dependencies)
    except Exception as e:
        return ""

def write_dependencies_to_file(env_name, manager, dependencies):
    # Проверяем, нужно ли записывать зависимости в файл
    if FreezeDependencies:
        print(f"Writing to file Dependencies for: manager- {manager}, env- {env_name}")
        # Формируем имя файла и записываем зависимости
        file_name = f"requirements_{manager}_{env_name}.txt"
        with open(file_name, "w") as file:
            file.write(dependencies)

def check_virtualenv_envs(path):
    # Задаем путь к папке с виртуальными окружениями
    virtualenvs_path = "C:\\Users\\user\\.virtualenvs"
    # Получаем список подпапок (виртуальных окружений)
    dirs = [d for d in os.listdir(virtualenvs_path) \
            if os.path.isdir(os.path.join(virtualenvs_path, d))]

    envs = []
    for dir_name in dirs:
        env_path = os.path.join(virtualenvs_path, dir_name)
        scripts_path = os.path.join(env_path, "Scripts")
        
        # Проверяем наличие папки Scripts, что является признаком виртуального окружения
        if os.path.exists(scripts_path):
            # Определяем, активно ли текущее виртуальное окружение
            is_active = "Yes" if os.environ.get("VIRTUAL_ENV") == env_path else "No"
            envs.append({
                "Environment": dir_name,
                "Manager": "virtualenv",
                "IsActive": is_active
            })
    
    return envs

def main():
    # Инициализация списка всех окружений
    all_envs = []
    
    print("*******************************************")
    print("*******************************************")
    print("checking envs...")
    print("")
    print("*******************************************")
    
    # Получаем информацию о виртуальных окружениях и добавляем ее в список
    all_envs += check_virtualenv_envs("C:\\Users\\user\\.virtualenvs")
    # Получаем информацию о Conda-окружениях и добавляем ее в список
    all_envs += check_conda_envs()
    # Получаем информацию о текущем venv-окружении и добавляем ее в список
    all_envs += check_venv_env()
    
    # Если активно окружение 'base', но его нет в списке, добавляем его
    if os.environ.get("CONDA_DEFAULT_ENV") == "base":
        all_envs.append({
            "Environment": "base",
            "Manager": "Conda",
            "IsActive": "Yes"
        })
    
    # Выводим таблицу соответствий
    for env in all_envs:
        print(env)

    
    print("*******************************************")
    print("")
    
    print("requirements file  for active envs updating...")
    print("")

    # Обработка активных окружений и запись файлов с зависимостями
    for env in all_envs:
        if env["IsActive"] == "Yes":
            dependencies = ""
            # Получаем зависимости в привязке к менеджерам окружений
            if env["Manager"] == "Conda":
                dependencies = get_conda_dependencies(env["Environment"])
            elif env["Manager"] == "venv":
                dependencies = get_venv_dependencies(os.environ.get("VIRTUAL_ENV"))
            
            # Записываем зависимости в файл
            if dependencies:
                write_dependencies_to_file(\
                    env["Environment"], env["Manager"], dependencies)
    
    print("envs checking finished...")
    print("")

# Set global variable for FreezeDependencies
FreezeDependencies = True

# Execute the main function
main()