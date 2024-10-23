import os

def find_python_executables():
    python_executables = {}

    # Пути, где могут быть установлены Python
    search_paths = ['/usr/bin', '/usr/local/bin', '/opt/conda/bin', '/usr/lib']

    for path in search_paths:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                # Проверяем, является ли файл исполняемым и начинается ли с "python"
                if os.path.isfile(full_path) \
                    and os.access(full_path, os.X_OK) \
                    and filename.startswith('python'):
                    # Получаем версию Python из имени файла
                    version = filename.split('python')[-1]
                    # Убираем десятичные разделители и лишние символы
                    version = version.replace('.', '').replace('-', '')
                    if version.isdigit():
                        version = f"{version[:1]}.{version[1:]}"
                        python_executables[version] = full_path

    return python_executables

# Получаем исходный словарь всех установленных Python-интерпретаторов
python_executables = find_python_executables()
for version, path in python_executables.items():
    print(f"Версия Python {version}: {path}")

def remove_symbolic_links(python_executables):
    real_python_executables = {}

    # Проверяем символические ссылки в исходном словаре и создаем новый словарь с реальными путями
    for version, path in python_executables.items():
        real_path = os.path.realpath(path)
        if real_path == path:
            real_python_executables[version] = real_path

    return real_python_executables


# Исключаем символические ссылки и получаем итоговый словарь
real_python_executables = remove_symbolic_links(python_executables)

if real_python_executables:
    print("Установленные Python-интерпретаторы:")
    print(real_python_executables)
    for version, path in real_python_executables.items():
        print(f"Версия Python {version}: {path}")
else:
    print("Python-интерпретаторы не найдены.")
