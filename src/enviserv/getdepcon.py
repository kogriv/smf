import subprocess

venv_path = '/usr/lib/python3.10/venv/scripts/common'
activate_script = 'activate'

def get_dependencies_in_virtualenv(venv_path):
    # Формируем команду для активации виртуального окружения через source
    command = f"source {venv_path}/{activate_script} && pip list"

    # Выполняем команду через bash
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')
    output, error = process.communicate()
    dependencies = {}
    if output:
        output_str = output.decode("utf-8")
        lines = output_str.strip().split('\n')[2:]  # Пропускаем первые две строки
        for line in lines:
            parts = line.split()
            package_name = parts[0]
            package_version = parts[1]
            dependencies[package_name] = package_version
    return dependencies

dependencies = get_dependencies_in_virtualenv(venv_path)
print(dependencies)
