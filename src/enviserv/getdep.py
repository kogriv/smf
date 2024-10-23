import subprocess
from .dictan import DictAnalyzer
from .mylog import MyLogger

dep = MyLogger('dep','INFO')

da = DictAnalyzer(dep,30)

virtualenvs = {
    'Archip_Logging': 'C:\\Users\\user\\.virtualenvs\\Archip_Logging\\Scripts\\',
    'Archip_OOPushka': 'C:\\Users\\user\\.virtualenvs\\Archip_OOPushka\\Scripts\\',
    #'Buratino': 'C:\\Users\\user\\.virtualenvs\\Buratino\\Scripts\\'
}

conda_envs = {
    'fastapi': 'C:\\Users\\user\\anaconda3\\envs\\fastapi',
    'spark': 'C:\\Users\\user\\anaconda3\\envs\\spark',
    #'tf': 'C:\\Users\\user\\anaconda3\\envs\\tf',
    #'vbt': 'C:\\Users\\user\\anaconda3\\envs\\vbt'
}


def get_dependencies_pip(virtualenv_path):
    pip_path = virtualenv_path + 'pip.exe'
    command = [pip_path, 'list']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    dependencies = output.decode("utf-8")
    packages = {}
    for line in dependencies.split('\n')[2:]:
        if line:
            package_info = line.split()
            package_name = package_info[0]
            package_version = package_info[1]
            packages[package_name] = package_version
    return packages

def get_dependencies_conda(conda_env_path):
    command = ['conda', 'list']
    process = subprocess.Popen(command, cwd=conda_env_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    dependencies = output.decode("utf-8")
    packages = {}
    for line in dependencies.split('\n')[4:]:
        if line:
            package_info = line.split()
            package_name = package_info[0]
            package_version = package_info[1]
            packages[package_name] = package_version
    return packages

dependencies_dict = {}

virtualenvs_cont = {
    #'conda_python3.11': '/opt/conda/lib/python3.11/venv/scripts/common/activate',
    'venv_python3.10': '/usr/lib/python3.10/venv/scripts/common/activate'
    }

conda_envs_cont = {
    'conda_python3.11': '/opt/conda/lib/python3.11/venv/scripts/common/activate'
    #'.../lib/python3.11/venv/scripts/common/activate',
    #'venv_python3.10': '/usr/lib/python3.10/venv/scripts/common/activate'
    }

def get_dependencies_pip_activ(venv_path):
    # Формируем команду для активации виртуального окружения через source
    command = f"source {venv_path} && pip list"
    print(command)
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

#for venv_name, venv_path in virtualenvs.items():
for venv_name, venv_path in virtualenvs_cont.items():
    dependencies_dict.setdefault('pip', {}).update(
        {venv_name: 
        #get_dependencies_pip(venv_path)
        get_dependencies_pip_activ(venv_path)
        }
        )

#for venv_name, venv_path in virtualenvs.items():
for venv_name, venv_path in conda_envs_cont.items():
    dependencies_dict.setdefault('conda', {}).update(
        {venv_name: 
        #get_dependencies_pip(venv_path)
        get_dependencies_conda(venv_path)
        }
        )
"""
#for env_name, env_path in conda_envs.items():
for env_name, env_path in conda_envs_cont.items():
    dependencies_dict.setdefault('conda', {}).update(
        {env_name: get_dependencies_conda(env_path)})
"""

da.print_dict(dependencies_dict)


