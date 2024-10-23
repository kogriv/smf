import sys
import os
import subprocess

from .environcheck import EnvironCheck
from .envsfider import VenvsFinder
from .mylog import MyLogger
from .dictan import DictAnalyzer

envilog = MyLogger(name='ech',create_level='INFO',enable_logging=True)

da = DictAnalyzer(envilog,30)
venvsfinder = VenvsFinder(llev=30,envilog=envilog,verbose=False)
ech = EnvironCheck(llev=30,envilog=envilog,verbose=False)



envs_path_list_win = []
envs_path_list_linux = []

result_list = []
# Определение разделителя пути в зависимости от операционной системы
slash = '\\' if sys.platform == 'win32' else '/'
grouped_paths = {}
# Группировка путей для virtualenvs, conda, venv
grouped_paths['venv'] = {}
grouped_paths['virtualenvs'] = {}
grouped_paths['conda'] = {}

environments = {}

def get_venvs_paths(envs_path_list_win=None,
                    envs_path_list_linux=None,
                    verbose=False):
    if sys.platform == 'win32':
        slash = '\\'
        home_path = os.environ['USERPROFILE']
        if envs_path_list_win is None \
            or envs_path_list_win==[]:
            envs_path_list_win = [
                home_path + '\\.virtualenvs',
                home_path + '\\anaconda3\\envs',
                home_path + '\\Documents\\pro',
                'C:\\Projects'
            ]
        
        for ptf in envs_path_list_win:
            if verbose:
                print("Searching in:",ptf)
            res = venvsfinder.search_venvs_path(
                path_find=ptf, shell='powershell',verbose=False)
            if res != ['']:
                result_list.extend(res)
        
            for path in result_list:
                parts = path.split(slash)
                if '.virtualenvs' in parts:
                    env_type = 'virtualenvs'
                    env_name = parts[parts.index('.virtualenvs') + 1]
                elif 'anaconda3' in parts:
                    env_type = 'conda'
                    if len(parts)>parts.index('anaconda3') + 2:
                        env_name = parts[parts.index('anaconda3') + 2]
                    else: env_name = 'undef_short_path'
                elif  'conda' in parts:
                    env_type = 'conda'
                    if len(parts)>parts.index('conda') + 2:
                        env_name = parts[parts.index('conda') + 2]
                    else: env_name = 'undef_short_path'
                else:
                    env_type = 'venv'
                    env_name = parts[-3]
                grouped_paths.setdefault(env_type, {}).update({env_name: path})

    if sys.platform.startswith('linux'):
        slash = '/'
        home_path = os.environ['HOME']
        if envs_path_list_linux is None \
            or envs_path_list_linux==[]:
            if ech.container_id:
                if verbose:
                    print("Progremm runing in docker container:",
                          ech.container_id)
                    print("So root path '/' will be used for searching")
                envs_path_list_linux = ["/"]
            else:
                envs_path_list_linux = [
                os.path.join(home_path, '.virtualenvs'),
                os.path.join(home_path, 'anaconda3', 'envs'),
                os.path.join(home_path, 'Documents', 'pro'),
                '/Projects'
            ]

        for ptf in envs_path_list_linux:
            res = venvsfinder.search_venvs_path(
                path_find=ptf, shell='bash',verbose=True)
            if res != ['']:
                result_list.extend(res)
                
        # Группировка путей для базовых окружений
        for path in result_list:
            parts = path.split('/')
            if 'common' in parts:
                if 'conda' in path:
                    grouped_paths['conda']['base'] = path
                elif 'venv' in parts:
                    grouped_paths['venv']['base'] = path

            elif 'bin' in parts and 'activate' in parts \
                    and 'conda' not in parts \
                    and '.virtualenvs' not in parts:
                env_name = parts[-3]
                grouped_paths['venv'][env_name]=path
            elif 'bin' in parts and 'activate' in parts \
                    and 'conda' not in parts \
                    and '.virtualenvs' in parts:
                env_name = parts[-3]
                grouped_paths['virtualenv'][env_name]=path

def get_conda_envs(verbose=False):
    conda_envs = {}
    try:
        # Проверяем, установлена ли Conda
        conda_check = subprocess.run(\
            ["conda", "--version"], capture_output=True, text=True)
        
        if conda_check.returncode != 0:
            if verbose:
                print("Conda is not installed.")
            grouped_paths['conda'] = False
            return {}
        else:
            if verbose:
                print("Conda installed.")
        # Выполняем команду "conda env list" и декодируем вывод
        process = subprocess.Popen(
                            ["conda", "env", "list"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                            )
        output, error = process.communicate()
        conda_list = output.decode("utf-8")
        #if conda_list.split('\n') > 2:
        for line in conda_list.split('\n')[2:]:
            if line:
                venv_info = line.split()
                if len(venv_info)>0:
                    venv_name = venv_info[0]
                    if len(venv_info)>1:
                        venv_path = venv_info[1]
                    else: venv_path = ''
                    grouped_paths['conda'][venv_name] = venv_path
                    conda_envs[venv_name] = venv_path
        return conda_envs
    except Exception as e:
        if verbose:
            print("Error during conda execution:",e)
        return False

def get_dependencies_conda(conda_env_name, verbose=False):
    packages = {} # dict for deps
    command = ["conda", "list", "-n", conda_env_name]
    if verbose:
        print("Attempt to run command:",command)
    process = subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            )
    output, error = process.communicate()
    dependencies = output.decode("utf-8")
    if not '# packages in environment at' in dependencies:
        if verbose:
            print("Invalid output of command: str (# packages in environment at) expected")
        return False
    if '\n' not in dependencies:
        if verbose:
            print("Invalid output of command: there is no lines divider = \\n")
        return False
    lines = dependencies.split('\n')
    if len(lines) > 3 and not (len(lines)==4 and lines[3]==''):
        lines = lines[3:]
    else:
        if verbose:
            print(f"There is no any dependencie in this venv = {conda_env_name}")
        return {}
    for line in lines:
        if line and line != '':
            package_info = line.split()
            if len(package_info)>0:
                package_name = package_info[0]
                if len(package_info)>1:
                    package_version = package_info[1]
                else: package_version = None
                packages[package_name] = package_version
            else:
                if verbose:
                    print(f"Invalid line = {str(line)[:10]} in command output..")
            
    return packages

def check_venv_founded(package_manager, env_name, environments,verbose=False):
    path = environments.get(package_manager, {}).get(env_name)
    if path is None:
        if verbose:
            print(f"Path for {package_manager} : {env_name} NOT found")
        return False
    elif path == '':
        if verbose:
            print(f"Path for {package_manager} : {env_name} is empty")
        return False
    return path

def get_dependencies(package_manager, env_name, environments,verbose=False):
    if verbose:
        print("Start geting deps...")
    dependencies = {}
    venv_path = check_venv_founded(package_manager, env_name, environments)
    shell_path = None
    # Формируем команду и путь к оболочке ОС для активации виртуального окружения
    if sys.platform.startswith('linux'):
        shell_path = os.environ.get('SHELL', '/bin/bash')
        command = f"source {venv_path} && pip list"
        if verbose:
            print('linux shell path:',shell_path)
    
    if sys.platform == 'win32':
        shell_path = os.environ.get('COMSPEC', 'cmd.exe')
        if verbose:
            print('win shell path:',shell_path)
        command = f"{venv_path} && pip list"
    if verbose:
        print("venv_path = check_venv_founded(package_manager, env_name, environments) :",venv_path)
    if venv_path:
        if package_manager == 'conda':
            dependencies = get_dependencies_conda(env_name,verbose)
            if verbose:
                print("dependencies = get_dependencies_conda(env_name) :",str(dependencies)[:50]+"....")
            return dependencies
        
        elif package_manager in ['virtualenvs','venv']:
            if verbose:
                print(f'attempt to use {package_manager} to run command:',command)
            # Выполняем команду через оболочку
            process = subprocess.Popen(command
                            , stdout=subprocess.PIPE
                            , stderr=subprocess.PIPE
                            , shell=True
                            , executable=shell_path
                            )
            output, error = process.communicate()
            
            dependencies = {}
            if output:
                output_str = output.decode("utf-8")
                lines = output_str.strip().split('\n')
                if len(lines) == 3 and lines[2] == '':
                    if verbose:
                        print(f"There is no any dependencie in this venv = {env_name}")
                    return {}
                if len(lines)>2:
                    lines=lines[2:]
                else:
                    if verbose:
                        print("Invalid output: len(lines) less than 3")
                    return False
                for line in lines:
                    parts = line.split()
                    package_name = parts[0]
                    if len(parts)>1:
                        package_version = parts[1]
                    else:
                        package_version = None
                    dependencies[package_name] = package_version
            return dependencies
    else:
        return False

def get_all_dependencies(environments,verbose=True):
    if verbose:
        print("---------------------------------------------------------")
        print("-Functon get_all_dependencies() started------------------")
        print("-Geting all deps. It can be long for a while-------------")
    for packman in [
        'virtualenvs',
        'conda',
        'venv'
        ]:
        if verbose:
            print("---------------------------------------------------------")
        if packman in environments:
            if verbose:
                print(f"------{packman} exists in venvs dict--------------------")
            for env_name in environments[packman]:
                if check_venv_founded(packman,env_name,environments):
                    if verbose:
                        print(f"venv dict contain packman = {packman} | env_name = {env_name}")
                    dependencies = \
                    get_dependencies(packman,env_name,environments)
                    if dependencies:
                        if verbose:
                            print("----deps geted. dict will be returned--------")
                    else:
                        if verbose:
                            print("----deps NOT geted. False will be returned--------")
                        continue
                else:
                    if verbose:
                        print(f"venv dict NOT contain packman = {packman} | env_name = {env_name}")
                    continue
        else:
            print(f"---venv dict NOT contain packman =  {packman}")
            continue
    if verbose:
        print('====FINDING VIRTUAL ENVS DEPENDENCIES FINISHED=====')


def get_some_dependencies(prompt_nested_lists=None,venvs=None,verbose=False):
    data_validated = True
    deps = {} # dict of dependencies
    if venvs is None \
        or venvs == {}:
        venvs = environments
    if not isinstance(venvs, dict):
        if verbose:
            print(
            f"Invalid input arg. Please ensure venvs param "+\
            f"is a dictionary."
                )
        data_validated = False
        return False
    if prompt_nested_lists is None \
    or prompt_nested_lists == [[]]:
        prompt_nested_lists = [
            ['conda','myenv'],
            ['conda','base'],
            ['venv','myenv'],
            ['venv','arch'],
            ['virtualenvs','Buratino']
            ]
    # проверка того, что prompt_nested_lists
    # является сруктурой типа
    # [[packman: str,env_name: str]]
    for sublist in prompt_nested_lists:
        if not (isinstance(sublist, list) and len(sublist) == 2):
            if verbose:
                print(
                f"Invalid input arg. Please ensure your input prompt_nested_lists"+\
                f"is a list of lists, with each sublist "+\
                f"containing 2 elements."
                )
            data_validated = False
            return False
        packman, env_name = sublist
        if not isinstance(packman, str) \
                and isinstance(env_name, str):
            if verbose:
                print(
                f"Invalid input args. Please ensure each sublist of prompt_nested_lists"+\
                f"contains a string name of packman, and a string name of venv."
                )
            data_validated = False
            return False

    if data_validated:
        for pr in prompt_nested_lists:
            packman = pr[0]
            venv_name = pr[1]
            if verbose:
                print(f'-----geting dependencies for packman = {packman} | '+\
                    f' venv_name = {venv_name} -----')
            deps_geted = deps.get(packman, {}).get(venv_name)
            if deps_geted is None:
                if verbose:
                    print("--dependencies not geted yet-----")
                if check_venv_founded(packman,venv_name,venvs,verbose):
                    if verbose:
                        print("venvs dict contains such data")
                        print("run get_dependencies()")
                    deps_geted = get_dependencies(packman,venv_name,venvs)
                    if deps_geted or deps_geted == {}:
                        if verbose:
                            print("---|OK|OK|OK|---deps geted! updating deps dict------")
                        deps.setdefault(packman, {})[venv_name] = deps_geted
                    else:
                        if verbose:
                            print("---deps NOT geted------")
                else:
                    if verbose:
                        print("---venvs dict is NOT contains such data")
            else:
                if verbose:
                    print("deps for this packman and venv allready geted")
    return deps
            
if __name__ == '__main__':
    print("Geting venvs paths...")
    # updating grouped_paths dict
    get_venvs_paths(envs_path_list_win=None,envs_path_list_linux=None)

    print("Geting conda env list...")
    # updating grouped_paths dict
    get_conda_envs()

    print("Checked activate scripts paths:")
    da.print_dict(grouped_paths)

    print("Geting dependencies for some venvs...")
    deps = get_some_dependencies(prompt_nested_lists=None,
                                 venvs=grouped_paths)
    print("---------------------------------------")
    da.print_dict(deps)
    #get_all_dependencies()


"""
# sample of environmets dict resulted as grouped_path
environments = \
{
'virtualenvs':
    {
        'Archip_Logging': 'C:\\Users\\user\\.virtualenvs\\Archip_Logging\\Scripts\\activate',
        'Archip_OOPushka': 'C:\\Users\\user\\.virtualenvs\\Archip_OOPushka\\Scripts\\activate',
        'Buratino': 'C:\\Users\\user\\.virtualenvs\\Buratino\\Scripts\\activate',
        'chicago_spark': 'C:\\Users\\user\\.virtualenvs\\chicago_spark\\Scripts\\activate',
        'DemoPyCharmProductive': 'C:\\Users\\user\\.virtualenvs\\DemoPyCharmProductive\\Scripts\\activate',
        'PapaPy': 'C:\\Users\\user\\.virtualenvs\\PapaPy\\Scripts\\activate',
        'PyCharmLearningProject': 'C:\\Users\\user\\.virtualenvs\\PyCharmLearningProject\\Scripts\\activate',
        'pycharm_Luchanos': 'C:\\Users\\user\\.virtualenvs\\pycharm_Luchanos\\Scripts\\activate',
        'pyhub': 'C:\\Users\\user\\.virtualenvs\\pyhub\\Scripts\\activate'
    },
'conda': 
    {
        'fastapi': 'C:\\Users\\user\\anaconda3\\envs\\fastapi\\Lib\\venv\\scripts\\common\\activate',
        'spark': 'C:\\Users\\user\\anaconda3\\envs\\spark\\Lib\\venv\\scripts\\common\\activate',
        'tf': 'C:\\Users\\user\\anaconda3\\envs\\tf\\Lib\\venv\\scripts\\common\\activate',
        'vbt': 'C:\\Users\\user\\anaconda3\\envs\\vbt\\Lib\\venv\\scripts\\common\\activate'},
'venv': 
    {
        '.test_venv': 'C:\\Users\\user\\Documents\\Pro\\Archip\\arch\\.test_venv\\Scripts\\activate',
        'myenv_prompt': 'C:\\Users\\user\\Documents\\Pro\\Archip\\arch\\myenv_prompt\\Scripts\\activate',
        'arch': 'C:\\Users\\user\\Documents\\Pro\\Archip\\arch\\Scripts\\activate'
    }
}
"""

