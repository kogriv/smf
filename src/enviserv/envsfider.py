import os
import subprocess
import sys
import ctypes
from ctypes import wintypes

from .mylog import MyLogger

class VenvsFinder:
    """
    Class to find venvs files in a given path
    using various shells or Python itself.
    """

    def __init__(self,
                 llev=30,
                 envilog=None,
                 verbose=False):
        """
        Initializes an VenvsFinder instance.

        Parameters:
        - llev (int): Logging message level.
        - envilog (MyLogger): Logger instance.
        - verbose (bool): Verbosity flag.
        """
        self.verbose = verbose

        if self.verbose:
            print("---------------------------------------------------")
            print("Attempt to initialise VenvsFinder's instance")
        
         #['DEBUG','INFO','WARNING','ERROR','CRITICAL']
        if llev not in [10,20,30,40,50]:
            llev = 30
        self.llev = llev

        if envilog is None:
            envilog = MyLogger('envilog', 'INFO')
        self.envilog = envilog
        
        if self.verbose:
            envilog.mylev(llev,
                "VenvsFinder's instance initialized")


    def get_actual_folder_name(self, path):
        """
        Takes a path and returns the actual name
        of the final folder in the path
        with case sensitivity.

        Args:
            path (str): The path to analyze.

        Returns:
            str: The actual name of the final folder.
            False: If the path doesn't exist.
        """
        # Windows platform specific implementation
        if sys.platform == 'win32':
            # Definition of the WIN32_FIND_DATA structure
            class WIN32_FIND_DATA(ctypes.Structure):
                _fields_ = [
                    ("dwFileAttributes", wintypes.DWORD),
                    ("ftCreationTime", wintypes.FILETIME),
                    ("ftLastAccessTime", wintypes.FILETIME),
                    ("ftLastWriteTime", wintypes.FILETIME),
                    ("nFileSizeHigh", wintypes.DWORD),
                    ("nFileSizeLow", wintypes.DWORD),
                    ("dwReserved0", wintypes.DWORD),
                    ("dwReserved1", wintypes.DWORD),
                    ("cFileName", wintypes.WCHAR * 260),
                    ("cAlternateFileName", wintypes.WCHAR * 14)
                ]
            # Initializing the WIN32_FIND_DATA structure
            find_data = WIN32_FIND_DATA()
            # Loading the kernel32.dll library
            kernel32 = ctypes.WinDLL('kernel32',
                                     use_last_error=True)
            # Getting the folder descriptor
            hFind = kernel32.FindFirstFileW(path,
                                ctypes.byref(find_data))
            # Checking if the folder is successfully opened
            if hFind != -1:
                # Closing the descriptor
                kernel32.FindClose(hFind)
                # Returning the actual folder name
                return find_data.cFileName
            else:
                return False

        # Non-Windows platform specific implementation
        else:
            # If the platform is not Windows,
            # returning the name of the last folder
            path = path.replace('\\', '/').\
                replace(':', '').strip('/')
            if os.path.exists(os.path.normpath(path)):
                path_components = path.split('/')
                if path_components:
                    return path_components[-1]
            else:
                return False

    def get_actual_path(self, folder_path):
        """
        Get the actual path of a folder
        considering case sensitivity.

        Args:
            folder_path (str): The folder path to analyze.

        Returns:
            str: The actual folder path.
        """
        if sys.platform == 'win32':
            if '\\' in folder_path:
                if folder_path.endswith('\\'):
                    folder_path = folder_path[:-1]

                path_components = folder_path.split('\\')
                actual_path = ''
                for component in path_components:
                    if component:
                        if len(component) == 2 \
                            and component[1] == ':':
                            actual_path += component
                        else:
                            actual_path = '\\'.join(
                                [actual_path,
                                 self.get_actual_folder_name(
                                actual_path + '\\' + component)]
                                )
                            if actual_path is None:
                                return None
                return actual_path
            else:
                return folder_path
        else:
            folder_path = folder_path.\
                replace(":",'').replace("\\","/")
            return folder_path

    def universalize_path(self,
                          input_path,
                          standard_path_win=None,
                          standard_path_linux=None):
        """
        Convert a path into a format
        that is understandable by
        both Windows and Linux.

        Args:
            input_path (str): The input path to be converted.
            standard_path_win (list, optional): List of
                        standard Windows paths for search.
                        Defaults to None.
            standard_path_linux (list, optional): List of
                        standard Linux paths for search.
                        Defaults to None.

        Returns:
            str: The corrected Windows path.
            str: The corresponding bash path.
        """
        if standard_path_win is None:
            standard_path_win = []
            if sys.platform == 'win32':
                standard_path_win.\
                    append(os.environ['USERPROFILE'])

        if standard_path_linux is None:
            standard_path_linux = []
            if sys.platform.startswith('linux'):
                standard_path_linux.\
                    append(os.environ['HOME'])
            if sys.platform == 'win32':
                home_path = os.environ['USERPROFILE']
                home_path = '/' + home_path.\
                    replace('\\', '/').replace(':', '', 1)
                standard_path_linux.append(home_path)

        if '\\' in input_path:
            if not os.path.exists(os.path.
                            normpath(input_path)):
                return False, False

            try:
                corrected_path = self.\
                    get_actual_path(input_path)
            except subprocess.CalledProcessError:
                return False, False

            bash_path = '/' + corrected_path.\
                replace('\\', '/').replace(':', '', 1)
            
            return corrected_path, bash_path
        else:
            if len(input_path) > 1 \
                and input_path[-1] == '/':
                input_path = input_path[:-1]

            if len(input_path) > 2 \
                and input_path[0] == '/' \
                and input_path[2] == '/':
                win_path = input_path[1].\
                    upper() + ':' + input_path[2:]
                win_path = win_path.replace('/', '\\')
            
            elif len(input_path) > 1 \
                and input_path[0] != '/' \
                and input_path[1] == '/':
                win_path = input_path[0] + \
                    ':' + input_path[1:]
                input_path = '/' + input_path
                win_path = win_path.replace('/', '\\')

            else:
                win_path = input_path.replace('/', '\\')

            if ':' not in win_path:
                path_found = False
                for parent_path in standard_path_win:
                    if path_found:
                        break
                    for root, dirs, files \
                        in os.walk(parent_path):
                        if win_path in root:
                            win_path = self.\
                                get_actual_path(root)
                            path_found = True
                            break

            if input_path[0] != '/':
                bash_path = '/' + input_path
            else: 
                bash_path = input_path

            return win_path, bash_path

    def check_shells(self,
                     shells=[
                    'powershell',
                    'bash', 'zsh', 'cmd'
                    ],
                    verbose=False):
        """
        Check the availability of various shells.

        Args:
            shells (list or str, optional): List of
                    shells to check. Defaults to
                    ['powershell', 'bash', 'zsh', 'cmd'].
            verbose (bool, optional): Verbosity flag.
                    Defaults to False.

        Returns:
            dict: A dictionary containing shell names
                as keys and availability status as values.
        """
        results = {}

        if isinstance(shells, str):
            shells = [shells]

        for shell in shells:
            if verbose:
                self.envilog.mylev(self.llev,f'Checking: {shell}')
            try:
                if shell == 'powershell':
                    subprocess.check_call([
                        'powershell', '-NonInteractive',
                        '-NoProfile', '-Command',
                        'Get-ChildItem'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    results[shell] = True
                elif shell == 'cmd':
                    subprocess.check_call([
                        'cmd', '/c',
                        'echo "Hello from cmd"'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    results[shell] = True
                else:
                    subprocess.check_call([
                            shell, '-c',
                            'echo "Hello from {}"'.\
                            format(shell)
                            ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    results[shell] = True
            except FileNotFoundError:
                results[shell] = False
            except subprocess.CalledProcessError:
                results[shell] = False
            except Exception:
                results[shell] = False

        if len(results) == 1:
            return results[shells[0]]
        else:
            return results

    def search_venvs_path(self,
                     path_find=None,
                     shell='python',
                     verbose=False):
        """
        Search activation files in a given path using
        various shells or Python itself.

        Args:
            path_find (str, optional): The path to
                        search in. Defaults to None.
            shell (str, optional): The shell to use
                    for searching. Defaults to 'python'.

        Returns:
            list: A list of file paths found.
        """
        results = []
        pyfind = False

        if path_find is None:
            home_directory = os.path.expanduser("~")
        else:
            home_directory = path_find

        win_path, bash_path = self.\
            universalize_path(home_directory)

        if win_path and bash_path:
            if shell != 'python':
                if self.check_shells(shell):
                    if verbose:
                        self.envilog.mylev(self.llev,
                            f"Shell {shell} is available")
                
                else:
                    pyfind = True
                    if verbose:
                        self.envilog.mylev(self.llev,
                        f"Shell {shell} is not available, "+\
                        f"finding using Python os.walk()"
                        )
                
                if home_directory == "/" and shell == 'bash':
                    pyfind = True
                    if verbose:
                        self.envilog.mylev(self.llev,
                            f"Shell Bash and Path to find equal to '/', "+\
                        f"finding using Python os.walk()"
                        )
            else:
                pyfind = True
        
            if pyfind:
                if verbose:
                    self.envilog.mylev(self.llev,
                        "Attempting to find files using Python")
                path_to_find = win_path
                if sys.platform.startswith('linux'):
                    path_to_find = bash_path
                for root, dirs, files in os.walk(path_to_find):
                    for file in files:
                        if file.endswith("activate"):
                            results.append(os.path.join(root, file))
            else:
                if shell == 'bash':
                    if verbose:
                        self.envilog.mylev(self.llev,
                            "Attempting to find files using bash")
                    try:
                        find_command = f'find {bash_path} '+\
                            f'-path "/proc" -prune -o '+\
                            f'-name "*activate" -type f -print'
                        output = subprocess.check_output(
                                find_command, shell=True,
                                text=True, stderr=subprocess.PIPE
                        )
                        results.extend(output.strip().split('\n'))
                    except subprocess.CalledProcessError as e:
                        self.envilog.mylev(self.llev,
                                    "Error occurred:", e)
                elif shell == 'powershell':
                    if verbose:
                        self.envilog.mylev(self.llev,
                            f"Attempting to find files "+\
                            f"using PowerShell (Windows)")
                    try:
                        find_command = f'Get-ChildItem -Path '+\
                            f'"{win_path}" -Recurse -File '+\
                            f'-Filter "*activate" | ' + \
                            f'Select-Object -ExpandProperty FullName'
                        output = subprocess.check_output([
                            'powershell', '-NonInteractive',
                            '-NoProfile', '-Command', find_command],
                            text=True)
                        results.extend(output.strip().split('\n'))
                    except subprocess.CalledProcessError as e:
                        self.envilog.mylev(self.llev,
                                    "Error occurred:", e)

        else:
            self.envilog.mylev(self.llev,"Path does not exist")
            return []
        
        return results

if __name__ == '__main__':
    # Example usage:
    #ptf_bash = "C:\\Users\\user\\documents"
    ptf_bash = "/"
    #ptf_bash = None
    """
    print("----------------------------------------------------")
    print("Finding using bash /// path to find:")
    print(ptf_bash)
    result_list = VenvsFinder().search_venvs_path(
        path_find=ptf_bash, shell='bash')
    print(result_list)

    print("----------------------------------------------------")
    print("Finding using Python /// path to find:")
    print(ptf_bash)
    result_list = VenvsFinder().search_venvs_path(
        path_find=ptf_bash)
    print(result_list)
    """
    print("----------------------------------------------------")
    print("Finding using Powershell /// path to find:")
    print(ptf_bash)
    result_list = VenvsFinder().search_venvs_path(
        path_find=ptf_bash, shell='powershell',verbose=True)
    print(result_list)
