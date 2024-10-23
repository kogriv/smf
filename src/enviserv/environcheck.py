import os
import platform

from .mylog import MyLogger

class EnvironCheck:
    """
    A class to check and manage environment variables.
    """
    def __init__(self,
                 llev=30,
                 envilog=None,
                 verbose=False):
        """
        Initializes an EnvironCheck instance.

        Parameters:
        - llev (int): Log level.
        - envilog (MyLogger): Logger instance.
        - verbose (bool): Verbosity flag.
        """
        if verbose:
            print("---------------------------------------------------")
            print("Attempt to initialise EnvironCheck's instance")
        
        #['DEBUG','INFO','WARNING','ERROR','CRITICAL']
        if llev not in [10,20,20,40,50]:
            llev = 30

        self.llev = llev

        if envilog is None:
            envilog = MyLogger('envilog', 'INFO')
        self.envilog = envilog
        
        if verbose:
            self.envilog.mylev(self.llev,
                "---------------------------------------------------")
            self.envilog.mylev(self.llev,
                "Attempt to execute check_in_container()...")
        self.container_id = self.check_in_container()
        if verbose:
            self.envilog.mylev(self.llev,
                "EnvironCheck's instance initialized")

    def print_environment_variables(self,
                    selected_vars=None,
                    save_to_file=False):
        """
        Prints environment variables.

        Parameters:
        - selected_vars (list): List of variables to print.
        - save_to_file (bool): Flag to save variables to a file.
        """
        if selected_vars is not None:
            full_list = True if "FULL_ENVIRON_LIST" \
                in selected_vars else False
        else:
            full_list = False
        # Get the dictionary of environment variables
        env_vars = os.environ

        # Output platform information
        self.envilog.mylev(self.llev,
            f"Platform Information: "
            f"{platform.system()} {platform.release()}")

        container_id = self.container_id

        # Select variables for output
        if selected_vars is None:
            selected_vars = ["_", "HOME", "SPARK_MASTER"]
        elif full_list:
            selected_vars = env_vars

        # Output information for each environment variable in the list
        self.envilog.mylev(self.llev,
                "\nEnvironment Variables Information:")
        for key in selected_vars:
            if key in env_vars:
                self.envilog.mylev(self.llev,
                        f"{key}: {env_vars[key]}")
            else:
                self.envilog.mylev(self.llev,
                            f"{key}: Not found")

        # Save to file if specified
        if save_to_file:
            self.save_env_to_file(env_vars,
                        selected_vars,
                        container_id
                        )

    def check_in_container(self,verbose=False):
        """
        Checks if the program is running inside a container.

        Parameters:
        - verbose (bool): Verbosity flag.
        
        Returns:
        - container_id (str): Container ID if inside a container, False otherwise.
        """
        # Check for the presence of /proc/1/cgroup
        cgroup_path = '/proc/1/cgroup'
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as cgroup_file:
                for line in cgroup_file:
                    if "docker" in line:
                        # Find the line containing "docker"
                        # and extract the container ID
                        parts = line.split(":")
                        if len(parts) >= 3:
                            container_id_full = \
                                parts[2].strip()
                            container_id = \
                                container_id_full.split("/")[-1]
                            if verbose:
                                self.envilog.mylev(self.llev,
                                f"Program is running inside a container: \
                                {container_id}"
                                )
                            return container_id

        # Check for the presence of /var/run/docker.sock
        docker_sock_path = '/var/run/docker.sock'
        if os.path.exists(docker_sock_path):
            return True

        if verbose:
            self.envilog.mylev(self.llev,
                "Program is NOT running inside a container.")
        return False

    def get_save_folder_path(self, user_path=None):
        """
        Retrieves the path to save environment variables.

        Parameters:
        - user_path (str): User-defined path.

        Returns:
        - save_folder_path (str): Path to save folder.
        """
        if user_path:
            return user_path

        container_id = self.container_id

        if container_id:
            pythonpath = os.environ.get('PYTHONPATH')
            if pythonpath:
                return os.path.join(pythonpath, "environ")
            else:
                home_path = os.path.expanduser("/")
                return os.path.join(home_path, "work", "environ")

        # Find the project root by looking for the .git folder
        current_path = os.path.dirname(os.path.abspath(__file__))
        while current_path != os.path.dirname(current_path):
            if os.path.exists(os.path.join(current_path, ".git")):
                return os.path.join(current_path, "environ")
            current_path = os.path.dirname(current_path)

        # If the .git folder is not found, use the current directory
        return os.path.join(os.getcwd(), "environ")

    def save_env_to_file(self,
                         verbose=False,
                         selected_vars=None,
                         user_path=None
                         ):
        """
        Saves environment variables to a file.

        Parameters:
        - verbose (bool): Verbosity flag.
        - selected_vars (list): List of variables to save.
        - user_path (str): User-defined path.
        """
        env_vars = os.environ
        container_id = self.container_id

        if selected_vars is None:
            selected_vars = ["_", "HOME", "PYTHONPATH", "SPARK_MASTER"]

        # Formulate lines to save
        lines_to_save = [
            f"{key}={value}" for key, value 
            in env_vars.items() 
            if key in selected_vars
        ]

        # Add container information if available
        if container_id:
            lines_to_save.append(f"CONTAINER_ID={container_id}")
        else:
            lines_to_save.append("CONTAINER_ID=NO_CONTAINER")

        # Add PYTHONPATH information if available, otherwise set to "NOT_SET"
        pythonpath = os.environ.get('PYTHONPATH', 'NOT_SET')
        lines_to_save.append(f"PYTHONPATH={pythonpath}")

        # Add SPARK_MASTER information if available, otherwise set to "NOT_SET"
        spark_master = os.environ.get('SPARK_MASTER', 'NOT_SET')
        lines_to_save.append(f"SPARK_MASTER={spark_master}")

        # Add platform information
        lines_to_save.append(f"PLATFORM={platform.system()}")
        lines_to_save.append(f"PLATFORM_RELEASE={platform.release()}")

        if verbose:
            self.envilog.mylev(self.llev,
                            "Data to be saved:")
            for line in lines_to_save:
                self.envilog.mylev(self.llev, line)

        # Get the path to the save folder
        save_folder_path = self.get_save_folder_path(user_path)

        # Create the folder if it does not exist
        os.makedirs(save_folder_path, exist_ok=True)

        # Formulate the path to save the file
        save_path = os.path.join(save_folder_path,
                                 ".environ_var_list")

        # Save to file
        with open(save_path, 'w') as save_file:
            save_file.write("\n".join(lines_to_save))

        if verbose:
            self.envilog.mylev(self.llev,
                f"Environment variables saved to file:")
            self.envilog.mylev(self.llev,save_path)


def load_env(file_path=".env"):
    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.strip() and not line.startswith("#"):
                    key, value = map(str.strip, line.split("=", 1))
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"File {file_path} not found. No environment variables loaded.")

if __name__ == '__main__':
    envilog = MyLogger('environcker','INFO')
    envichecker = EnvironCheck(None,None,True)
    envilog.mylev(30,"---------------------------------------------------")
    container_id = envichecker.check_in_container(True)
    envilog.mylev(30,"---------------------------------------------------")
    envilog.mylev(30,"print_environment_variables():")
    envichecker.print_environment_variables()
    """
    envilog.mylev(30,"---------------------------------------------------")
    envilog.mylev(30,"print_environment_variables(['FULL_ENVIRON_LIST']):")
    envichecker.print_environment_variables(['FULL_ENVIRON_LIST'])
    """
    envilog.mylev(30,"---------------------------------------------------")
    envilog.mylev(30,"Get path to save environment vars:")
    pth = envichecker.get_save_folder_path()
    envilog.mylev(30,pth)
    envilog.mylev(30,"---------------------------------------------------")
    envilog.mylev(30,"Attempt to save environment vars")
    envilog.mylev(30,"save_env_to_file(Verbose=True)")
    envichecker.save_env_to_file(True)
