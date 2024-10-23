from .mylog import MyLogger, find_logging_objects
import os

envilog = MyLogger('envilog', 30)

class CompareRequirements:
    def __init__(self,
                 llev = 30, # msg level fo custom logger
                 path_base='/work/requirements_jup.txt',
                 path_compare='/work/requirements_vscode.txt',
                 path_base_not_in_compared='/work/requirements_jup_no_vscode.txt',
                 path_compared_not_in_base='/work/requirements_vscode_no_jup.txt') -> None:
        self.llev = llev
        self.path_base = path_base
        self.path_compare = path_compare
        self.path_base_not_in_compared = path_base_not_in_compared
        self.path_compared_not_in_base = path_compared_not_in_base

    def read_requirements(self, file_path):
        if not self.check_file_existence(file_path):
            return set()  # Возвращаем пустое множество, так как файла нет
        with open(file_path, 'r') as file:
            return set(line.strip() for line in file)

    def check_file_existence(self, file_path):
        if not os.path.isfile(file_path):
            envilog.mylev(self.llev,f"File not found: {file_path}")
            return False
        return True

    def compare_requirements(self, path_base=None, path_compare=None):
        if path_base is None:
            path_base = self.path_base
        if path_compare is None:
            path_compare = self.path_compare

        if not self.check_file_existence(path_base) or not self.check_file_existence(path_compare):
            envilog.mylev(self.llev,'one of compared files NOT EXISTS... Exit')
            return set(), set()  # Возвращаем пустые множества, 
                                 # так как один из файлов отсутствует

        base_requirements_set = self.read_requirements(path_base)
        compared_requirements_set = self.read_requirements(path_compare)

        unique_base_reqirements = base_requirements_set - compared_requirements_set
        unique_compared_requirements = compared_requirements_set - base_requirements_set

        return unique_base_reqirements, unique_compared_requirements

    def write_unique_reqirements_to_files(self,
                            path_base=None,
                            path_compare=None,
                            reqs_base_not_in_campared_file_path=None,
                            reqs_compared_not_in_base_file_path=None):
        if path_base is None:
            path_base = self.path_base
        if path_compare is None:
            path_compare = self.path_compare
        if reqs_base_not_in_campared_file_path is None:
            reqs_base_not_in_campared_file_path = self.path_base_not_in_compared
        if reqs_compared_not_in_base_file_path is None:
            reqs_compared_not_in_base_file_path = self.path_compared_not_in_base

        if not self.check_file_existence(path_base) or \
           not self.check_file_existence(path_compare):
           envilog.mylev(self.llev,'one of compared files NOT EXISTS... Exit')
           return  # Выходим из метода, так как один
                   # из файлов отсутствует

        unique_base, unique_compared = \
        self.compare_requirements(path_base, path_compare)

        with open(reqs_base_not_in_campared_file_path, 'w') as file:
            for dep in sorted(unique_base):
                file.write(f"{dep}\n")

        with open(reqs_compared_not_in_base_file_path, 'w') as file:
            for dep in sorted(unique_compared):
                file.write(f"{dep}\n")

def _main(loglevel=30, base_path=None, compared_path=None):
    if base_path is not None and compared_path is not None:
        comparer = CompareRequirements(loglevel, base_path, compared_path)
    else: comparer = CompareRequirements()
    unique_base, unique_compared = comparer.compare_requirements()
    print(unique_base)
    print(unique_compared)

if __name__ == '__main__':
    _main(30)