import ctypes

# Загружаем библиотеку ntdll.dll
ntdll = ctypes.windll.ntdll

# Структура, описывающая PEB
class PEB(ctypes.Structure):
    pass

# Определение структуры PEB
class TEB(ctypes.Structure):
    _fields_ = [
        ("Reserved1", ctypes.c_byte * 1952),
        ("ProcessEnvironmentBlock", ctypes.POINTER(PEB))
    ]

# Функция GetCurrentProcess()
GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess

# Получаем указатель на PEB
peb = PEB()
peb_address = id(peb)
print(f"PEB Address: {peb_address}")
