import ctypes
import time

# Определение констант
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
ERROR_ACCESS_DENIED = 5

# Получение функций из kernel32.dll
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.Psapi

# Загрузка библиотеки user32.dll
user32 = ctypes.windll.user32

# Функция для проверки состояния клавиши
def is_key_pressed(key_code):
    return user32.GetAsyncKeyState(key_code) & 0x8000 != 0

# Коды клавиш
VK_Q = 0x51
VK_P = 0x50
VK_S = 0x53

def get_process_list():
    process_ids = (ctypes.c_ulong * 1024)()
    cb = ctypes.sizeof(process_ids)
    bytes_returned = ctypes.c_ulong()

    # Получение списка процессов
    if psapi.EnumProcesses(ctypes.byref(process_ids), cb, ctypes.byref(bytes_returned)):
        count = bytes_returned.value // ctypes.sizeof(ctypes.c_ulong)
        for i in range(count):
            process_id = process_ids[i]
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, process_id)
            if handle:
                image_filename = (ctypes.c_char * 1024)()
                if psapi.GetModuleBaseNameA(handle, None, ctypes.byref(image_filename), ctypes.sizeof(image_filename)):
                    # Получение командной строки, с которой был запущен процесс
                    cmdline_buffer = ctypes.create_string_buffer(1024)
                    length = ctypes.c_ulong(1024)
                    psapi.GetProcessImageFileNameA(handle, cmdline_buffer, length)
                    cmdline = cmdline_buffer.value.decode()
                    yield process_id, image_filename.value.decode(), cmdline
                kernel32.CloseHandle(handle)

def get_command_line(pid):
    h_process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if h_process:
        try:
            buffer_size = 1024
            lp_base_address = ctypes.c_void_p()
            lp_buffer = ctypes.create_string_buffer(buffer_size)
            lp_number_of_bytes_written = ctypes.c_ulong(0)

            kernel32.ReadProcessMemory(
                h_process,
                ctypes.c_void_p(2004397230288),  # Это адрес, где находится таблица PEB
                ctypes.byref(lp_base_address),
                ctypes.sizeof(lp_base_address),
                ctypes.byref(lp_number_of_bytes_written)
            )

            # Считываем аргументы командной строки из PEB
            kernel32.ReadProcessMemory(
                h_process,
                ctypes.cast(lp_base_address.value + 0x70, ctypes.c_void_p),
                ctypes.byref(lp_buffer),
                buffer_size,
                ctypes.byref(lp_number_of_bytes_written)
            )

            return lp_buffer.value.decode()
        except Exception as e:
            print(f"Error reading command line: {e}")
        finally:
            kernel32.CloseHandle(h_process)


def monitor_processes():
    while True:
        # Проверка нажатия клавиш
        if is_key_pressed(ord('q')):
            print("Выход")
            break
        elif is_key_pressed(ord('p')):
            print("Приостановлено. Для возобновления нажмите s")
            while True:
                if is_key_pressed(ord('s')):
                    print("Возобновлено. Для приостановки нажмите p")
                    break
        else:
            print("Process monitoring...")
            for process_id, process_name, cmdline in get_process_list():
                if 'conda' in process_name.lower():
                    print(f"Процесс conda был запущен с PID {process_id}")
                    print(f"Process ID: {process_id}, Process Name: {process_name}, Command Line: {cmdline}")
                    cmdline = get_command_line(process_id)
                    print(f"Command Line: {cmdline}")
            time.sleep(2)

if __name__ == "__main__":
    monitor_processes()
