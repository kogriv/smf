import tracemalloc

# Включаем трассировку
tracemalloc.start()

# Ваш код, который вы хотите профилировать

# Получаем текущую статистику использования памяти
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 memory users ]")
for stat in top_stats[:100]:
    print(stat)
