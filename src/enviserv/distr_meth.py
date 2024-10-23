from importlib.metadata import distributions

# Получаем первый объект distribution
distribution = next(distributions(), None)

# Если distribution не равен None, т.е. есть установленные пакеты
if distribution:
    # Получаем список всех методов и атрибутов объекта distribution
    all_methods_and_attributes = dir(distribution)
    print(all_methods_and_attributes)
else:
    print("Нет установленных пакетов")