class Dummy:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return 'dummy_class'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        # Определяем логику сравнения
        if isinstance(other, Dummy):
            # Сравниваем объекты по длине имени
            if 1 <= len(self.name) <= 10 and 1 <= len(other.name) <= 10:
                return True
            elif 11 <= len(self.name) <= 20 and 11 <= len(other.name) <= 20:
                return True
            elif len(self.name) >= 21 and len(other.name) >= 21:
                return True
        return False
    
obj1 = Dummy('first')
obj2 = Dummy('second')
obj3 = Dummy('more_ten_letters')
obj4 = Dummy('more_than_twenty_letters')

objects = [obj1,obj2,obj3]

print("obj1:", obj1)
print("obj2:", obj2)
print("obj3:", obj3)
print("obj4:", obj4)

print('equation of obj1 and obj2:',obj1==obj2)
print('obj4 is in [obj1,obj2,obj3]:',obj4 in objects)

# Хэши объектов
print("Хэш obj1:", hash(obj1))
print("Хэш obj2:", hash(obj2))

dict1 = {obj1:1,obj2:2}

print(dict1)
