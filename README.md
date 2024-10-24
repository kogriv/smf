# smf
NLP multiclass classification model

# Проект: Автоматизация классификации мультфильмов по проектам

## Описание проекта

Цель проекта — создать автоматизированное решение для классификации видеороликов по проектам на основе анализа текстовых описаний, субтитров и других доступных данных. Это важно для студий мультфильмов, поскольку позволяет принимать обоснованные бизнес-решения, увеличивать популярность новых проектов и более эффективно анализировать успешность старых. 

## Задачи:
- Определение принадлежности видеороликов к проектам на основе текста (описания, субтитры).
- Тексты могут быть на разных языках, содержать ошибки и опечатки.
- Видео могут быть размещены на разных каналах, а не все ролики можно обогатить дополнительными данными (например, лайв-стримы и закрытые каналы).
- Цель — максимизировать метрику классификации, сделав решение гибким и масштабируемым.

## Текущие результаты

### Максимизация метрики
Мной была достигнута максимальная F1-метрика **0.96** при использовании подхода на основе фильтрации и инструкций для классификации. Это сделало меня "чемпионом" на этом этапе, но метод пока не автоматизирован. Примерно понятно, как это сделать с использованием каскадной классификации методом "один-против-всех". Но автоматизация этого подхода в рамках исследования не была произведена.

### Альтернативный подход
Также был протестирован подход с использованием предобученных языковых моделей, таких как **DistilBERT**. На модели **distilbert-base-multilingual-cased** удалось достичь F1-метрики **0.56** при ограниченных вычислительных ресурсах (Nvidia 1060 6GB, 32GB ОЗУ). Обучение заняло около 5 часов на 5 эпох (1 час на эпоху).

## Гибридная модель

Создана гибридная модель, которая использует трансформеры для обработки текстовых данных и Dense-слои для обработки нетекстовых признаков. Выходы этих слоев объединяются для финальной классификации. Модель написана на основе предобученных трансформеров (например, DistilBERT) и поддерживает расширение для использования других трансформеров.

### Основные особенности:
- **Трансформер** для обработки текстов.
- **Dense слои** для числовых и других нетекстовых признаков.
- Объединение результатов двух частей модели для финальной классификации.

## Перспективы и выводы

Результат в 0.96 является хорошим, но не масштабируемым, так как метод основан на ручной фильтрации и инструкциях. В будущем можно автоматизировать этот процесс через каскадную классификацию.

При увеличении вычислительных мощностей (GPU и RAM) можно попробовать более мощные языковые модели и улучшить результаты, например, путем увеличения числа эпох, размеров батчей, длины токенов и изменения скорости обучения.

## Структура проекта

- **Тесты и исследования**: папка notebooks. Логика тренировки и тестирования моделей, включая методы оценки качества.
- **hymo.py**: папка srs содержит основной модуль hymo.py с классами для работы с моделью (включая гибридные архитектуры). также в папке другие сервисные модули

## Как запустить

1. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt

## Пример использования гибридной модели
Ниже приведен пример использования гибридной модели, которая объединяет трансформеры для обработки текстов и полносвязные слои для числовых данных. В этом примере используются как сниппеты данных, так и полные данные, и демонстрируется типичный пайплайн работы с моделью: от предобработки данных до обучения и оценки.

Шаги:
1. Инициализация процессора данных Мы используем `hymo.DataPreprocessor` для предобработки данных, включая текстовые данные для модели трансформера и числовые признаки.
```python
# Создадим обработчик данных
processor = hymo.DataPreprocessor(transformer_model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)
```
2. Подготовка сниппетов данных Мы можем использовать сниппеты данных, например, для быстрого тестирования. Поля выбираются в зависимости от структуры данных. Также можно задать количество отсутствующих или редких объектов для исключения.
```python
# Создадим сниппет данных для тестирования
sample_dtr = processor.get_data(df=dtr, use_snippet=True,
                                fields=['is_shorts', 'broadcast',
                                        'yt_channel_type', 'flag_closed', 'international'],
                                none_obj_count=2,
                                rare_obj_count=2)

sample_dts = processor.get_data(df=dts, use_snippet=True,
                                fields=['is_shorts', 'broadcast',
                                        'yt_channel_type', 'flag_closed', 'international'],
                                none_obj_count=2,
                                rare_obj_count=2)
```
3. Инициализация модели Гибридная модель, комбинирующая текстовые и числовые данные, создается с помощью `hymo.HybridModel`.
```python
# Создадим модель
hymodel = hymo.HybridModel(transformer_model_name=MODEL_NAME, num_labels=num_classes)
```
4. Настройка сервисных объектов для модели `hymo.ModelService` используется для управления параметрами модели, такими как оптимизатор, критерий потерь, кодировка целевой переменной и другие настройки.
```python
# Создадим необходимые объекты для модели
hyserv = hymo.ModelService(model=hymodel, transformer_model_name=MODEL_NAME,
                           df=sample_dtr, target=target,
                           optimizer_type='adam', learning_rate=LEARNING_RATE)
```
5. Предобработка данных. Текстовые и числовые данные подготавливаются для обучения и тестирования. Кодировка категориальных признаков и их трансформация происходят в методе `preprocess_data`.
```python
# Подготовим данные
train_text_encodings, train_numeric_features, le_broadcast, le_channel_type = \
processor.preprocess_data(sample_dtr, is_train=True)

test_text_encodings, test_numeric_features = \
processor.preprocess_data(sample_dts, is_train=False)
```
6. Инициализация тренера `hymo.Trainer` используется для управления процессом обучения, включая оптимизацию и обработку шагов градиентного накопления.
```python
# Создадим тренера
hytrainer = hymo.Trainer(model=hymodel,
                         optimizer=hyserv.optimizer,
                         criterion=hyserv.criterion,
                         device=hyserv.device,
                         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
```
7. Обучение модели. Модель обучается с помощью текстовых и числовых признаков, а целевые метки кодируются через ModelService.
```python
# Запустим обучение
hytrainer.train(
    train_text_encodings=train_text_encodings,
    train_numeric_features=train_numeric_features,
    train_labels=hyserv.encoded_labels,
    batch_size=BATCH_SIZE,
    num_epochs=num_epochs,
)
```
8. Оценка модели. После обучения модель можно оценить на тренировочных и тестовых данных. Если метки отсутствуют (например, для тестовых данных), метрики, такие как accuracy и F1-score, не будут рассчитаны, но предсказания все равно будут доступны.
```python
# Оценка на обучающих данных
preds_train = hytrainer.evaluate(train_text_encodings, train_numeric_features, hyserv.encoded_labels, BATCH_SIZE)

# Оценка на тестовых данных (без меток)
preds_test = hytrainer.evaluate(test_text_encodings, test_numeric_features, None, BATCH_SIZE)
```
## Заключение
Этот пример демонстрирует, как можно использовать гибридную модель с различными типами данных и как настраивать рабочий процесс для задач, где доступны и текстовые, и числовые данные. Подход работает как для небольших сниппетов, так и для полного набора данных.