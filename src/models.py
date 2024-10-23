    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    from tqdm import tqdm

    # Гибридная модель с трансформером и числовыми признаками
    class HybridModel(nn.Module):
        def __init__(self, transformer_model_name, num_labels, num_numeric_features):
            super(HybridModel, self).__init__()
            
            # Модель трансформера для текстов (уменьшенная версия DistilBERT)
            self.transformer = AutoModel.from_pretrained(transformer_model_name)
            
            # Полносвязный слой для обработки текстовых данных
            self.text_fc = nn.Linear(self.transformer.config.hidden_size, 128)
            
            # Полносвязный слой для числовых данных (и других нетекстовых признаков)
            self.numeric_fc = nn.Linear(num_numeric_features, 128)
            
            # Финальный классификационный слой
            self.classifier = nn.Linear(128 * 2, num_labels)
            
        def forward(self, text_input, numeric_input):
            # Обрабатываем текстовые данные через трансформер
            transformer_output = self.transformer(**text_input)
            # Используем последнее скрытое состояние и берем среднее по всем токенам
            text_features = torch.mean(transformer_output.last_hidden_state, dim=1)
            text_features = self.text_fc(text_features)
            
            # Обрабатываем числовые данные
            numeric_features = self.numeric_fc(numeric_input)
            
            # Объединяем текстовые и числовые признаки
            combined_features = torch.cat((text_features, numeric_features), dim=1)
            
            # Применяем финальный классификационный слой
            logits = self.classifier(combined_features)
            
            return logits

    # Инициализируем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Кодирование меток
    le = LabelEncoder()
    y_train = le.fit_transform(dtr['cartoon'])
    labels = torch.tensor(y_train, dtype=torch.long)

    # Создаем модель
    model = HybridModel(MODEL_NAME, num_labels=num_classes,
                        num_numeric_features=train_numeric_features.shape[1])
    model.to(device)

    # Вычисляем веса классов
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Оптимизатор
    # Используемый оптимизатор. Возможные альтернативы: Adam, SGD.
    # AdamW подходит для трансформеров, но может использовать больше памяти, чем SGD.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) 
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)




# Функция для разделения данных на батчи (для предсказания метки не нужны)
def batch_loader(text_encodings, numeric_features, labels=None, batch_size=32):
    data_size = len(numeric_features)  # используем размер числовых признаков (или текстов, они одинаковы по размеру)
    
    for i in range(0, data_size, batch_size):
        if labels is not None:
            yield {k: v[i:i + batch_size] for k, v in text_encodings.items()}, numeric_features[i:i + batch_size], labels[i:i + batch_size]
        else:
            yield {k: v[i:i + batch_size] for k, v in text_encodings.items()}, numeric_features[i:i + batch_size]

# Функция для токенизации и обработки признаков
def preprocess_data(df, tokenizer, max_seq_length, is_train=True, le_broadcast=None, le_channel_type=None):
    """
    Функция для токенизации текстовых данных и обработки числовых/категориальных признаков.
    
    Аргументы:
    - df: DataFrame с данными
    - tokenizer: токенайзер для текстовых данных
    - max_seq_length: максимальная длина последовательности для токенайзера
    - is_train: булево значение, указывающее, тренировочный ли это набор данных (True) или тестовый (False)
    - le_broadcast: объект LabelEncoder для поля broadcast (используется, если is_train=False)
    - le_channel_type: объект LabelEncoder для поля yt_channel_type (используется, если is_train=False)
    
    Возвращает:
    - text_encodings: токенизированные текстовые данные
    - numeric_features: тензор с числовыми признаками
    - (le_broadcast, le_channel_type): возвращает энкодеры, если is_train=True, иначе None
    """
    
    # 1. Объединение текстовых полей для токенизации
    df['combined_text'] = df['text'] + ' ' + df['reel_name']

    # 2. Токенизация текста
    text_encodings = tokenizer(
        list(df['combined_text']),
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    
    # 3. Обработка категориальных признаков
    if is_train:
        # Для тренировочного набора данных создаем энкодеры и обучаем их
        le_broadcast = LabelEncoder()
        df['broadcast_encoded'] = le_broadcast.fit_transform(df['broadcast'])

        le_channel_type = LabelEncoder()
        df['yt_channel_type_encoded'] = le_channel_type.fit_transform(df['yt_channel_type'])
    else:
        # Для тестового набора используем уже обученные энкодеры
        df['broadcast_encoded'] = le_broadcast.transform(df['broadcast'])
        df['yt_channel_type_encoded'] = le_channel_type.transform(df['yt_channel_type'])
    
    # 4. Преобразование числовых признаков в тензор
    numeric_features = torch.tensor(
        df[['seconds', 'is_shorts', 'broadcast_encoded', 'yt_channel_type_encoded', 
            'flag_closed', 'international']].values,
        dtype=torch.float32
    )
    
    # Если это тренировочный набор данных, возвращаем также энкодеры
    if is_train:
        return text_encodings, numeric_features, le_broadcast, le_channel_type
    else:
        return text_encodings, numeric_features

# ===================================================
# внешние параметры

# Выбор модели трансформера
MODEL_NAME = 'distilbert-base-multilingual-cased'  
# Другие возможные модели:
# 'distilbert-base-uncased' - еще более легкая версия модели BERT,
                            # работает быстрее, но может быть менее точной.
# 'bert-base-multilingual-cased' - более мощная версия модели, требует больше памяти.
# 'xlm-roberta-base' - мощная многоязычная модель, больше памяти и вычислительных затрат.
# Выбор модели влияет на производительность и точность:
# более легкие модели требуют меньше памяти, но могут снижать качество результатов.

# Изменяемые параметры

BATCH_SIZE = 16  # Размер батча. Возможные значения: от 8 до 64.
                 # Меньшие значения уменьшают использование памяти,
                 # но увеличивают количество шагов на эпоху.
GRADIENT_ACCUMULATION_STEPS = 4  # Количество шагов для накопления градиентов.
                 # Возможные значения: 1-8. Большее значение уменьшает
                 # нагрузку на память за счет увеличения времени обучения.
MAX_SEQ_LENGTH = 256  # Максимальная длина последовательности (в токенах).
                 # Возможные значения: от 64 до 512. Меньшие значения
                 # уменьшают использование памяти, но могут обрезать важные части текста.

LEARNING_RATE = 1e-5  # Скорость обучения. Возможные значения: 1e-6 - 1e-4.
                 # Меньшие значения могут потребовать больше эпох для обучения,
                 # но сделать обучение более стабильным.

num_epochs = 5
num_classes = 45

# ================================================


train_text_encodings, train_numeric_features, le_broadcast, le_channel_type = preprocess_data(
    dtr, tokenizer, MAX_SEQ_LENGTH, is_train=True
)



# Выбираем все строки, где класс не 'none'
rare_class_mask = sample_dtr['cartoon'] != 'none'
rare_class_indices = sample_dtr[rare_class_mask].index

# Выбираем строки с классом 'none'
none_class_mask = sample_dtr['cartoon'] == 'none'
none_class_indices = sample_dtr[none_class_mask].index

# Определяем, какой процент данных класса 'none' добавить (например, 10%)
none_sample_size = int(1 * len(none_class_indices))
none_sample_indices = np.random.choice(none_class_indices, none_sample_size, replace=False)

# Объединяем индексы редких классов и часть данных класса 'none'
selected_indices = np.concatenate([rare_class_indices, none_sample_indices])

selected_indices = dtr.index

# Теперь отбираем данные по этим индексам
sample_text_encodings = {k: v[selected_indices] for k, v in train_text_encodings.items()}
sample_numeric_features = train_numeric_features[selected_indices]
sample_labels = labels[selected_indices]

# Обучение с использованием батчей и градиентного накопления
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    
    for i, (batch_text, batch_numeric, batch_labels) \
            in enumerate(batch_loader(sample_text_encodings,
                                      sample_numeric_features,
                                      sample_labels,
                                      BATCH_SIZE)):
        batch_text = {k: v.to(device) for k, v in batch_text.items()}
        batch_numeric = batch_numeric.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        logits = model(batch_text, batch_numeric)
        loss = criterion(logits, batch_labels)
        loss = loss / GRADIENT_ACCUMULATION_STEPS  # Делим loss для накопления
        
        # Backward pass
        loss.backward()
        
        # Шаг оптимизатора на каждые GRADIENT_ACCUMULATION_STEPS
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')


# Расчет метрик. Перенесем модель и данные на ЦПУ в случае нехватки памяти на ГПУ
model_cpu = model.to('cpu')

sample_text_encodings = {k: v.to(device) for k, v in sample_text_encodings.items()}
sample_numeric_features = sample_numeric_features.to(device)
sample_labels = sample_labels.to(device)

sample_text_encodings_cpu = {k: v.to('cpu') for k, v in sample_text_encodings.items()}
sample_numeric_features_cpu = sample_numeric_features.to('cpu')
sample_labels_cpu = sample_labels.to('cpu')  # Если метки тоже на GPU

torch.cuda.empty_cache()  # Очищает кеш, но не освобождает всю память

model_cpu.eval()  # Переключаем модель в режим оценки
with torch.no_grad():  # Отключаем градиенты
    logits = model_cpu(sample_text_encodings_cpu, sample_numeric_features_cpu)
    predicted_probs = torch.softmax(logits, dim=1)  # Вероятности для каждого класса
    predicted_labels = torch.argmax(predicted_probs, dim=1)  # Предсказанные метки

    # Теперь можно рассчитать метрики, например:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    accuracy = accuracy_score(sample_labels_cpu, predicted_labels)
    f1 = f1_score(sample_labels_cpu, predicted_labels, average='weighted')
    # Если у вас многоклассовая классификация, можно использовать `average='macro'` или 'weighted'
    
    print(f'Accuracy: {accuracy}, F1: {f1}')

test_text_encodings, test_numeric_features = preprocess_data(
    dts, tokenizer, MAX_SEQ_LENGTH, is_train=False, le_broadcast=le_broadcast, le_channel_type=le_channel_type
)

print("Train numeric features shape:", train_numeric_features.shape)
print("Test numeric features shape:", test_numeric_features.shape)

# Переключаем модель в режим оценки
model.eval()

# Отключаем градиенты, чтобы сэкономить память
with torch.no_grad():
    all_preds = []
    
    # Прогоняем данные через модель батчами
    for batch_text, batch_numeric in batch_loader(test_text_encodings, test_numeric_features, labels=None, batch_size=BATCH_SIZE):
        batch_text = {k: v.to(device) for k, v in batch_text.items()}
        batch_numeric = batch_numeric.to(device)
        
        # Получаем предсказания
        logits = model(batch_text, batch_numeric)
        
        # Получаем предсказанные классы
        predicted_labels = torch.argmax(logits, dim=1)
        
        # Сохраняем предсказания
        all_preds.extend(predicted_labels.cpu().numpy())

