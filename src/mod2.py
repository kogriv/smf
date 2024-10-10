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
MAX_SEQ_LENGTH = 128  # Максимальная длина последовательности (в токенах).
                 # Возможные значения: от 64 до 512. Меньшие значения
                 # уменьшают использование памяти, но могут обрезать важные части текста.

LEARNING_RATE = 1e-5  # Скорость обучения. Возможные значения: 1e-6 - 1e-4.
                 # Меньшие значения могут потребовать больше эпох для обучения,
                 # но сделать обучение более стабильным.

num_epochs = 5
num_classes = 45

# Инициализируем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Функция для разделения данных на батчи с одинаковым размером для текста и числовых признаков
def batch_loader(text_encodings, numeric_features, labels, batch_size):
    for i in range(0, len(labels), batch_size):
        yield {k: v[i:i + batch_size] for k, v in text_encodings.items()}, numeric_features[i:i + batch_size], labels[i:i + batch_size]

# Подготовка данных (токенизация с обрезкой)
text_encodings = tokenizer(list(dtr['text']), truncation=True, padding=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
numeric_features = torch.tensor(dtr[['seconds', 'flag_closed', 'international']].values, dtype=torch.float32)

# Кодирование меток
le = LabelEncoder()
y_train = le.fit_transform(dtr['cartoon'])
labels = torch.tensor(y_train, dtype=torch.long)

# Создаем модель
model = HybridModel(MODEL_NAME, num_labels=num_classes,
                    num_numeric_features=numeric_features.shape[1])
model.to(device)

# Вычисляем веса классов
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Оптимизатор
# Используемый оптимизатор. Возможные альтернативы: Adam, SGD.
# AdamW подходит для трансформеров, но может использовать больше памяти, чем SGD.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) 

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# # Быстрая проверка кода на части данных
# # Используем часть данных для быстрой отладки (например, 10% от общего объема)
# sample_size = int(0.1 * len(labels))
# sample_text_encodings = {k: v[:sample_size] for k, v in text_encodings.items()}
# sample_numeric_features = numeric_features[:sample_size]
# sample_labels = labels[:sample_size]

# Выбираем все строки, где класс не 'none'
rare_class_mask = dtr['cartoon'] != 'none'
rare_class_indices = dtr[rare_class_mask].index

# Выбираем строки с классом 'none'
none_class_mask = dtr['cartoon'] == 'none'
none_class_indices = dtr[none_class_mask].index

# Определяем, какой процент данных класса 'none' добавить (например, 10%)
none_sample_size = int(0.07 * len(none_class_indices))
none_sample_indices = np.random.choice(none_class_indices, none_sample_size, replace=False)

# Объединяем индексы редких классов и часть данных класса 'none'
selected_indices = np.concatenate([rare_class_indices, none_sample_indices])

# Теперь отбираем данные по этим индексам
sample_text_encodings = {k: v[selected_indices] for k, v in text_encodings.items()}
sample_numeric_features = numeric_features[selected_indices]
sample_labels = labels[selected_indices]

# Изменим BATCH_SIZE для быстрой проверки
BATCH_SIZE = 2  # можно увеличить для большего объема

# Обучение с использованием батчей и градиентного накопления
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    
    for i, (batch_text, batch_numeric, batch_labels) in enumerate(batch_loader(sample_text_encodings, sample_numeric_features, sample_labels, BATCH_SIZE)):
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

