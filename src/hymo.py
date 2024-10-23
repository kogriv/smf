# Модуль с классами для создания гибридной модели, обработки даныых и обучения
import pandas as pd
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
    """
    Гибридная модель, объединяющая трансформер и числовые признаки для задач классификации.

    Args:
        transformer_model_name (str): Имя предобученной модели трансформера.
        num_labels (int): Количество классов для классификации.
        num_numeric_features (int): Количество числовых признаков/ по умолчанию = 6.

    Attributes:
        transformer (nn.Module): Модель трансформера для обработки текстовых данных.
        text_fc (nn.Linear): Полносвязный слой для обработки выходов трансформера.
        numeric_fc (nn.Linear): Полносвязный слой для обработки числовых данных.
        classifier (nn.Linear): Финальный классификационный слой.
    """
    
    def __init__(self, transformer_model_name: str, num_labels: int, num_numeric_features: int = 6) -> None:
        super(HybridModel, self).__init__()
        
        # Модель трансформера для текстов (уменьшенная версия DistilBERT)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        
        # Полносвязный слой для обработки текстовых данных
        self.text_fc = nn.Linear(self.transformer.config.hidden_size, 128)
        
        # Полносвязный слой для числовых данных (и других нетекстовых признаков)
        self.numeric_fc = nn.Linear(num_numeric_features, 128)
        
        # Финальный классификационный слой
        self.classifier = nn.Linear(128 * 2, num_labels)
        
    def forward(self, text_input: dict, numeric_input: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход по модели.

        Args:
            text_input (dict): Входные данные текстового типа, включающие токены и их маски.
            numeric_input (torch.Tensor): Входные данные числового типа.

        Returns:
            torch.Tensor: Логиты для классов.
        """
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
        
class ModelService:
    def __init__(self, model, transformer_model_name, df, target, optimizer_type='adam', learning_rate=5e-5):
        """
        Инициализация сервисного класса для работы с гибридной моделью.
        
        Args:
        - model: объект класса HybridModel
        - transformer_model_name: название модели трансформера
        - target: целевая переменная (метки)
        - optimizer_type: тип оптимизатора ('adam' или 'sgd')
        - learning_rate: скорость обучения для оптимизатора
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Кодирование целевой переменной
        self.le = LabelEncoder()
        self.encoded_labels = self._encode_labels(df,target)
        
        # Вычисление весов классов
        self.class_weights = self._compute_class_weights(self.encoded_labels)
        
        # Инициализация критерия
        self.criterion = self._get_criterion()
        
        # Инициализация оптимизатора
        self.optimizer = self._get_optimizer(optimizer_type, learning_rate)
    
    def _encode_labels(self, df, target):
        """
        Кодирование целевой переменной.
        
        Args:
        - target: целевая переменная (метки)
        
        Returns:
        - закодированные метки в формате torch.tensor
        """
        encoded_labels = self.le.fit_transform(df[target])
        return torch.tensor(encoded_labels, dtype=torch.long).to(self.device)

    def _compute_class_weights(self, encoded_labels):
        """
        Вычисление весов классов для балансировки
        
        Args:
        - encoded_labels: закодированные метки
        
        Returns:
        - class_weights: веса классов в формате torch.tensor
        """
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(encoded_labels.cpu().numpy()),
                                             y=encoded_labels.cpu().numpy())
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def _get_optimizer(self, optimizer_type, learning_rate):
        """
        Инициализация оптимизатора для обучения модели.
        
        Args:
        - optimizer_type: тип оптимизатора ('adam' или 'sgd')
        - learning_rate: скорость обучения
        
        Returns:
        - оптимизатор для модели
        """
        if optimizer_type == 'adam':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError("Неправильный тип оптимизатора: выберите 'adam' или 'sgd'.")
        return optimizer
    
    def _get_criterion(self):
        """
        Определяет функцию потерь с учетом весов классов.
        
        Returns:
        - Функция потерь CrossEntropyLoss с весами классов.
        """
        return nn.CrossEntropyLoss(weight=self.class_weights)

class DataPreprocessor:
    def __init__(self, transformer_model_name,max_seq_length):
        """
        Инициализация класса DataPreprocessor.
        
        Args:
        - tokenizer: токенайзер для текстовых данных
        - max_seq_length: максимальная длина последовательности для токенайзера
        """
        # self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.max_seq_length = max_seq_length
        self.le_broadcast = None
        self.le_channel_type = None

    def preprocess_data(self, df, is_train=True):
        """
        Токенизация текстовых данных и обработка числовых/категориальных признаков.
        
        Args:
        - df: DataFrame с данными
        - is_train: True для тренировочного набора данных, False для тестового
        
        Returns:
        - text_encodings: токенизированные текстовые данные
        - numeric_features: тензор с числовыми признаками
        - (le_broadcast, le_channel_type): возвращает энкодеры для тренировочного набора
        """
        # 1. Объединение текстовых полей для токенизации
        df['combined_text'] = df['text'] + ' ' + df['reel_name']

        # 2. Токенизация текста
        text_encodings = self.tokenizer(
            list(df['combined_text']),
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # 3. Обработка категориальных признаков
        if is_train:
            # Для тренировочного набора создаем и обучаем энкодеры
            self.le_broadcast = LabelEncoder()
            df['broadcast_encoded'] = self.le_broadcast.fit_transform(df['broadcast'])

            self.le_channel_type = LabelEncoder()
            df['yt_channel_type_encoded'] = self.le_channel_type.fit_transform(df['yt_channel_type'])
        else:
            # Для тестового набора используем уже обученные энкодеры
            df['broadcast_encoded'] = self.le_broadcast.transform(df['broadcast'])
            df['yt_channel_type_encoded'] = self.le_channel_type.transform(df['yt_channel_type'])
        
        # 4. Преобразование числовых признаков в тензор
        numeric_features = torch.tensor(
            df[['seconds', 'is_shorts', 'broadcast_encoded', 'yt_channel_type_encoded', 
                'flag_closed', 'international']].values,
            dtype=torch.float32
        )
        
        if is_train:
            return text_encodings, numeric_features, self.le_broadcast, self.le_channel_type
        else:
            return text_encodings, numeric_features
    
    def df_sample(self, df, fields, n):
        """
        Функция для формирования тестовых сниппетов с выборкой строк для всех уникальных значений в категориальных полях.
        
        Args:
        - df: DataFrame с данными
        - fields: список категориальных полей
        - n: количество выборок для каждого уникального значения поля
        
        Returns:
        - DataFrame со сниппетами
        """
        sample_df = pd.DataFrame()
        for field in fields:
            if 'cartoon' in df.columns:
                # Для строк, где cartoon == 'none'
                sample_df = pd.concat([sample_df, df[df['cartoon'] == 'none'].groupby(field).sample(n=n, random_state=42)])
            else:
                # Для остальных строк
                sample_df = pd.concat([sample_df, df.groupby(field).sample(n=n, random_state=42)])
        sample_df = sample_df.drop_duplicates()
        return sample_df

    def get_data(self, df, use_snippet=True, fields=['is_shorts', 'broadcast',
                                                     'yt_channel_type', 'flag_closed',
                                                     'international'],
                 none_obj_count=2, rare_obj_count=2):
        """
        Функция для возвращения либо всего DataFrame, либо минимального сниппета.
        
        Args:
        - df: DataFrame с данными
        - use_snippet: True, если нужно использовать сниппет, False для использования всех данных
        - fields: список категориальных полей
        - none_obj_count: количество выборок для строк, где cartoon == 'none'
        - rare_obj_count: количество выборок для строк, где cartoon != 'none'
        
        Returns:
        - DataFrame либо полный, либо сниппет
        """
        if use_snippet:
            # Формируем сниппет
            sample_df_none = self.df_sample(df, fields, none_obj_count)
            sample_df_rare = df[df['cartoon'] != 'none'].groupby('cartoon').sample(n=rare_obj_count, random_state=42)
            sample_df = pd.concat([sample_df_none, sample_df_rare])
            return sample_df
        else:
            # Возвращаем весь DataFrame
            return df
    
    def prepare_data_for_training(self, dtr,
                                  text_encodings, numeric_features,
                                  labels):
        """
        Функция для токенизации данных, обработки признаков и выбора данных для обучения.
        
        Args:
        - dtr: DataFrame с полными данными
        - text_encodings, numeric_features: токенизированный текст, и тензор числовых данных
        - labels: Закодированные метки (labels из ModelService)
        - is_train: Является ли это тренировочными данными (True) или тестовыми (False)
        
        Returns:
        - sample_text_encodings: Токенизированные текстовые данные, отобранные по индексам
        - sample_numeric_features: Числовые признаки, отобранные по индексам
        - sample_labels: Метки классов, отобранные по индексам
        """
        
        # Шаг 1. Препроцессинг полного набора данных
        # train_text_encodings, train_numeric_features, le_broadcast, le_channel_type = self.preprocess_data(
        #     dtr, self.tokenizer, self.max_seq_length, is_train=is_train
        # )
        
        # # Шаг 2. Выбираем все строки, где класс не 'none'
        # rare_class_mask = dtr['cartoon'] != 'none'
        # rare_class_indices = dtr[rare_class_mask].index

        # # Шаг 3. Выбираем строки с классом 'none'
        # none_class_mask = dtr['cartoon'] == 'none'
        # none_class_indices = dtr[none_class_mask].index

        # # Шаг 4. Объединяем индексы редких классов и класса 'none'
        # selected_indices = np.concatenate([rare_class_indices, none_class_indices])

        selected_indices = dtr.index.to_
        print(len(selected_indices))

        # Шаг 5. Отбираем данные по этим индексам
        sample_text_encodings = {k: v[selected_indices] for k, v in text_encodings.items()}
        sample_numeric_features = numeric_features #[selected_indices]
        sample_labels = labels #[selected_indices]

        return sample_text_encodings, sample_numeric_features, sample_labels
    

class Trainer:
    def __init__(self, model, optimizer, criterion, device, gradient_accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def batch_loader(self, text_encodings, numeric_features, labels=None, batch_size=32):
        """
        Функция для разделения данных на батчи.
        
        Args:
        - text_encodings: Токенизированные текстовые данные
        - numeric_features: Числовые признаки
        - labels: Метки классов (если есть)
        - batch_size: Размер батча
        
        Yields:
        - Текущий батч текстовых данных, числовых признаков и меток классов (если есть)
        """
        data_size = len(numeric_features)  # используем размер числовых признаков (или текстов, они одинаковы по размеру)

        for i in range(0, data_size, batch_size):
            if labels is not None:
                yield (
                    {k: v[i:i + batch_size] for k, v in text_encodings.items()},
                    numeric_features[i:i + batch_size],
                    labels[i:i + batch_size]
                )
            else:
                yield (
                    {k: v[i:i + batch_size] for k, v in text_encodings.items()},
                    numeric_features[i:i + batch_size]
                )

    def train(self, train_text_encodings, train_numeric_features, train_labels, batch_size, num_epochs):
        """
        Функция для обучения модели с использованием батчей и градиентного накопления.

        Args:
        - train_text_encodings: Токенизированные текстовые данные для обучения
        - train_numeric_features: Числовые признаки для обучения
        - train_labels: Метки классов для обучения
        - batch_size: Размер батча
        - num_epochs: Количество эпох
        """
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            total_loss = 0

            for i, (batch_text, batch_numeric, batch_labels) \
                    in enumerate(self.batch_loader(train_text_encodings,
                                                   train_numeric_features,
                                                   train_labels,
                                                   batch_size)):

                batch_text = {k: v.to(self.device) for k, v in batch_text.items()}
                batch_numeric = batch_numeric.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                logits = self.model(batch_text, batch_numeric)
                loss = self.criterion(logits, batch_labels)
                loss = loss / self.gradient_accumulation_steps  # Делим loss для накопления

                # Backward pass
                loss.backward()

                # Шаг оптимизатора на каждые GRADIENT_ACCUMULATION_STEPS
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')
