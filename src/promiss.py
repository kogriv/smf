# module with finctions for processing misiing values
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def base_dataset_info():

    bdf = """
--------------------------------------------------
 base dataset info

 0   date               83411 non-null  object 
 1   reel_name          83408 non-null  object 
 2   yt_reel_id         83411 non-null  object 
 3   cartoon            83411 non-null  object 
 4   url                83411 non-null  object 
 5   text               83411 non-null  object 
 6   seconds            74653 non-null  float64
 7   is_shorts          74653 non-null  float64
 8   broadcast          74653 non-null  object 
 9   yt_channel_id      83366 non-null  object 
 10  yt_channel_name    83363 non-null  object 
 11  yt_ch_url          83363 non-null  object 
 12  yt_channel_type    83363 non-null  object 
 13  flag_closed        83363 non-null  float64
 14  international      83363 non-null  float64
 15  language           497 non-null    object 
 16  language_detected  83411 non-null  object 
"""
    print(bdf)

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, fill_strategy='basic', fill_values=None):
        """
        Класс для обработки пропусков в данных.

        Параметры:
        - fill_strategy: стратегия заполнения пропусков. Возможные значения:
            - 'basic': базовое заполнение (медиана для числовых, мода для категориальных).
            - 'custom': заполнение значениями, переданными через fill_values.
            - 'drop': удаление строк с пропусками.
        - fill_values: словарь значений для кастомного заполнения. 
          Ключ - название колонки, значение - значение для заполнения пропусков.
        """
        self.fill_strategy = fill_strategy
        self.fill_values = fill_values or {}

    def fit(self, X, y=None):
        if self.fill_strategy == 'basic':
            self.fill_values_ = {}
            for column in X.columns:
                if X[column].dtype == np.number:
                    self.fill_values_[column] = X[column].median()
                else:
                    self.fill_values_[column] = X[column].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()

        if self.fill_strategy == 'basic':
            for column, value in self.fill_values_.items():
                X[column].fillna(value, inplace=True)

        elif self.fill_strategy == 'custom':
            for column, value in self.fill_values.items():
                X[column].fillna(value, inplace=True)

        elif self.fill_strategy == 'drop':
            X.dropna(inplace=True)

        return X
    

