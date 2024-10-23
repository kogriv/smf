# module with finctions for processing misiing values
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

missval_dict = {
'date'           : 'agg_mode',
'reel_name'      : 'value_NA',
'yt_reel_id'     : 'drop',
'cartoon'        : 'value_none',
'url'            : 'value_NA',
'text'           : 'value_NA',
'seconds'        : 'agg_median_by_yt_channel_type',
'is_shorts'      : 'agg_mode',
'broadcast'      : 'agg_mode_by_yt_channel_type',
'yt_channel_id'  : 'value_NA',
'yt_channel_name': 'value_NA',
'yt_ch_url'      : 'value_NA',
'yt_channel_type': 'agg_mode',
'flag_closed'    : 'agg_mode',
'international'  : 'agg_mode',
'language'       : 'value_NA'
}


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
            - 'custom_basic': кастомная стратегия на основе словаря missval_dict.
        - fill_values: словарь значений для кастомного заполнения, либо для custom_basic стратегии.
          Ключ - название колонки, значение - значение для обработки пропусков.
        """
        self.fill_strategy = fill_strategy
        self.fill_values = fill_values or {}

    def fit(self, X, y=None):
        if self.fill_strategy == 'basic':
            self.fill_values_ = {}
            for column in X.columns:
                if X[column].dtype == np.number:
                    if X[column].nunique() < 10:
                        self.fill_values_[column] = X[column].mode()[0]
                    else:
                        self.fill_values_[column] = X[column].median()
                else:
                    self.fill_values_[column] = X[column].mode()[0]
        elif self.fill_strategy == 'custom_basic':
            self.fill_values_ = self.fill_values  # Используем переданный словарь missval_dict
        return self

    def transform(self, X):
        # print('test_transform')
        X = X.copy()

        # Обрезаем словарь fill_values_, оставляя только колонки, которые есть в X
        relevant_fill_values = {col: val for col, val in self.fill_values_.items() if col in X.columns}

        # print(relevant_fill_values)

        if self.fill_strategy == 'basic':
            for column, value in relevant_fill_values.items():
                X[column].fillna(value, inplace=True)

        elif self.fill_strategy == 'custom':
            # Аналогично обрезаем self.fill_values для custom стратегии
            relevant_fill_values = {col: val for col, val in self.fill_values.items() if col in X.columns}
            for column, value in relevant_fill_values.items():
                X[column].fillna(value, inplace=True)

        elif self.fill_strategy == 'custom_basic':
            X = self._apply_custom_basic(X)

        elif self.fill_strategy == 'drop':
            X.dropna(inplace=True)

        return X

    def _apply_custom_basic(self, X):
        relevant_fill_values = {col: val for col, val in self.fill_values_.items() if col in X.columns}

        for column, strategy in relevant_fill_values.items():
            if strategy.startswith('value_'):
                value = strategy.split('value_')[1]
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass

                if isinstance(value, (int, float)):
                    X[column] = X[column].fillna(value)
                else:
                    if value.lower() == 'na':
                        X[column] = X[column].fillna('NA')
                    elif value.lower() == 'none':
                        X[column] = X[column].fillna('none')
                    else:
                        X[column] = X[column].fillna(value)

            elif strategy.startswith('agg_'):
                agg_info = strategy.split('agg_')[1]
                if '_by_' in agg_info:
                    agg_func, group_column = agg_info.split('_by_')
                    X[column] = self._fill_agg_by_group(X, column, group_column, agg_func)
                else:
                    agg_func = agg_info
                    X[column] = X[column].fillna(self._calculate_agg(X, column, agg_func))

        return X

    def _fill_agg_by_group(self, X, column, group_column, agg_func):
        grouped = X.groupby(group_column)[column]

        if agg_func == 'median':
            fill_values = grouped.transform('median')
            fallback_value = X[column].median()
        elif agg_func == 'mean':
            fill_values = grouped.transform('mean')
            fallback_value = X[column].mean()
        elif agg_func == 'mode':
            fill_values = grouped.transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
            fallback_value = X[column].mode()[0] if not X[column].mode().empty else np.nan
        elif agg_func == 'min':
            fill_values = grouped.transform('min')
            fallback_value = X[column].min()
        elif agg_func == 'max':
            fill_values = grouped.transform('max')
            fallback_value = X[column].max()
        elif agg_func == 'sum':
            fill_values = grouped.transform('sum')
            fallback_value = X[column].sum()
        elif agg_func == 'std':
            fill_values = grouped.transform('std')
            fallback_value = X[column].std()
        elif agg_func == 'var':
            fill_values = grouped.transform('var')
            fallback_value = X[column].var()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

        fill_values = fill_values.fillna(fallback_value)
        return X[column].fillna(fill_values)


    def _calculate_agg(self, X, column, agg_func):
        if agg_func == 'median':
            return X[column].median()
        elif agg_func == 'mean':
            return X[column].mean()
        elif agg_func == 'mode':
            return X[column].mode()[0] if not X[column].mode().empty else np.nan
        elif agg_func == 'min':
            return X[column].min()
        elif agg_func == 'max':
            return X[column].max()
        elif agg_func == 'sum':
            return X[column].sum()
        elif agg_func == 'std':
            return X[column].std()
        elif agg_func == 'var':
            return X[column].var()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

    

