from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

def plot_class_distribution(df, target_column='cartoon', verbose=False):
    # Подсчёт количества примеров для каждого класса
    class_counts = df[target_column].value_counts()
    
    if verbose:
        # Вывод распределения классов
        print("Class distribution:\n", class_counts)

    # Построение графика распределения классов
    class_counts.plot(kind='bar', title='Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

# Устранение дисбаланса
def balance_dataset(df, target_column, method='smote'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if method == 'smote':
        # Используем SMOTE для увеличения редких классов
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("After SMOTE, class distribution: ", Counter(y_balanced))

    elif method == 'undersample':
        # Используем undersampling для уменьшения крупных классов
        undersample = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersample.fit_resample(X, y)
        print("After undersampling, class distribution: ", Counter(y_balanced))

    return X_balanced, y_balanced

# Вычисление взвешенных потерь (в качестве альтернативы балансу данных)
def compute_class_weights(df, target_column):
    # Вычисляем веса классов
    class_weights = compute_class_weight('balanced', classes=np.unique(df[target_column]), y=df[target_column])
    
    # Словарь весов классов
    class_weight_dict = dict(enumerate(class_weights))
    
    print("Class weights:", class_weight_dict)
    return class_weight_dict