import re
import os
# import nltk
from nltk.corpus import stopwords
import pandas as pd

try:
    # Проверяем, выполняется ли код на Kaggle
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    # Попробуем импортировать необходимые библиотеки
    try:
        from langdetect import detect, DetectorFactory
        from langdetect.lang_detect_exception import LangDetectException

        # Настройка для стабилизации детектора языка
        DetectorFactory.seed = 0

        # Функция для определения языка текста
        def detect_language(text):
            try:
                return detect(text)
            except LangDetectException:
                return 'unknown'

        # Функция для определения языка текста с проверкой столбца 'language_detected'
        def detect_language_column(df):
            if 'language_detected' not in df.columns:
                df['language_detected'] = df['text'].apply(detect_language)
            else:
                print("'language_detected' уже существует в DataFrame.")

        # Подсчёт доли каждого языка в датасете
        def language_distribution(df, column):
            language_counts = df[column].value_counts(normalize=True)
            print(f"Распределение языков в датасете:\n{language_counts}")

    except ImportError as e:
        if is_kaggle:
            print(f"Модуль langdetect недоступен на Kaggle: {e}")
        else:
            # На локальной машине можно попробовать установить модуль
            print(f"Модуль langdetect не установлен: {e}")
            print("Попробуйте установить его с помощью 'pip install langdetect'.")

except Exception as e:
    print(f"Произошла ошибка: {e}")

# распределение языков в датасете
lang_distrib_train = {
'ru' :      0.790567,
'en' :      0.152090,
'bg' :      0.012960,
'pl' :      0.009064,
'ja' :      0.006546,
'pt' :      0.006138,
'et' :      0.005815,
'ar' :      0.003369,
'uk' :      0.003105,
'es' :      0.002805,
'mk' :      0.001247,
'ca' :      0.000935,
'no' :      0.000623,
'hi' :      0.000539,
'id' :      0.000480,
'nl' :      0.000444,
'sv' :      0.000384,
'ko' :      0.000384,
'de' :      0.000384,
'da' :      0.000360,
'hr' :      0.000288,
'af' :      0.000276,
'fr' :      0.000180,
'ro' :      0.000156,
'tr' :      0.000120,
'it' :      0.000120,
'tl' :      0.000096,
'vi' :      0.000096,
'sl' :      0.000084,
'cy' :      0.000060,
'sq' :      0.000048,
'fi' :      0.000036,
'unknown' : 0.000036,
'so' :      0.000036,
'th' :      0.000036,
'sw' :      0.000024,
'lt' :      0.000024,
'hu' :      0.000024,
'el' :      0.000012,
'sk' :      0.000012
}

lang_distrib_test = {
'ru' :      0.788430,
'en' :      0.152496,
'bg' :      0.013002,
'pl' :      0.009909,
'pt' :      0.006240,
'et' :      0.006168,
'ja' :      0.005862,
'uk' :      0.003776,
'ar' :      0.003273,
'es' :      0.002805,
'mk' :      0.001241,
'ca' :      0.000683,
'hi' :      0.000683,
'id' :      0.000665,
'ko' :      0.000557,
'nl' :      0.000486,
'de' :      0.000432,
'no' :      0.000396,
'da' :      0.000378,
'af' :      0.000342,
'hr' :      0.000342,
'fr' :      0.000324,
'sv' :      0.000288,
'fi' :      0.000198,
'tr' :      0.000180,
'tl' :      0.000162,
'th' :      0.000090,
'so' :      0.000090,
'it' :      0.000072,
'unknown' : 0.000072,
'sl' :      0.000072,
'cy' :      0.000072,
'sk' :      0.000054,
'ro' :      0.000036,
'el' :      0.000036,
'vi' :      0.000036,
'sq' :      0.000018,
'hu' :      0.000018,
'lt' :      0.000018
}

# Словарь stop_words_dict, содержащий языки, детектированные в датасете
stop_words_dict = {
    'ru': set(stopwords.words('russian')),  # Русский
    'en': set(stopwords.words('english')),  # Английский
    # 'es': set(stopwords.words('spanish')),  # Испанский
    # # 'bg': set(stopwords.words('bulgarian')),  # Болгарский
    # # 'pl': set(stopwords.words('polish')),  # Польский
    # 'pt': set(stopwords.words('portuguese')),  # Португальский
    # 'ar': set(stopwords.words('arabic')),  # Арабский
    # # 'uk': set(stopwords.words('ukrainian')),  # Украинский
    # 'ca': set(stopwords.words('catalan')),  # Каталанский
    # 'no': set(stopwords.words('norwegian')),  # Норвежский
    # # 'hi': set(stopwords.words('hindi')),  # Хинди
    # 'id': set(stopwords.words('indonesian')),  # Индонезийский
    # 'nl': set(stopwords.words('dutch')),  # Нидерландский
    # 'sv': set(stopwords.words('swedish')),  # Шведский
    # # 'ko': set(stopwords.words('korean')),  # Корейский
    # 'de': set(stopwords.words('german')),  # Немецкий
    # 'da': set(stopwords.words('danish')),  # Датский
    # # 'hr': set(stopwords.words('croatian')),  # Хорватский
    # # 'af': set(stopwords.words('afrikaans')),  # Африкаанс
    # 'fr': set(stopwords.words('french')),  # Французский
    # 'ro': set(stopwords.words('romanian')),  # Румынский
    # 'tr': set(stopwords.words('turkish')),  # Турецкий
    # 'it': set(stopwords.words('italian')),  # Итальянский
    # # 'tl': set(stopwords.words('tagalog')),  # Тагальский
    # # 'vi': set(stopwords.words('vietnamese')),  # Вьетнамский
    # # 'sl': set(stopwords.words('slovenian')),  # Словенский
    # 'fi': set(stopwords.words('finnish')),  # Финский
    # # 'lt': set(stopwords.words('lithuanian')),  # Литовский
    # 'hu': set(stopwords.words('hungarian')),  # Венгерский
    # 'el': set(stopwords.words('greek')),  # Греческий
    # # 'sk': set(stopwords.words('slovak')),  # Словацкий
}

# Языки, для которых нет встроенных стоп-слов в NLTK
# 'bulgarian', 'polish', 'ukrainian', 'hindi', 'korean', 'croatian'
# 'afrikaans', 'tagalog', 'vietnamese', 'slovenian', 'lithuanian', 'slovak'
# 'ja', 'et', 'mk', 'cy', 'so', 'th', 'sw'

# Словарь с регулярными выражениями для каждого языка
# в базовом варианте используем только русский и английский алфавиты
# как самые распространенные в датасете
language_characters = {
    'ru': r'[^а-яА-Я\s]',  # Только русские буквы
    'en': r'[^a-zA-Z\s]',  # Только латиница (английские буквы)
    # 'es': r'[^a-zA-ZñÑ\s]',  # Латиница и буква ñ для испанского
    # 'bg': r'[^а-яА-Я\s]',  # Болгарский (кириллица)
    # 'pl': r'[^а-яА-Яa-zA-ZąĄćĆęĘłŁńŃóÓśŚźŹżŻ\s]',  # Польский (с диакритикой)
    # 'pt': r'[^а-яА-Яa-zA-ZáÁâÂãÃàÀçÇéÉêÊíÍóÓôÔõÕúÚüÜ\s]',  # Португальский
    # 'ar': r'[^а-яА-Яa-zA-Z\u0600-\u06FF\s]',  # Арабский алфавит
    # 'uk': r'[^а-яА-Яa-zA-ZіїґєІЇҐЄ\s]',  # Украинский
    # 'ca': r'[^а-яА-Яa-zA-ZçÇ\s]',  # Каталанский
    # 'hi': r'[^а-яА-Яa-zA-Z\u0900-\u097F\s]',  # Хинди (деванагари)
    # 'id': r'[^а-яА-Яa-zA-Z\s]',  # Индонезийский (латиница)
    # 'nl': r'[^а-яА-Яa-zA-Z\s]',  # Нидерландский (латиница)
    # 'sv': r'[^а-яА-Яa-zA-ZåÅäÄöÖ\s]',  # Шведский (латиница с диакритикой)
    # 'ko': r'[^а-яА-Яa-zA-Z\uAC00-\uD7AF\s]',  # Корейский (хангыль)
    # 'de': r'[^а-яА-Яa-zA-ZäÄöÖüÜß\s]',  # Немецкий (с диакритикой)
    # 'da': r'[^а-яА-Яa-zA-ZæÆøØåÅ\s]',  # Датский (латиница с диакритикой)
    # 'fr': r'[^а-яА-Яa-zA-ZàÀâÂçÇéÉèÈêÊîÎôÔûÛ\s]',  # Французский
}

# Функция для очистки текста
def clean_text(text, lang):
    # Приводим к нижнему регистру
    text = str(text).lower()

    # Удаляем ссылки
    text = re.sub(r'http\S+|www\S+', '', text)

    # Оставляем только буквы, соответствующие алфавиту языка
    regex = language_characters.get(lang, r'[^a-zA-Zа-яА-Я\s]')  # Базовая очистка, если язык не найден
    text = re.sub(regex, '', text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Функция для удаления стоп-слов на основе языка
def remove_stopwords(text, lang):
    words = text.split()
    if lang in stop_words_dict:
        stop_words = stop_words_dict[lang]
        words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def clean_text_apply(df,col):
    col_name = 'clan'+col
    df[col_name] = df.apply(lambda row: clean_text(row[col], row['language_detected']), axis=1)

def remove_stopwords_apply(df,col):
    df[col] = df.apply(lambda row: remove_stopwords(row[col], row['language_detected']), axis=1)