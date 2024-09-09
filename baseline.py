import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.sparse import lil_matrix, hstack, csr_matrix
import joblib
import json
import re
from collections import defaultdict
import optuna

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import logging
import time
from tqdm import tqdm
import Levenshtein

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

from related_split import related_split_train_valid

# Вести ли дебаг
DEBUG = 1
FRAC = 1
# Какую долю данных для тренировки использовать. Если 1, то целиком
# FRAC_TRAIN = 0.01
# Какую долю данных для тренировки использовать. Если 1, то целиком
# FRAC_VALID = 0.01
# Загружать полные начальные данные (attributes и прочее, нужно при смене долей в данных)
FULL_INIT_DATA = True
#FULL_INIT_DATA = False
# Максимальное число токенов в строке
#MAX_TOKENS=1024
MAX_TOKENS=1024
# Загружать ли ранее подготовленные данные?
LOAD_PREPAIRED = False
#LOAD_PREPAIRED = True
# Загружать сразу подготовленную обучающую выборку
LOAD_PREPAIRED_X = False
#LOAD_PREPAIRED_X = True
# Загружать ли подготовленные признаки
LOAD_FEATURES = False
#LOAD_FEATURES = True
# Делать ли новые векторайзеры
MAKE_NEW_TFIDF = True
#MAKE_NEW_TFIDF = False

# Для интеграции tqdm с pandas
tqdm.pandas()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

logging.disable(logging.WARNING)

notebook_starttime = time.time()

def make_attributes_name():
    '''
    Подготавливает имена атрибутов для поиска
    return: exact_attributes -  список с именами атрибутов
    return: patern_attributes -  список с шаблонами имен атрибутов
    '''
    # Чтение имен атрибутов
    with open('exact_attributes.txt', 'r', encoding='utf-8') as file:
        exact_attributes = [line.strip() for line in file]
    #Чтение патернов атрибутов
    with open('pattern_attributes.txt', 'r', encoding='utf-8') as file:
        pattern_attributes = [line.strip() for line in file]
    print('Начальная длина списка атрибутов:', len(exact_attributes))
    # Фильтрация списка exact_attributes. удаляем элементы. соответствующие шаблонам
    # exact_attributes = [
    #     attr for attr in exact_attributes
    #     if not any(re.match(pattern, attr, re.IGNORECASE) for pattern in pattern_attributes)
    # ]
    print('Длина списка атрибутов после фильтрации:', len(exact_attributes))
    if DEBUG:
        # Запись отфильтрованного списка на диск
        with open('filtered_attributes.txt', 'w', encoding='utf-8') as file:
            for attr in exact_attributes:
                file.write(attr + '\n')
    
    # Оставим только первые 500 значений имен атрибутов
    exact_attributes = exact_attributes[:500]
    print('Длина списка атрибутов после отсечения:', len(exact_attributes))
    return exact_attributes, pattern_attributes

exact_attributes, pattern_attributes = make_attributes_name()
# exact_attributes = []
# pattern_attributes = []

# Возвращает сколько уже работает ноутбук
def p_time():
    #
    run_time = round(time.time() - notebook_starttime)
    return str(run_time).zfill(5)+' sec:'

def load_data():
    attributes_path = './data/train/attributes.parquet'
    resnet_path = './data/train/resnet.parquet'
    text_and_bert_path = './data/train/text_and_bert.parquet'
    train_path = './data/train/train.parquet'

    if FULL_INIT_DATA:
        # Загружем исходные данные как были
        if DEBUG: print(p_time(), '1.2 Load full init data:')
        attributes = pd.read_parquet(attributes_path)
        resnet = pd.read_parquet(resnet_path)
        text_and_bert = pd.read_parquet(text_and_bert_path)
        train = pd.read_parquet(train_path)

        if FRAC < 1:
            # Удаляем часть записей из train
            train = train.sample(frac=FRAC, random_state=42)
            
        # Удаляем в загруженных данных те, на который нет ссылок из train
        remaining_variantids = pd.concat([train['variantid1'], train['variantid2']]).unique()
        
        # Фильтруем DataFrame attributes, resnet и text_and_bert на основе оставшихся variantid
        attributes = attributes[attributes['variantid'].isin(remaining_variantids)]
        resnet = resnet[resnet['variantid'].isin(remaining_variantids)]
        text_and_bert = text_and_bert[text_and_bert['variantid'].isin(remaining_variantids)]
        
        train.reset_index(drop=True, inplace=True)
        attributes.reset_index(drop=True, inplace=True)
        resnet.reset_index(drop=True, inplace=True)
        text_and_bert.reset_index(drop=True, inplace=True)

        if FRAC < 1:    
            joblib.dump(attributes, './data/attributes.pkl')
            joblib.dump(resnet, './data/resnet.pkl')
            joblib.dump(text_and_bert, './data/text_and_bert.pkl')
            joblib.dump(train, './data/train.pkl')
    else:
        if DEBUG: print(p_time(), '1.2 Load sampled init data:')
        attributes = joblib.load('./data/attributes.pkl')
        resnet = joblib.load('./data/resnet.pkl')
        text_and_bert = joblib.load('./data/text_and_bert.pkl')
        train = joblib.load('./data/train.pkl')

    return attributes, resnet, text_and_bert, train

def extract_text_from_row(row):

    category_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['categories'].values())]
    )
    attributes_text = ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['characteristic_attributes_mapping'].values())]
    )
    #!!! return f"{category_text} {attributes_text}"
    # return f"{category_text}"
    return f"{attributes_text}"

def process_attributes(df):
    df['categories'] = df['categories'].apply(json.loads)
    df['characteristic_attributes_mapping'] = df['characteristic_attributes_mapping'].apply(json.loads)
    df['combined_text'] = df.apply(extract_text_from_row, axis=1)
    
    return df

# Функция для объединения всех вложенных numpy.ndarray в один массив с сохранением структуры
def combine_pic_embeddings(row):
    # Извлечение массивов из колонок
    embed1 = row['main_pic_embeddings_resnet_v1']
    embed2 = row['pic_embeddings_resnet_v1']
    
    # Создаем пустой список для хранения всех вложенных массивов
    combined_embeds = []
    
    # Если поле не пустое, добавляем его вложенные массивы в общий список
    if isinstance(embed1, np.ndarray):
        combined_embeds.extend(embed1)  # Добавляем все элементы из embed1
    if isinstance(embed2, np.ndarray):
        combined_embeds.extend(embed2)  # Добавляем все элементы из embed2
    
    # Преобразуем список обратно в numpy массив
    return np.array(combined_embeds)

def process_dfs(attributes, resnet):
    attributes = process_attributes(attributes)
    # Применение функции к DataFrame и обновление колонки main_pic_embeddings_resnet_v1
    resnet['main_pic_embeddings_resnet_v1'] = resnet.apply(combine_pic_embeddings, axis=1)
    resnet = resnet.drop(columns=['pic_embeddings_resnet_v1'])

    return attributes, resnet

# Функция для вычисления всех необходимых метрик сравнекния картинок
def calculate_pics_metrics(row):
    embed1 = row['pic_embeddings_1']
    embed2 = row['pic_embeddings_2']
    
    # Список для хранения всех попарных косинусных расстояний
    cosine_distances = []
    
    # Итерация по всем парам вложенных массивов
    for vec1 in embed1:
        for vec2 in embed2:
            dist = 1-cosine(vec1, vec2)
            cosine_distances.append(dist)
    
    # Вычисление минимума, максимума и среднего
    # min_distance = np.min(cosine_distances)
    max_distance = np.max(cosine_distances)
    # avg_distance = np.mean(cosine_distances)
    
    # Усреднение поэлементно сложенных массивов
    # avg_embedding1 = np.mean(embed1, axis=0)
    # avg_embedding2 = np.mean(embed2, axis=0)
    
    # Вычисление косинусного расстояния между усредненными массивами
    # avg_distance_between_means = cosine(avg_embedding1, avg_embedding2)
    
    #return min_distance, max_distance, avg_distance, avg_distance_between_means
    #return min_distance, max_distance, avg_distance
    return max_distance


class Features:
    def __init__(self, row, vectorizers):
        self.row = row
        self.vectorizers = vectorizers
        # атрибуты первого товара
        self.attr_1 = row['characteristic_attributes_mapping_1']
        # атрибуты второго товара
        self.attr_2 = row['characteristic_attributes_mapping_2']

        self.attr_1 = self.rename_dop_artikul(self.attr_1)
        self.attr_2 = self.rename_dop_artikul(self.attr_2)
        
        # Ключи которые есть в обоимх атрибутах
        self.attr_1_and_2 = self.attr_1.keys() & self.attr_2.keys()
        # Список фич
        self.ret_value = []
        # Строка с заметками о генерации
        self.remark = ''
        self.clean_attributes()


    def clean_attributes(self):
        '''
        Удаляем атрибуты в которых вроде что-то написано но фактически нет информации
        '''
        # Разбиьарем ключ состав
        key = 'Состав'
        if key in self.attr_1_and_2:
            
            pattern = r'указан.{,7}упаковк'
            
            if ((re.search(pattern, self.attr_1[key][0]) and (len(self.attr_1[key][0]) < 100)) or
                (re.search(pattern, self.attr_2[key][0]) and (len(self.attr_2[key][0]) < 100))):
                # если есть короткое описание в одном из товаров и там только про то что состав
                # на упаковке то не чего и сравнивать.
                self.attr_1_and_2.remove(key)
                self.remark += f'\n\n   удалил сравнение : {key} потому что нет инфы'

    
    def rename_dop_artikul(self, dict):
        '''
        Переименовывает дополнительный артикул в основной, если нет основного артикула
        '''
        key_to_check = 'Партномер'
        key_to_check_dop = 'Партномер (артикул производителя)'
        if ((key_to_check not in dict)
            and (key_to_check_dop in dict)):
            # Есть дополнительный артикул, но нет основного.
            # Переименовываем дополнительный артикул в основгой
            dict[key_to_check] = dict.pop(key_to_check_dop)
            # print('renamed Партномер (артикул производителя)')
        return dict
        
    
    def append(self, value):
        """
        Добавляет значение в ret_value
        """
        self.ret_value.append(value)

    def extend(self, value):
        """
        Добавляет массив значений в ret_value
        """
        self.ret_value.extend(value)

    def get_description_similarity(self, text1, text2):
        # Преобразуем строки в TF-IDF векторы с помощью обученного tfidf_vectorizer
        if text1 == text2:
            return 1
        vectorizer = self.vectorizers.tfidf_description_vectorizer
        tfidf_vector1 = vectorizer.transform([text1])
        tfidf_vector2 = vectorizer.transform([text2])
        similarity =  cosine_similarity(tfidf_vector1, tfidf_vector2)[0][0]
        
        return similarity
    
    def get_category_similarity(self, text1, text2, category_level):
        # Преобразуем строки в TF-IDF векторы с помощью обученного tfidf_vectorizer
        if text1 == text2:
            return 1
        vectorizer = self.vectorizers.categories_vectorizers[category_level]
        tfidf_vector1 = vectorizer.transform([text1])
        tfidf_vector2 = vectorizer.transform([text2])
        similarity =  cosine_similarity(tfidf_vector1, tfidf_vector2)[0][0]
        
        return similarity
        # return(0)
        

    def round_to_nearest(self, value, targets=[0.0, 0.1, 0.2, 0.6, 0.9, 0.92, 0.97, 1]):
        '''
        Округляем к ближайшему числу из списка
        '''
        return min(targets, key=lambda x: abs(x - value))
    
    def extract_words_space(self, text):
        text = re.sub(r'\модель\b|\bеще что удалить\b', '', text)
        pattern = r'\b([А-ЯA-Z\d\s\-\/]+)\b'
        words = re.findall(pattern, text)
        
        # Фильтруем слова, чтобы оставить только те, в которых не менее 7 букв
        result = [re.sub(r'[\s\-]+', ' ', word).strip() for word in words
                  if (len(word.strip()) >= 6 and
                      len(re.findall(r'\d', word)) >= 1 and
                      len(re.findall(r'[А-ЯA-Z]', word)) >= 1
                     )]
        
        return result

    def extract_words(self, text):
        text = re.sub(r'\модель\b|\bеще что удалить\b', '', text)
        pattern = r'\b([А-ЯA-Z\d\-\/]+)\b'
        words = re.findall(pattern, text)
        
        # Фильтруем слова, чтобы оставить только те, в которых не менее 7 букв
        result = [re.sub(r'[\s\-]+', ' ', word).strip() for word in words
                  if (len(word.strip()) >= 6 and
                      len(re.findall(r'\d', word)) >= 1 and
                      len(re.findall(r'[А-ЯA-Z]', word)) >= 1
                     )]
        
        return result

    
    def find_common_words(self, list1, list2):
        # Преобразуем списки в множества и находим пересечение
        common_words = set(list1) & set(list2)
        return list(common_words)
    
    def get_texts_similarity(self, text1, text2):
        '''
        Возвращает сходство между текстами text1 и text2
        '''
        # Преобразование текстов в TF-IDF векторы
        # print('text1:', text1, 'text2:', text2)
        if text1 == text2:
            return 1
        match1 = re.search(r'^\d+$', text1)
        match2 = re.search(r'^\d+$', text2)
        if match1 and match2:
            # Если это числа возвращаем соотношение
            match1 = int(match1.group())
            match2 = int(match2.group())
            if max(match1,match2) != 0:
                # Соотношение цифр
                ratio = min(match1,match2)/max(match1,match2)
                # Если разница не велика возвращаем 1
                # ratio = 1 if ratio > 0.98 else ratio:
                ratio= self.round_to_nearest(ratio)
                    
                return(ratio)
            else:
                return(0)
        else:
            # Создание экземпляра TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r'\b\w+\b',  # Учитывать любые слова, включая цифры
                stop_words=None  # Не использовать встроенные стоп-слова
            )
            try:
                tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
                return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except ValueError as e:
                return 0
        # Вычисление косинусного сходства между двумя текстовыми векторами
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]



    def get_attributes_text(self, attr_value):
        '''
        Переводит значение атрибута в текст
        '''
        if isinstance(attr_value, list):
            return ' '.join(map(str, attr_value))
        else:
            return str(attr_value)
            
    
    def make_one_feature(self, feature_name):
        '''
        Добавляет одну специфичную фичу по названию
        param: feature_name - название фичи
        '''
        #if (feature_name in self.attr_1) and (feature_name in self.attr_2):
        if feature_name in self.attr_1_and_2:
            # атрибут есть в фичах обоих товаров
            text1 = self.get_attributes_text(self.attr_1[feature_name])
            text2 = self.get_attributes_text(self.attr_2[feature_name])
                                             
            max_len = max(len(text1),len(text2))
            min_len = min(len(text1),len(text2))
            self.append(1)
            # if min_len > 0:
            #     self.append(max_len/min_len)
            # else:
            #     self.append(1000)
            if self.attr_1[feature_name] == self.attr_2[feature_name]:
                self.append(1)
            else:
                #Если атрибуты дословно не равны, то вычисляем насколько они похожи
                self.append(self.get_texts_similarity(text1, text2))
                
            self.attr_1_and_2.remove(feature_name)
            # del self.attr_1[feature_name], self.attr_2[feature_name]
        else:
            # атрибута нет по крайней мере в одном товаре
            # и эти ключи не равны, так как их нет
            #self.ret_value.extend([0,0,0])
            self.ret_value.extend([0,0])



    def make_one_pattern_feature(self, feature_pattern):
        '''
        Добавляет одну специфичную фичу название которой можно найти
        с помощью регулярного выражения feature_pattern
        param: feature_pattern - патерн названия фичи
        '''
        # Поиск ключей, соответствующих регулярному выражению
        matching_keys = [key for key in self.attr_1_and_2 if re.search(feature_pattern, key)]
        if matching_keys:
            key = matching_keys[0]
            # атрибут есть в фичах обоих товаров
            text1 = self.get_attributes_text(self.attr_1[key])
            text2 = self.get_attributes_text(self.attr_2[key])
                                             
            max_len = max(len(text1),len(text2))
            min_len = min(len(text1),len(text2))+0.00001
            
            self.append(1)
            # if min_len > 0:
            #     self.append(max_len/min_len)
            # else:
            #     self.append(1000)
            
            if self.attr_1[key] == self.attr_2[key]:
                self.append(1)
            else:
                #Если атрибуты дословно не равны, то вычисляем насколько они похожи
                self.append(self.get_texts_similarity(text1, text2))
            self.attr_1_and_2.remove(key)
        else:
            # атрибута нет по крайней мере в одном товаре
            # и эти ключи не равны, так как их нет
            #self.ret_value.extend([0,0,0])
            self.ret_value.extend([0,0])

    def make_partnumber(self, key_to_check):
        # **********************        
        # int((key_to_check in frs.attr_1) and (key_to_check in frs.attr_2))
        is_artikul = int(key_to_check in self.attr_1_and_2)
        self.append(is_artikul)
        # if is_artikul:
        #     frs.append(int(key_to_check not in differing_values))
        # else:
        #     frs.append(0)
        if is_artikul:
            self.append(int(self.attr_1[key_to_check] == self.attr_2[key_to_check]))
            
            # Вычисляем Расстояние Левенштейна для указания разницы между значениями ключей
            if self.attr_1[key_to_check] != self.attr_2[key_to_check]:
                self.append(Levenshtein.distance(self.attr_1[key_to_check][0], self.attr_2[key_to_check][0]))
            else:
                self.append(0)
            self.attr_1_and_2.remove(key_to_check)
        else:
            self.extend([0, 0])

        # Вносим длину артукула
        if key_to_check in self.attr_1:
            self.append(len(self.attr_1[key_to_check][0]))
        else:
            self.append(0)
        if key_to_check in self.attr_2:
            self.append(len(self.attr_2[key_to_check][0]))
        else:
            self.append(0)

    def compare_partnumbers(self, text1, text2):
        '''
        Сравнивает партномера в двух текстах
        '''
        # Извлекаем разные партномера и названия товаров
        parts_in_name_1 = self.extract_words(text1)
        parts_in_name_2 = self.extract_words(text2)
        if len(parts_in_name_1) > 0 and len(parts_in_name_2) > 0:
            # Есть партномера в обоих названиях
            self.append(1)
            common_in_names = self.find_common_words(parts_in_name_1, parts_in_name_2)
            if len(common_in_names) > 0:
                self.append(1)
                self.remark += f'\n нашел общие партномеры в текстах: {common_in_names}\n'                
            else:
                self.append(0)
        else:
            self.extend([0,0])

        parts_in_name_1 = self.extract_words_space(text1)
        parts_in_name_2 = self.extract_words_space(text2)
        if len(parts_in_name_1) > 0 and len(parts_in_name_2) > 0:
            # Есть партномера в обоих названиях
            self.append(1)
            common_in_names = self.find_common_words(parts_in_name_1, parts_in_name_2)
            self.append(1)
            if len(common_in_names) > 0:
                self.remark += f'\n нашел общие партномеры с пробелами в текстах: {common_in_names}\n'                
        else:
            self.extend([0,0])


            

def make_features_for_row(row, vectorizers, categories_embeddings_dict):
    '''
    Добавляет фичи в строку
    param: строка датафрейма
    return: frs.ret_value - numpy ndarray с фичами
    return: frs.remark - строка с заметками о генерации фич
    '''
    # Создаем объект для обработки фич
    frs = Features(row, vectorizers)
    # Вычисляем косинусное сходство для имен товаров
    frs.append(1-cosine(row['name_bert_1'], row['name_bert_2']))
    
    # Вычисляем косинусное сходство для главных картинок
    frs.append(1-cosine(row['pic_embeddings_1'][0], row['pic_embeddings_2'][0]))
    
    # Вычисляем метрики сходства всех картинок
    #frs.extend(calculate_pics_metrics(row))
    frs.append(calculate_pics_metrics(row))

    # *******************
    # Вычисляем метрики сходства категорий
    # Список категорий, по которым будем вычислять расстояния
    categories = ['1', '2', '3', '4']
    for i, cat in enumerate(categories):
        frs.append(1-cosine(categories_embeddings_dict[row['categories_1'][cat]],
                                  categories_embeddings_dict[row['categories_2'][cat]]))
        
        # frs.append(frs.get_texts_similarity(row['categories_1'][cat],
        #                                     row['categories_2'][cat]))
        frs.append(frs.get_category_similarity(
            row['categories_1'][cat],
            row['categories_2'][cat],
            category_level = i))
    # *******************
        
        # frs.append(int(row['categories_1'][cat] == row['categories_2'][cat]))

    #############
    # Добавляем сравнение текстов
    
    # frs.append(frs.get_description_similarity(row['description_1'], row['description_2']))
    # frs.append(len(row['description_1']))
    # frs.append(len(row['description_2']))

    # Извлекаем разные партномера из названия товаров
    frs.compare_partnumbers(row['name_1'], row['name_2'])
    frs.compare_partnumbers(row['name_1'] + row['description_1'],
                             row['name_2'] + row['description_2'])
    
        
    # emb1 = get_bert_embedding(model, tokenizer, row['description_1'])
    # emb2 = get_bert_embedding(model, tokenizer, row['description_2'])
    # frs.append(1-cosine(emb1, emb2))
    # Добавляем минимальную длину текста
    # min_text_len = min(len(row['description_1']), len(row['description_2']))
    # frs.append(min_text_len)
    # # Добавляем соотношение длин текстов
    # if min_text_len > 0:
    #      frs.append(max(len(row['description_1']), len(row['description_2']))/min_text_len)
    # else:
    #     frs.append(0)
            
    # Добавляем минимальную длину текста
    
    # Сравним словари с атрибутами
    # Число ключей в словаре
    # frs.extend([len(frs.attr_1), len(frs.attr_2)])
    # Равны ли словари
    #!!! frs.append(int(frs.attr_1==frs.attr_2))
    # Различия в словаре
    differing_values = {key: (frs.attr_1[key], frs.attr_2[key]) for key in frs.attr_1.keys() & frs.attr_2.keys() if frs.attr_1[key] != frs.attr_2[key]}
    frs.remark += f'   различные ключи: {differing_values}'
    # frs.append(len(differing_values))
    same_values = {key: (frs.attr_1[key], frs.attr_2[key]) for key in frs.attr_1.keys() & frs.attr_2.keys() if frs.attr_1[key] == frs.attr_2[key]}
    # Совпадающие под вопросом. чуть уменьшили скор
    frs.remark += f'\n\n   совпадающие ключи: {same_values}'
    # frs.append(len(same_values))

    # Обработаем артикул. Есть ли он в обоих товарах и отличается ли
    # **********************
    frs.make_partnumber(key_to_check = 'Артикул')
    frs.make_partnumber(key_to_check = 'Партномер (артикул производителя)')
    frs.make_partnumber(key_to_check = 'Партномер')
    # **********************

    # Добавим названия атрибутов по патернам из файла pattern_attributes.txt
    for feature_pattern in pattern_attributes:
        frs.make_one_pattern_feature(feature_pattern)
    
    # Добавим названия атрибутов из файла exact_attributes.txt
    for feature_name in exact_attributes:
        frs.make_one_feature(feature_name)

    
    # Вносим длины оставшихся не обработанных ключей
    # frs.append(len(differing_values))
    # frs.append(len(same_values))

        
    return [frs.ret_value, frs.remark]


# Функция для создания эмбеддингов BERT
def get_bert_embedding(model, tokenizer, text):
    inputs = tokenizer(text,
                       return_tensors='pt',
                       padding=True,
                       clean_up_tokenization_spaces=True,
                       truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


# Функция для создания словаря эмбеддингов категорий
def get_categories_bert_embedding(df):
    # model_name = 'cointegrated/rubert-tiny2'
    model_name = './rubert-tiny2'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    # Создание множества для хранения всех уникальных значений категорий
    unique_values = set()
    
    # Проход по всем строкам DataFrame и сбор уникальных значений
    for index, row in df.iterrows():
        # Добавляем все значения категорий в множество уникальных значений
        unique_values.update(row['categories_1'].values(),
                            row['categories_2'].values())
    
    # Создание словаря эмбеддингов категорий, где ключи - уникальные значения, а значения - их эмбеддинги BERT
    categories_embeddings_dict = {value: get_bert_embedding(model, tokenizer, value) for value in unique_values}
    return categories_embeddings_dict

def add_features(df, vectorizers, make_remarks = False):
    '''
    Добавляет фичи в датафрейм
    param: make_remarks - делать ли примечания к добавлению фич
    '''
    # Создание словаря эмбеддингов категорий, где ключи - уникальные значения, а значения - их эмбеддинги BERT
    if DEBUG: print(p_time(), 'Get categories bert embedding:')
    categories_embeddings_dict = get_categories_bert_embedding(df)
    if DEBUG: print(p_time(), 'Adding features:')

    ret_list= make_features_for_row(df.iloc[0], vectorizers, categories_embeddings_dict)
    # Список списков фич
    #!features = np.zeros((df.shape[0], len(ret_list[0])))
    features = lil_matrix((df.shape[0], len(ret_list[0])))
    # Список строк заметок к фичам
    remarks = []
    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing rows", ncols=100, leave=True)):
        ret_list= make_features_for_row(row, vectorizers, categories_embeddings_dict)
        # Добавление новой строки с массивом фичей в numpy массив features
        # features[i] = np.array(ret_list[0])
        features[i, :] = ret_list[0]  # Используем формат LIL для эффективного изменения значений
        
        # Добавление строки в список remarks
        if make_remarks:
            remarks.append(ret_list[1])
        else:
            remarks.append('')

    # df['features'] = features
    # df['remark'] = remarks
    # return df
    return features.tocsr(), remarks

def merge_data(attributes, train, resnet, text_and_bert):
    # Подсоединяем данные из resnet
    train_data = train.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    train_data = train_data.drop(columns=['variantid'])

    text_and_bert['description'] = text_and_bert['description'].fillna(' ')
    # Подсоединяем данные из text_and_bert
    text_and_bert_columns_to_merge = ['variantid', 'description', 'name_bert_64', 'name']
    train_data = train_data.merge(text_and_bert[text_and_bert_columns_to_merge], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={
        'name_bert_64': 'name_bert_1',
        'description': 'description_1',
        'name': 'name_1',
    })
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.merge(text_and_bert[text_and_bert_columns_to_merge], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={
        'name_bert_64': 'name_bert_2',
        'description': 'description_2',
        'name': 'name_2',
    })
    train_data = train_data.drop(columns=['variantid'])
    
    # Подсоединяем данные из attributes
    attributes_columns_to_merge = ['variantid', 'combined_text', 'categories', 'characteristic_attributes_mapping']
    
    train_data = train_data.merge(attributes[attributes_columns_to_merge], left_on='variantid1', right_on='variantid', how='left')
    train_data = train_data.rename(columns={
        'combined_text': 'text_1',
        'categories': 'categories_1',
        'characteristic_attributes_mapping': 'characteristic_attributes_mapping_1'
    })
    train_data = train_data.drop(columns=['variantid'])
    
    train_data = train_data.merge(attributes[attributes_columns_to_merge], left_on='variantid2', right_on='variantid', how='left')
    train_data = train_data.rename(columns={
        'combined_text': 'text_2',
        'categories': 'categories_2',
        'characteristic_attributes_mapping': 'characteristic_attributes_mapping_2'
    })
    train_data = train_data.drop(columns=['variantid'])

    train_data = train_data.dropna()

    return train_data

def combine_embeddings(row):
    pic_embeddings = np.concatenate([row['pic_embeddings_1'][0], row['pic_embeddings_2'][0]])
    text_embeddings = np.concatenate([row['text_embedding_1'], row['text_embedding_2']])
    return np.concatenate([pic_embeddings, text_embeddings])


class Vectorizers:
    def __init__(self, train_data):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=3000)
        self.prepare_vectorizer(train_data)
        
        self.tfidf_description_vectorizer = TfidfVectorizer(max_features=9000)
        self.prepare_description_vectorizer(train_data)
    
        self.categories_vectorizers = []
        self.prepare_categories_vectorizers(train_data)

    def prepare_description_vectorizer(self, train_data):
        '''
        Тренирует и возвращает tfidf_vectorizer
        для описаний товаров
        параметр: train_data - обучающие данные
        '''
        text_1_series = pd.Series(train_data['description_1'])
        text_2_series = pd.Series(train_data['description_2'])
        # Конкатенация Series для создания удвоенного списка строк
        text_data = pd.concat([text_1_series, text_2_series], ignore_index=True)
        self.tfidf_description_vectorizer.fit(text_data)
        
        
    def prepare_vectorizer(self, train_data):
        '''
        Тренирует и возвращает tfidf_vectorizer
        параметр: train_data - обучающие данные
        '''
        if DEBUG: print(p_time(), 'Sum text_data:')
        text_data = train_data['text_1'] + ' ' + train_data['text_2']
        # text_1_series = pd.Series(train_data['text_1'])
        # text_2_series = pd.Series(train_data['text_2'])
        # Конкатенация Series для создания удвоенного списка строк
        # text_data = pd.concat([text_1_series, text_2_series], ignore_index=True)
        
        if DEBUG: print(p_time(), 'Fit tfidf_vectorizer:')
        #text_embeddings = self.tfidf_vectorizer.fit_transform(text_data).toarray()
        self.tfidf_vectorizer.fit(text_data)


    def prepare_categories_vectorizers(self, train_data):
        '''
        Делает векторайзеры для заголовков
        '''
        # Созадаем списки всех текстов категорий товаров
        categories_list = [[] for _ in range(4)]
        for i, (index, row) in enumerate(tqdm(train_data.iterrows(), total=len(train_data), desc="Processing rows", ncols=100, leave=True)):
            # Для каждой из 4=х категорий добавляем заголовки обоих товаров
            for j in range(1,5):
                categories_list[j-1].append(str(row[f'categories_1'][f'{j}']))
                categories_list[j-1].append(str(row[f'categories_2'][f'{j}']))
        
        # Создаем веторайзеры категорий
        for i in range(4):
            cur_tfidf_vectorizer = TfidfVectorizer(max_features=3000)
            cur_tfidf_vectorizer.fit(categories_list[i])
            self.categories_vectorizers.append(cur_tfidf_vectorizer)

    
def prepare_data(data, features, vectorizers):
    '''
    Подготавливает данные к обучению
    параметр: data - обучающие данные
    параметр: vectorizers - объект класса Vectorizers содержащий несколько tfidf_vectorizer
    возвращаем: X - фичи, y - метки
    '''
    if DEBUG: print(p_time(), 'Get text embeddings:')
    text_data = data['text_1'] + ' ' + data['text_2']
    text_embeddings = csr_matrix(vectorizers.tfidf_vectorizer.transform(text_data).toarray())
    del text_data
    
    if DEBUG: print(p_time(), 'Make X:')
    # y = data['target']
    # Сделаем tfidf векторы для заголовков
    # Списки всех названий категорий первого товара
    categories_1_list = [[] for _ in range(4)]
    # Списки всех названий категорий второго товара
    categories_2_list = [[] for _ in range(4)]
    for i, (index, row) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Processing rows", ncols=100, leave=True)):
        # Добавляем в списки названия категорий
        for j in range(1,5):
            categories_1_list[j-1].append(str(row[f'categories_1'][f'{j}']))
            categories_2_list[j-1].append(str(row[f'categories_2'][f'{j}']))
    # Список категорий эмбедингов первого товара
    catesegori_embeddings_1 = []
    # Список категорий эмбедингов второго товара
    catesegori_embeddings_2 = []
    for i in range(4):
        # эмбединг категории i первого товара
        tfidf_ones = csr_matrix(vectorizers.categories_vectorizers[i].transform(categories_1_list[i]).toarray())
        # Заменяем в idf значения для слов, если они есть на 1, так как нужно просто знать есть слово или нет
        tfidf_ones[tfidf_ones > 0] = 1
        catesegori_embeddings_1.append(tfidf_ones)
        # эмбединг категории i второго товара
        tfidf_ones = csr_matrix(vectorizers.categories_vectorizers[i].transform(categories_2_list[i]).toarray())
        tfidf_ones[tfidf_ones > 0] = 1
        catesegori_embeddings_2.append(tfidf_ones)
    return  (hstack([features,
                     # text_embeddings,
                     # catesegori_embeddings_1[0], #эмбединги категории 1 первого товара
                     # catesegori_embeddings_2[0], #эмбединги категории 1 второго товара
                     # catesegori_embeddings_1[1], #эмбединги категории 2 первого товара
                     # catesegori_embeddings_2[1], #эмбединги категории 2 второго товара
                     # catesegori_embeddings_1[2], #эмбединги категории 3 первого товара
                     # catesegori_embeddings_2[2], #эмбединги категории 3 второго товара
                     # catesegori_embeddings_1[3], #эмбединги категории 4 первого товара
                     # catesegori_embeddings_2[3], #эмбединги категории 4 второго товара
                  ]))


def load_features():
    '''
    Загружает сохранненные ранее подготовленные данные для обучения
    '''
    X_train = joblib.load('./data/X_train.pkl.gz')
    X_val = joblib.load('./data/X_val.pkl.gz')
    y_train = joblib.load('./data/y_train.pkl.gz')
    y_val = joblib.load('./data/y_val.pkl.gz')
    return X_train, X_val, y_train, y_val

def save_features(X_train, X_val, y_train, y_val):
    '''
    Cохраняет данные для обучения
    '''
    # Сохранение данных на диск
    joblib.dump(X_train, './data/X_train.pkl.gz', compress='gzip')
    joblib.dump(X_val, './data/X_val.pkl.gz', compress='gzip')
    joblib.dump(y_train, './data/y_train.pkl.gz', compress='gzip')
    joblib.dump(y_val, './data/y_val.pkl.gz', compress='gzip')

# Определите функцию для оптимизации
def objective(trial):
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        #'border_count': trial.suggest_int('border_count', 5, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
        'max_bin': trial.suggest_int('max_bin', 1, 255),
        'verbose': trial.suggest_categorical('verbose', [100])
    }

    model = CatBoostClassifier(**params, cat_features=[])
    model.fit(train_pool)
    
    y_pred_prob = model.predict_proba(val_pool)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    #score = accuracy_score(y_val, y_pred)
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    #return score
    return prauc

def find_params():
    '''
    Ищет гиперпараметры для моделе
    '''
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)  # Укажите количество испытаний (n_trials)

    print("Best parameters: ", study.best_params)
    print("Best score: ", study.best_value)
    joblib.dump(study, './data/study.pkl.gz', compress='gzip')
    
    # Обучаем модельна лучших параметрах
    train_pool = Pool(X_train, y_train)
    model = CatBoostClassifier(**best_params)
    model.fit(train_pool)
    joblib.dump(model, 'baseline.pkl')
    
    return model

# Определяем пользовательскую метрику PRAUC для LightGBM
def prauc_metric(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return 'PRAUC', pr_auc, True  # True означает, что чем больше метрика, тем лучше


def train_lgbm_model(X_train, y_train, X_val, y_val):

    model = lgb.LGBMClassifier(n_estimators=1000)

    # params = {
    #     'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    #     'objective': 'binary',  # Двухклассовая классификация
    #     #'objective': 'cross_entropy',  # Двухклассовая классификация
    #     #'metric': 'binary_logloss',  # Логарифмическая потеря
    #     #'metric': 'auc',
    #     #'metric': 'prauc',
    #     # 'is_unbalance': True, # Включает автоматическое взвешивание классов
    #     'num_leaves': 64,  # Количество листьев в каждом дереве
    #     'max_depth': -1,  # Без ограничения глубины
    #     'learning_rate': 0.02,  # Скорость обучения
    #     'feature_fraction': 0.7,  # Доля признаков, используемых при построении каждого дерева
    #     'bagging_fraction': 0.8,  # Доля данных для бэггинга
    #     'bagging_freq': 5,  # Частота бэггинга
    #     'min_data_in_leaf': 20,  # Минимальное количество данных в листе
    #     'lambda_l1': 0.1,  # L1-регуляризация
    #     'lambda_l2': 0.1,  # L2-регуляризация
    #     'max_bin': 255,  # Максимальное количество бинов для разбиения
    #     'scale_pos_weight': 2 # Веc позитивного класса
    #     #'verbose': -1,  # Отключить вывод информации о процессе обучения
    # }

    params = {
        # 'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        # 'objective': 'binary',  # Двухклассовая классификация
        # #'objective': 'cross_entropy',  # Двухклассовая классификация
        # #'metric': 'binary_logloss',  # Логарифмическая потеря
        # #'metric': 'auc',
        # #'metric': 'prauc',
        # # 'is_unbalance': True, # Включает автоматическое взвешивание классов
        # 'num_leaves': 64,  # Количество листьев в каждом дереве
        # 'max_depth': -1,  # Без ограничения глубины
        # 'learning_rate': 0.02,  # Скорость обучения
        # 'feature_fraction': 0.7,  # Доля признаков, используемых при построении каждого дерева
        # 'bagging_fraction': 0.8,  # Доля данных для бэггинга
        # 'bagging_freq': 5,  # Частота бэггинга
        # 'min_data_in_leaf': 20,  # Минимальное количество данных в листе
        # 'lambda_l1': 0.1,  # L1-регуляризация
        # 'lambda_l2': 0.1,  # L2-регуляризация
        # 'max_bin': 255,  # Максимальное количество бинов для разбиения
        # 'scale_pos_weight': 2 # Веc позитивного класса
        # #'verbose': -1,  # Отключить вывод информации о процессе обучения
    }

    # params = {'num_leaves': 77, 'max_depth': 7, 'learning_rate': 0.05876030546485946, 'num_boost_round': 3666, 'min_child_samples': 11, 'subsample': 0.6261357287421725, 'colsample_bytree': 0.9442874660665626, 'reg_alpha': 9.8158419058229, 'reg_lambda': 0.5466574387472323, 'bagging_fraction': 0.7379790155722563, 'bagging_freq': 5}

    # Подготовка данных для LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    # model = lgb.train(params,
    #                   train_data,
    #                   num_boost_round=9000,
    #                   valid_sets=[val_data],
    #                   feval=lambda preds, train_data: prauc_metric(train_data.get_label(), preds),
    #                   callbacks=[lgb.early_stopping(stopping_rounds=100),]
    #                  )
    model = lgb.train(params,
                      train_data,
                      num_boost_round=1000,
                      valid_sets=[val_data],
                      feval=lambda preds, train_data: prauc_metric(train_data.get_label(), preds),
                      callbacks=[lgb.early_stopping(stopping_rounds=100),]
                     )
    
    # model.fit(X_train, y_train)
    
    return model

#def train_model(X_train, y_train):
def train_catboost_model(X_train, y_train, X_val, y_val):
    train_pool = Pool(X_train, y_train)
    # Создаем Pool для валидационной выборки
    val_pool = Pool(data=X_val, label=y_val)
    
    model = CatBoostClassifier(
        iterations=10000,        # Начальное количество итераций
        depth=8,                # Глубина дерева
        learning_rate=0.03,     # Скорость обучения
        subsample=0.8,          # Размер подвыборки
        colsample_bylevel=0.7,  # Сэмплирование признаков
        l2_leaf_reg=5,          # Регуляризация
        loss_function='Logloss',# функция потерь для бинарной классификации
        eval_metric='PRAUC',      # Метрика для ранней остановки
        early_stopping_rounds=100, # Количество итераций без улучшения для остановки
        # task_type='GPU',
        thread_count=8,       # Устанавливаем количество потоков CPU
        verbose=100             # вывод информации о процессе обучения каждые 100 итераций
        #verbose=0             # вывод информации о процессе обучения каждые 100 итераций
    )
    # best_params = best_params = {'depth': 10,
    #                              'learning_rate': 0.1125896062415432,
    #                              'iterations': 507,
    #                              #'iterations': 20,
    #                              'l2_leaf_reg': 1.0832270978970537,
    #                              'bagging_temperature': 0.1250992900915905,
    #                              'scale_pos_weight': 6.727614915644361,
    #                              'random_strength': 5.485268141003309,
    #                              'max_bin': 130}
    # model = CatBoostClassifier(**best_params, verbose=100)

    # Обучаем модель
    model.fit(train_pool, eval_set=val_pool)
    # joblib.dump(model, 'catboost.pkl')
    return model


def evaluate_model(model, model_type, X_val, y_val):
    if model_type == 'lgb':
        y_pred_prob = model.predict(X_val)
        # print(type(y_pred_prob))
        # y_pred_prob = min(y_pred_prob + 0.2, 1)
        # y_pred_prob[(y_pred_prob > 0.3) & (y_pred_prob < 0.5)] += 0.2
        # y_pred_prob = min(y_pred_prob + 0.2, 0)
        print("ddddd")
    elif model_type == 'catboost':
        val_pool = Pool(X_val, y_val)
        y_pred_prob = model.predict_proba(val_pool)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    # prauc = auc(recall, precision)
    # print(f'PRAUC: {prauc}')

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision_metric = precision_score(y_val, y_pred)
    recall_metric = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision_metric}')
    print(f'Recall: {recall_metric}')
    print(f'ROC AUC: {roc_auc}')
    print(f'PRAUC: {prauc}')
    
    return y_pred, y_pred_prob

def train_val_split(train_data):
    '''
    Разбивает данные на train и valid
    параметр: train_data - тренировочные данные
    возвращает: train_data - тренировочные данные 
    возвращает: val_data - валидационные данные 
    '''
    # Разбиваем на трейн и валидацию:
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    #train_data, val_data = train_test_split(train_data, test_size=0.9, random_state=42)

    # Удалим из трейна те товары что есть в валидации (сравнения могут быть перекрестными)
    # Шаг 1: Собираем все значения из variantid1 и variantid2 в val_data
    val_variants = set(val_data['variantid1']).union(set(val_data['variantid2']))
    # Шаг 2: Фильтруем строки в train_data
    train_data = train_data[
        ~((train_data['variantid1'].isin(val_variants)) | (train_data['variantid2'].isin(val_variants)))
    ]

    # Сбрасываем индексы в датафреймах
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    return train_data, val_data

# Функция для получения предсказаний
def get_predictions_in_batches(model, tokenizer, val_texts, batch_size=64):    
    # Токенизация текстов
    inputs = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKENS)
    # Создание TensorDataset и DataLoader для обработки по батчам
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # Массивы для хранения предсказаний
    all_preds = []
    all_probs = []

    # Проход по всем батчам
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing", unit="batch"):
            input_ids, attention_mask, token_type_ids = [x.to(device) for x in batch]

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().detach().numpy())
            all_probs.extend(probs[:, 1].cpu().detach().numpy())

    return all_preds, all_probs

def train_val_split(train_data):
    '''
    Разбиваем на трейн и валидацию и чтобы не было пересечений по товарам в которые сравниваются
    в разных выборках
    параметр: train_data - тренировочные данные
    возвращает: train_data - тренировочные данные 
    возвращает: val_data - валидационные данные 
    '''
    
    train_data, val_data = related_split_train_valid(train_data, valid_size=0.2)
    #train_data, val_data = related_split_train_valid(train_data, valid_size=0.001)
    # train_data.reset_index(drop=True, inplace=True)
    # val_data.reset_index(drop=True, inplace=True)
    joblib.dump(train_data, './data/train_idx.pkl')
    joblib.dump(val_data, './data/val_idx.pkl')
    return train_data, val_data

def main_check():
    check_bert()

def make_one_model(X_train, X_val, y_train, y_val, remove_fraction):
    '''
    Делает одну модель для ансамбля
    '''

    if DEBUG: print(p_time(), '7. Train model:')
    X_train_balanced, y_train_balanced = adjust_class_balance(X_train,
                                                              y_train,
                                                              target=0,
                                                              remove_fraction=remove_fraction)
    
    model = train_lgbm_model(X_train_balanced, y_train_balanced, X_val, y_val)
    
    return model


def main_new():
    if not LOAD_PREPAIRED_X:
        if DEBUG: print(p_time(), '1. Start load data:')
        attributes, resnet, text_and_bert, train = load_data()
    
        if DEBUG: print(p_time(), '2. Process attributes:')
        attributes, resnet = process_dfs(attributes, resnet)
    
        if DEBUG: print(p_time(), '3. Merge data:')
        train_data_all = merge_data(attributes, train, resnet, text_and_bert)
        del attributes, train, resnet, text_and_bert
        
        if DEBUG: print(p_time(), '3.1 tfidf_vectorizer:')
        vectorizers = Vectorizers(train_data_all)
        joblib.dump(vectorizers, 'vectorizers.pkl')
        
        # Делаем дополнительные фичи
        if DEBUG: print(p_time(), '3.1 Add Features:')
        if not LOAD_FEATURES:
            features, train_data_all['remark'] = add_features(train_data_all, vectorizers, make_remarks = True)
            if DEBUG: print(p_time(), '3.2 Dump train_data_all:')
            joblib.dump(features, './data/features.pkl.gz', compress='gzip')
        else:
            features = joblib.load('./data/features.pkl.gz')
        
    
        if DEBUG: print(p_time(), '5. Split data:')
        train_data, val_data, train_features, val_features = train_test_split(
            train_data_all,
            features,
            test_size=0.04,
            random_state=42
        )
        # Предсказания моделей ансамбля
        # y_preds = []
        X_train = prepare_data(train_data, train_features, vectorizers)
        y_train = train_data['target']
        X_val = prepare_data(val_data, val_features, vectorizers)
        y_val = val_data['target']
        save_features(X_train, X_val, y_train, y_val)
    else:
        X_train, X_val, y_train, y_val = load_features()

    # Снижаем число меток 1 в ч раза для валиадции
    X_val, y_val = adjust_class_balance(X_val, y_val, target=1, remove_fraction=0.75)
    # Доля класса 0 который нужно удалить при обучении модели
    remove_fractions = [0.001,
                        0.002,
                        0.003,
                        0.015,
                        0.02,
                        0.017,
                       ]
    # remove_fractions = [0.01,
    #                    ]
    # Вероятности предсказания моделй ансамбля
    y_pred_prob_sum = None
    # Тренируем модели ансамбля
    # Модели ансамбля
    models = []
    if False:
    #if True:
        # Делаем модели
        for i, remove_fraction in enumerate(remove_fractions):
            print('******* train model:', i, '******* ')
            model = make_one_model(X_train, X_val, y_train, y_val, remove_fraction=remove_fraction)
            models.append(model)
            # joblib.dump(model, f'lgbm_{i}.pkl')
            y_pred, y_pred_prob = evaluate_model(model, 'lgb', X_val, y_val)
            if y_pred_prob_sum is None:
                y_pred_prob_sum = y_pred_prob
            else:
                y_pred_prob_sum += y_pred_prob
        joblib.dump(models, 'models.pkl')
    else:
        # Берем готовые модели
        models2 = joblib.load('models3.pkl')
        models1 = joblib.load('models4.pkl')
        models = models1 + models2
        joblib.dump(models, 'models.pkl')
        for model in models:
            y_pred, y_pred_prob = evaluate_model(model, 'lgb', X_val, y_val)
            if y_pred_prob_sum is None:
                y_pred_prob_sum = y_pred_prob
            else:
                y_pred_prob_sum += y_pred_prob
    
    y_pred_prob = y_pred_prob_sum / len(remove_fractions)
    y_pred = (y_pred_prob>= 0.5).astype(int)

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision_metric = precision_score(y_val, y_pred)
    recall_metric = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    print('---------------------------------------------------')
    print('Метрики ансамбля:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision_metric}')
    print(f'Recall: {recall_metric}')
    print(f'ROC AUC: {roc_auc}')
    print(f'PRAUC: {prauc}')
    

import numpy as np
from scipy.sparse import csr_matrix

def adjust_class_balance(X, y, target=1, remove_fraction=0.2):
    """
    Удаляет определенную долю строк с меткой 1, чтобы изменить баланс классов.
    
    :param X: разряженная матрица признаков (csr_matrix)
    :param y: метки классов (numpy array)
    :param remove_fraction: доля строк с меткой 1, которую нужно удалить (значение от 0 до 1)
    :param: target: каккой класс удялять
    :return: измененные X и y
    """
    # Находим индексы строк с меткой 1
    #indices_one = np.where(y == 1)[0]
    indices_one = np.where(y == target)[0]
    
    # Определяем количество строк, которые нужно удалить
    num_to_remove = int(len(indices_one) * remove_fraction)
    
    # Случайным образом выбираем строки для удаления
    np.random.seed(42)  # для воспроизводимости
    indices_to_remove = np.random.choice(indices_one, num_to_remove, replace=False)
    
    # Удаляем выбранные строки из X и y
    mask = np.ones(len(y), dtype=bool)
    mask[indices_to_remove] = False
    
    X_balanced = X[mask]
    y_balanced = y[mask]
    
    return X_balanced, y_balanced


def main():
    if DEBUG: print(p_time(), '1. Start load data:')
    # Делать ли индексы для val и train
    make_idx = False
    #make_idx = True
    # Загружать ли ранее подготовленные данные?
    if not LOAD_PREPAIRED_X:
        if not LOAD_PREPAIRED:
            # Если делаем все признаки фром скратч
            attributes, resnet, text_and_bert, train = load_data()
            
            if DEBUG: print(p_time(), '2. Process attributes:')
            #! text_and_bert = process_attributes(text_and_bert)
            attributes, resnet = process_dfs(attributes, resnet)
        
            if DEBUG: print(p_time(), '3. Merge data:')
            train_data_all = merge_data(attributes, train, resnet, text_and_bert)
            del attributes, train, resnet, text_and_bert

            # Предварительно разбиваем для векторайзера
            train_data, val_data = train_test_split(
                train_data_all,
                #test_size=0.2,
                #test_size=0.02,
                test_size=0.1,
                random_state=42
            )

            if DEBUG: print(p_time(), '3.1 tfidf_vectorizer:')
            if MAKE_NEW_TFIDF:
                vectorizers = Vectorizers(train_data)
                joblib.dump(vectorizers, 'vectorizers.pkl')
            else:
                # tfidf_vectorizer = joblib.load('vectorizer.pkl')
                vectorizers = joblib.load('vectorizers.pkl')
            
            # Делаем дополнительные фичи
            if DEBUG: print(p_time(), '3.1 Add Features:')
            # train_data_all = add_features(train_data_all, make_remarks = True)
            if not LOAD_FEATURES:
                features, train_data_all['remark'] = add_features(train_data_all, vectorizers, make_remarks = True)
                if DEBUG: print(p_time(), '3.2 Dump train_data_all:')
                joblib.dump(features, './data/features.pkl.gz', compress='gzip')
            else:
                features = joblib.load('./data/features.pkl.gz')
            
            if DEBUG: print(p_time(), '4. Split data:')
            train_data, val_data, train_features, val_features = train_test_split(
                train_data_all,
                features,
                #test_size=0.2,
                test_size=0.1,
                #test_size=0.02,
                random_state=42
            )
            del train_data_all, features
            print(type(train_features))
            if DEBUG: print(p_time(), '4.1 Dump train_data and val_data:')
            #!!! joblib.dump(train_data, './data/train_data.pkl', compress='gzip')
            #!!! joblib.dump(val_data, './data/val_data.pkl', compress='gzip')
            joblib.dump(train_features, './data/train_features.pkl', compress='gzip')
            joblib.dump(val_features, './data/val_features.pkl', compress='gzip')
        else:
            # Загружаем ранее подготовленные признаки
            if DEBUG: print(p_time(), 'Load prepaired train_data and val_data:')
            train_data = joblib.load('./data/train_data.pkl')
            val_data = joblib.load('./data/val_data.pkl')
            train_features = joblib.load('./data/train_features.pkl')
            val_features = joblib.load('./data/val_features.pkl')
        
        if DEBUG: print(p_time(), '6. Prepair data:')
        X_train = prepare_data(train_data, train_features, vectorizers)
        y_train = train_data['target']
        X_val = prepare_data(val_data, val_features, vectorizers)
        y_val = val_data['target']
        if DEBUG:
            save_features(X_train, X_val, y_train, y_val)
            pass
    else:
        val_data = joblib.load('./data/val_data.pkl')
        X_train, X_val, y_train, y_val = load_features()

    X_val, y_val = adjust_class_balance(X_val, y_val, target=1, remove_fraction=0.75)
    X_train, y_train = adjust_class_balance(X_train, y_train, target=1, remove_fraction=0.75)
    
    if DEBUG: print(p_time(), '7. Train model:')
    # model = train_catboost_model(X_train, y_train, X_val, y_val)
    # joblib.dump(model, 'catboost.pkl')
    model = train_lgbm_model(X_train, y_train, X_val, y_val)
    joblib.dump(model, 'lgbm.pkl')
    # model = find_params()
    if DEBUG: print(p_time(), '8. Evaluate model:')

    # y_pred, y_pred_prob = evaluate_model(model, 'catboost', X_val, y_val)
    y_pred, y_pred_prob = evaluate_model(model, 'lgb', X_val, y_val)

    # train_bert()
    # Сохраняем валидайионные данные вместе со сделанными предсказаниями

    # val_data['y_pred'] = y_pred
    # val_data['y_pred_prob'] = y_pred_prob
    # joblib.dump(val_data, './data/result_data.pkl')
    
    # return val_data

def repeat():
    '''
    Делает обучение модели по уже подготовленным данным
    '''
    tfidf_vectorizer = joblib.load('vectorizer.pkl')
    X_train, X_val, y_train, y_val = load_features()

    if DEBUG: print(p_time(), '1. Train model:')
    model = train_model(X_train, y_train)
    if DEBUG: print(p_time(), '2. Evaluate model:')
    evaluate_model(model, X_val, y_val)

# Вычисление метрик
def evaluate_predictions(y_val, y_pred, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    prauc = auc(recall, precision)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision_metric = precision_score(y_val, y_pred)
    recall_metric = recall_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision_metric}')
    print(f'Recall: {recall_metric}')
    print(f'ROC AUC: {roc_auc}')
    print(f'PRAUC: {prauc}')


def compute_metrics(eval_pred):
    '''
    Вычисляет метрики
    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Вычисление метрик
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    roc_auc = roc_auc_score(labels, logits[:, 1])  # Используем вероятности второго класса (положительный класс)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

class ProductDataset(Dataset):
    #def __init__(self, texts1, texts2, labels, tokenizer, max_len):
    #def __init__(self, texts1, texts2, labels, tokenizer, max_len):
    def __init__(self, texts1, labels, tokenizer, max_len):
        self.texts1 = texts1
        #self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text1 = str(self.texts1[item])
        #text2 = str(self.texts2[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text1,
            #text2,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            #return_overflowing_tokens=True,  # Вернуть обрезанные токены
            #truncation_strategy='only_first',  # Усекаем только первую последовательность
            return_tensors='pt'
        )

        #if 'overflowing_tokens' in encoding:
        #    print(f"Обрезанные токены: {encoding['overflowing_tokens']}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_list_to_train_bert(df):
    '''
    Создает текстовы для обучения bert
    :param: df: датафрейм с исходными данными
    :return: ret_list списокс с текстами для обучения. В каждом тексте информация
    о двух сравниваемых товарых
    '''
    ret_list = []
    for i, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing rows", ncols=100, leave=True)):
        # Атрибуты товаров
        attr_1 = row['characteristic_attributes_mapping_1']
        attr_2 = row['characteristic_attributes_mapping_2']
        # Текст сравнения товаров
        compare_text = ''
        compare_text += f"Название первого: {row['name_1']}:\n"
        compare_text += f"Название второго: {row['name_2']}:\n"

        compare_text += f"Категории первого: {row['categories_1']}:\n"
        compare_text += f"Категории второго: {row['categories_2']}:\n"

        differing_values = {key: (attr_1[key], attr_2[key]) for key in attr_1.keys() & attr_2.keys() if attr_1[key] != attr_2[key]}
        compare_text += f'различные ключи: {differing_values}\n'
        same_values = {key: (attr_1[key], attr_2[key]) for key in attr_1.keys() & attr_2.keys() if attr_1[key] == attr_2[key]}
        # Совпадающие под вопросом. чуть уменьшили скор
        compare_text += f'совпадающие ключи: {same_values}\n'
        
        # compare_text += f"Описание первого: {row['description_1']}:\n"
        # compare_text += f"Описание второго: {row['description_2']}:\n"

        # # Запись ключей и значений из characteristic_attributes_mapping_1
        # compare_text += f"Ключи первого:\n"
        # for key, value in attr_1.items():
        #     compare_text += f"{key}: {', '.join(value)}\n"
        # compare_text += f"Ключи второго:\n"
        # for key, value in attr_2.items():
        #     compare_text += f"{key}: {', '.join(value)}\n"
        
        if i == 0:
            print(compare_text)
        ret_list.append(compare_text)
        
    return ret_list


def train_bert():
    if DEBUG: print(p_time(), '1. Start load data:')
    if True:
    #if False:
        # Если делаем данные из исходных таблиц
        attributes, resnet, text_and_bert, train = load_data()
            
        if DEBUG: print(p_time(), '2. Process attributes:')
        #! text_and_bert = process_attributes(text_and_bert)
        attributes = process_attributes(attributes)
        # attributes, resnet = process_dfs(attributes, resnet)
    
        if DEBUG: print(p_time(), '3. Merge data:')
        train_data = merge_data(attributes, train, resnet, text_and_bert)
        del attributes, train, resnet, text_and_bert
        # joblib.dump(train_data, './data/train_data.pkl')
    
        if DEBUG: print(p_time(), '4. Split data:')

        train_data, val_data, train_labels, val_labels= train_test_split(
            train_data,
            train_data['target'].values,
            test_size=0.2,
            #test_size=0.02,
            #test_size=0.5,
            random_state=42
        )
        if DEBUG: print(p_time(), '5. Get texts to train bert:')
        train_texts = get_list_to_train_bert(train_data)
        val_texts = get_list_to_train_bert(val_data)
        
        joblib.dump(train_texts, './data/train_texts.pkl')
        joblib.dump(val_texts, './data/val_texts.pkl')
        joblib.dump(train_labels, './data/train_labels.pkl')
        joblib.dump(val_labels, './data/val_labels.pkl')
        
        del train_data, val_data
    else:
        train_texts = joblib.load('./data/train_texts.pkl')
        val_texts = joblib.load('./data/val_texts.pkl')
        train_labels = joblib.load('./data/train_labels.pkl')
        val_labels = joblib.load('./data/val_labels.pkl')
    
    # if DEBUG: print(p_time(), '5. tokenizer:')
    #tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('./rubert-tiny2')
    if DEBUG: print(p_time(), '6. Datasets:')
    # train_dataset = ProductDataset(train_texts1, train_texts2, train_labels, tokenizer, max_len=MAX_TOKENS)
    # val_dataset = ProductDataset(val_texts1, val_texts2, val_labels, tokenizer, max_len=MAX_TOKENS)
    # train_dataset = ProductDataset(train_texts1+train_texts2, train_labels, tokenizer, max_len=MAX_TOKENS)
    # val_dataset = ProductDataset(val_texts1+val_texts2, val_labels, tokenizer, max_len=MAX_TOKENS)
    train_dataset = ProductDataset(train_texts, train_labels, tokenizer, max_len=MAX_TOKENS)
    val_dataset = ProductDataset(val_texts, val_labels, tokenizer, max_len=MAX_TOKENS)
    if DEBUG: print(p_time(), '7. Настройка модели и тренера:')
    #model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=2)
    model = BertForSequenceClassification.from_pretrained('./rubert-tiny2', num_labels=2)
    
    model.to(device)
    # Замораживаем все параметры, кроме классификатора
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # for param in model.bert.embeddings.parameters():
    #     param.requires_grad = False

    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    if DEBUG: print(p_time(), '8. Обучение модели:')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics  # Передаем функцию вычисления метрик
    )
    # Обучение модели
    trainer.train()
    if DEBUG: print(p_time(), '8. Оценка модели:')
    eval_result = trainer.evaluate()
    print(eval_result)
    if DEBUG: print(p_time(), '9. End')


# Токенизация данных и создание тензоров
def tokenize_texts(tokenizer, texts1, texts2):
    return tokenizer(texts1, texts2, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKENS)


def check_bert():
    val_texts = joblib.load('./data/val_texts.pkl')
    val_labels = joblib.load('./data/val_labels.pkl')
    
    # Загрузка токенизатора из файла
    if DEBUG: print(p_time(), '5. Загрузка модели:')
    # tokenizer = joblib.load("tokenizer.pkl")
    tokenizer = BertTokenizer.from_pretrained('./rubert-tiny2')
    model = torch.load ('./bert.pkl')

    # checkpoint_path = "./old/checkpoint-4000/"
    # # Загрузка модели из checkpoint
    # model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    # # Загрузка токенизатора
    # tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    
    model.eval()
    model.to(device)
    
    if DEBUG: print(p_time(), '6. Оценка модели:')
    # Получение предсказаний на валидационном наборе
    y_pred, y_pred_prob = get_predictions_in_batches(model, tokenizer, val_texts, batch_size=64)
    # Вызов функции для вычисления и отображения метрик
    evaluate_predictions(val_labels, y_pred, y_pred_prob)

    if DEBUG: print(p_time(), '7. End')


def save_trained_model():
    '''
    Сохраняет обученную модель из checkpoint
    '''
    # Укажите путь к директории с checkpoint
    #checkpoint_path = "./results/checkpoint-4000/"
    checkpoint_path = "./results/checkpoint-17000"
    
    # Загрузка модели из checkpoint
    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    torch.save(model, 'bert.pkl')
    
    # Загрузка токенизатора
    # tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    # tokenizer.save_pretrained('./rubert_tokenizer')
    # joblib.dump(tokenizer, "tokenizer.pkl")


# Делаем отчет о работе валидатора
def make_report():
    '''
    Создает отчет о валидации и системе генерации фич в текстовый файл out.txt
    '''
    result_data = joblib.load('./data/result_data.pkl')
    result_data.reset_index(drop=True, inplace=True)
    val_features = joblib.load('./data/val_features.pkl')
    val_features = val_features.toarray()
    # Запись в текстовый файл
    with open('output.txt', 'w', encoding='utf-8') as file:
        # Ограничение на количество итераций
        max_iterations = 100
        result_data = result_data.head(max_iterations)
        #for index, row in tqdm(enumerate(result_data.iterrows()), total=min(len(result_data), max_iterations)):
        for index, row in tqdm(result_data.iterrows()):
            # if (val_features[index][12] < 1 and
            #     val_features[index][14] < 1
            #    ):
            #     continue
            # else:
            #     # Если нет совпадающий партномеров в названии
            #     if row['target'] == 1:
            #         continue
            # if (val_features[index][12] < 1 and
            #     val_features[index][14] < 1
            #    ):
            #     # Если нет совпадающий партномеров в названии
            #     pass
            #     if row['target'] == 1:
            #         continue
            # Если нагенерировали для тысячи товаров то и хватит
            file.write(f"Row: {index}:, Target: {row['target']}, y_pred: {row['y_pred']}, y_pred_prob: {row['y_pred_prob']}:")
            if row['target'] == row['y_pred']:
                file.write(f"  Good\n")
            else:
                file.write(f"  Mistake!!!\n")
            file.write(f"Name_1: {row['name_1']}:\n\n")
            file.write(f"Description_1:\n {row['description_1']}:\n\n")
            file.write(f"Categories_1: {row['categories_1']}:\n")
            # Запись ключей и значений из characteristic_attributes_mapping_1
            if pd.notnull(row['characteristic_attributes_mapping_1']):
                file.write("characteristic_attributes_mapping_1:\n")
                for key, value in row['characteristic_attributes_mapping_1'].items():
                    file.write(f"  * {key}: {', '.join(value)}\n")
            
            file.write(f"+++++:\n")
            file.write(f"Name_2: {row['name_2']}:\n\n")
            file.write(f"Description_2:\n {row['description_2']}:\n\n")
            file.write(f"Categories_2: {row['categories_2']}:\n")
            # Запись ключей и значений из characteristic_attributes_mapping_2
            if pd.notnull(row['characteristic_attributes_mapping_2']):
                file.write("characteristic_attributes_mapping_2:\n")
                for key, value in row['characteristic_attributes_mapping_2'].items():
                    file.write(f"  * {key}: {', '.join(value)}\n")
    
            feature_names = [
                'Сходство эмбеддингов имен',
                'Сходство эмбеддингов главных картинок',
                #'Минимальное сходство всех картинок',
                'Максимальное сходство всех картинок',
                #'Среднее сходство всех картинок',
                'Сходство в первой категории',
                'Сходство во второй категории',
                'Сходство в третьей категории',
                'Сходство в четвертой категории',
                'Сходство tfidf в первой категории',
                'Сходство tfidf во второй категории',
                'Сходство tfidf в третьей категории',
                'Сходство tfidf в четвертой категории',
                # 'Сходство описаний товара',
                # 'Длина описаний товара_1',
                # 'Длина описаний товара_2',
                'Есть ли партномера в обоих названиях',
                'Есть ли партномера совпадающие партномера в названиях',
                'Есть ли партномера с пробелами в обоих названиях',
                'Есть ли партномера с пробелами  совпадающие партномера в названиях',
                'Есть ли партномера в обоих названиях+описания',
                'Есть ли партномера совпадающие партномера в названиях+описания',
                'Есть ли партномера с пробелами в обоих названиях+описания',
                'Есть ли партномера с пробелами  совпадающие партномера в названиях+описания',
                # 'Число атрибутов_1',
                # 'Число атрибутов_2',
                # 'Совпадают ли атрибуты_1 и атрибуты_2 полностью',
                # 'Число различающихся атрибутов',
                # 'Число совпадающих атрибутов',
                'Есть ли артикул в обоих товарах',
                'Артикул совпадает в товарах',
                'Длина артикула_1',
                'Длина артикула_2',
                'Расстояние Левенштейна для разных артикулов',
                'partnumber_1. Есть ли артикул в обоих товарах',
                'partnumber_1. Артикул совпадает в товарах',
                'partnumber_1. Длина артикула_1',
                'partnumber_1. Длина артикула_2',
                'partnumber_1. Расстояние Левенштейна для разных артикулов',
                'partnumber_2. Есть ли артикул в обоих товарах',
                'partnumber_2. Артикул совпадает в товарах',
                'partnumber_2. Длина артикула_1',
                'partnumber_2. Длина артикула_2',
                'partnumber_2. Расстояние Левенштейна для разных артикулов',
            ]
            # Добавляем атрибуты из pattern_attributes
            for feature_name in pattern_attributes:
                feature_names.append(f'*{feature_name} есть')
                #feature_names.append(f'*{feature_name} соотношение длин')
                feature_names.append(f'*{feature_name} совпадают')
            
            # Добавляем атрибуты из exact_attributes    
            for feature_name in exact_attributes:
                feature_names.append(f'*{feature_name} есть')
                #feature_names.append(f'*{feature_name} соотношение длин')
                feature_names.append(f'*{feature_name} совпадают')
            # feature_names.extend(['Число оставшихся различающихся атрибутов',
            #                       'Число оставшихся совпадающих атрибутов'])
            file.write(f"\nФичи:\n\n")
            # Печатать ли вторую строку
            print_next = False
            # что дальше идет первая строка атрибута
            is_first = True
            # что дальше идет вторая строка атрибута
            is_second = False
            is_third = False
            # for i, feature in enumerate(row['features']):
            for i, feature in enumerate(val_features[index]):
                #print(i, feature_names[i])
                if feature_names[i].startswith('*'):
                    if is_first:
                        # Первую запись о том что метка вообще есть пропускаем
                        is_first = False
                        is_second = True
                        is_third = False
                        print_next = feature
                    elif is_second:
                        is_first = False
                        is_second = False
                        is_third = True
                        if print_next:
                            file.write(f"  * {i} {feature_names[i]}: {feature}\n")
                    elif is_third:
                        is_first = True
                        is_second = False
                        is_third = False
                        if print_next:
                            file.write(f"  * {i} {feature_names[i]}: {feature}\n")
                else:
                    file.write(f"  * {i} {feature_names[i]}: {feature}\n") 
            file.write("\n----------------\n\n") 
            file.write(f"\n{row['remark']}\n")
            file.write("\n**************************************************************************\n\n") 

def make_exact_attributes_from_train():
    # Создание словаря именований атрибутов
    key_frequency = defaultdict(int)
    
    attributes_path = './data/train/attributes.parquet'
    train_path = './data/train/train.parquet'

    print(p_time(), 'Load data:')
    attributes = pd.read_parquet(attributes_path)
    train = pd.read_parquet(train_path)

    train_variantids = pd.concat([train['variantid1'], train['variantid2']]).unique()
    print('init attributes shape', attributes.shape)
    # Фильтруем DataFrame attributes на основе оставшихся variantid
    attributes_in_train = attributes[attributes['variantid'].isin(train_variantids)]
    del attributes
    
    print('train attributes shape', attributes_in_train.shape)
    print(p_time(), 'Apply json:')
    attributes_in_train['characteristic_attributes_mapping'] = attributes_in_train['characteristic_attributes_mapping'].apply(json.loads)
    print(p_time(), 'Extract atributes:')
    with open('exact_attributes.txt', 'w', encoding='utf-8') as file:
        # Проход по всем строкам в val_data и подсчёт ключей
        for i, row in tqdm(attributes_in_train.iterrows()):
            # Учитываем ключи из первой колонки
            if pd.notnull(row['characteristic_attributes_mapping']):
                for key in row['characteristic_attributes_mapping'].keys():
                    key_frequency[key] += 1
        
        # Сортировка ключей по частоте встречаемости в порядке убывания
        sorted_key_frequency = sorted(key_frequency.items(), key=lambda item: item[1], reverse=True)
        
        # Вывод результатов
        print(f"Количество уникальных ключей: {len(sorted_key_frequency)}")
        
        # Вывод результатов
        # print("Ключи отсортированы по убыванию встречаемости:")
        # for key, freq in sorted_key_frequency:
        #     print(f"{key}: {freq} раз(а)")
        
        for key, freq in sorted_key_frequency:
            file.write(f"{key}\n")

if __name__ == "__main__":
    main()
