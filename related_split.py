# Для разделения данных на две не пересекающиеся выборки

import pandas as pd
import numpy as np
from tqdm import tqdm

def get_related_rows(df, indices):
    # Функция для рекурсивного поиска всех связанных строк
    visited = set()
    to_visit = set(indices)
    related_indices = set()

    while to_visit:
        current_idx = to_visit.pop()
        if current_idx in visited:
            continue
        visited.add(current_idx)
        related_indices.add(current_idx)

        # Получаем значения variantid1 и variantid2 текущей строки
        vid1, vid2 = df.loc[current_idx, ['variantid1', 'variantid2']]

        # Находим все строки, которые содержат vid1 или vid2
        related_rows = df[
            (df['variantid1'].isin([vid1, vid2])) | 
            (df['variantid2'].isin([vid1, vid2]))
        ].index

        # Добавляем эти строки для дальнейшего поиска связанных строк
        to_visit.update(related_rows)

    return list(related_indices)

# Обновленный вариант функции с корректным индексированием
def select_percent(df, valid_size=0.2):
    total_rows = len(df)
    target_size = int(valid_size * total_rows)
    selected_indices = set()

    old_len = 0
    with tqdm(total=target_size, desc="Processing", unit="rows") as pbar:
        while len(selected_indices) < target_size:
            # Выбираем случайную строку, которая еще не была выбрана
            # print('size:', len(selected_indices), 'target_size:', target_size)
            random_indices = np.random.choice(df.index.difference(list(selected_indices)), size=1000, replace=False)
            related_indices = get_related_rows(df, random_indices)
            
            # Добавляем только те строки, которые не превышают цель
            for idx in related_indices:
                selected_indices.add(idx)
            pbar.update(len(selected_indices) - old_len)
            old_len = len(selected_indices)
    
    # Преобразуем множество в список перед передачей в .loc
    return df.loc[list(selected_indices)]

def related_split_train_valid(df, valid_size=0.2):
    '''
    Используем ранее разработанную функцию для получения валидационного набора
    param: valid_size - доля в данных у валидационной выборки
    '''
    valid_data = select_percent(df, valid_size)
    
    # Остальные данные считаются тренировочными
    train_data = df.drop(valid_data.index)
    
    return train_data, valid_data

# Применим финальную версию функции к train_data
# selected_data_final = select_20_percent_final(train_data)