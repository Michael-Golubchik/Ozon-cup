import pandas as pd
import numpy as np
import joblib
from baseline import (
            process_dfs,
            merge_data,
            p_time,
            get_predictions_in_batches,
            evaluate_predictions,
            add_features,
            prepare_data,
            get_list_to_train_bert
)

import time
import polars as pl
from catboost import CatBoostClassifier, Pool

from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import lil_matrix, hstack, csr_matrix
import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Вести ли дебаг
DEBUG = 1
#Если тестовый прогон
#IS_CHECK = True
IS_CHECK = False

notebook_starttime = time.time()

# Загрузает отдельно test
def load_test():
    val_path = './data/train/train.parquet' if IS_CHECK else './data/test/test.parquet'
    test = pd.read_parquet(val_path, engine='pyarrow')
    # ! Только для проверки не для сабмита реального
    if IS_CHECK:
        # num_rows = int(len(test) * 0.1)  # Вычисляем количество строк, соответствующее 1%
        # test = test.head(num_rows)  # Получаем первые num_rows строк
        test = test.sample(frac=0.01, random_state=42)
        #test = test.sample(frac=0.1, random_state=424)
        test.reset_index(drop=True, inplace=True)
    return test

def load_test_data():    
    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'

    test = load_test()
    
    variantids = pd.concat([test['variantid1'], test['variantid2']]).unique()
    
    #attributes = pd.read_parquet(attributes_path, engine='pyarrow')
    #resnet = pd.read_parquet(resnet_path, engine='pyarrow')
    #text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')

    #! print('text_and_bert')
    text_and_bert = (
        pl.scan_parquet(text_and_bert_path)
        .filter(pl.col("variantid").is_in(variantids))
        .collect()
    ).to_pandas()
    #! print('attributes')
    attributes = (
        pl.scan_parquet(attributes_path)
        .filter(pl.col("variantid").is_in(variantids))
        .collect()
    ).to_pandas()
    #!1 print('resnet')
    resnet = (
        pl.scan_parquet(resnet_path)
        .filter(pl.col("variantid").is_in(variantids))
        .collect()
    ).to_pandas()
    
    return attributes, resnet, text_and_bert, test


def main_new():
    if DEBUG: print(p_time(), 'Start old:')
    if DEBUG: print(p_time(), 'load all data:')
    attributes, resnet, text_and_bert, test = load_test_data()
    #! text_and_bert = process_text_and_bert(text_and_bert)
    attributes, resnet = process_dfs(attributes, resnet)

    #! test_data = merge_data(test, resnet, text_and_bert)
    if DEBUG: print(p_time(), 'Merge:')
    test_data = merge_data(attributes, test, resnet, text_and_bert)
    del attributes, resnet, text_and_bert
    if DEBUG: print(p_time(), 'Add Features:')
    vectorizers = joblib.load('vectorizers.pkl')
    # test_data = add_features(test_data, make_remarks = False)
    features, _ = add_features(test_data, vectorizers, make_remarks = False)
    if DEBUG: print(p_time(), 'End add Features:')
    # del attributes, resnet, text_and_bert

    # return test_data

    # tfidf_vectorizer = joblib.load('vectorizer.pkl')
    # text_data = test_data['text_1'] + ' ' + test_data['text_2']
    # text_embeddings = csr_matrix(tfidf_vectorizer.transform(text_data).toarray())
    # del text_data, tfidf_vectorizer

    #features = hstack([text_embeddings, features])
    # features = hstack([features])
    # vectorizers = joblib.load('vectorizers.pkl')
    features = prepare_data(test_data, features, vectorizers)
    
    # Test2
    #Lgbm
    models = joblib.load('models.pkl')
    # Вероятности предсказания моделй ансамбля
    y_pred_prob_sum = None
    for model in models:
        y_pred_prob = model.predict(features)
        if y_pred_prob_sum is None:
            y_pred_prob_sum = y_pred_prob
        else:
            y_pred_prob_sum += y_pred_prob
        if IS_CHECK:
            predictions = (y_pred_prob >= 0.5).astype(int)
            evaluate_predictions(test_data['target'].values, predictions, y_pred_prob)
    
    predictions_prob = y_pred_prob_sum / len(remove_fractions)

    # Catboost
    # test_pool = Pool(features)
    # predictions_prob = model.predict_proba(test_pool)[:, 1]
    predictions = (predictions_prob >= 0.5).astype(int)

    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        #'target': predictions
        'target': predictions_prob
    })
    if IS_CHECK:
        #evaluate_predictions(test_data['target'].values, y_pred, y_pred_prob)
        print('---------------------------------------------------')
        print('Метрики ансамбля:')
        evaluate_predictions(test_data['target'].values, predictions, predictions_prob)
        
    submission.to_csv('./data/submission.csv', index=False)
    if DEBUG: print(p_time(), 'End:')
    # return
    return predictions, predictions_prob


def main():
    if DEBUG: print(p_time(), 'Start old:')
    if DEBUG: print(p_time(), 'load all data:')
    attributes, resnet, text_and_bert, test = load_test_data()
    #! text_and_bert = process_text_and_bert(text_and_bert)
    attributes, resnet = process_dfs(attributes, resnet)

    #! test_data = merge_data(test, resnet, text_and_bert)
    if DEBUG: print(p_time(), 'Merge:')
    test_data = merge_data(attributes, test, resnet, text_and_bert)
    del attributes, resnet, text_and_bert
    if DEBUG: print(p_time(), 'Add Features:')
    vectorizers = joblib.load('vectorizers.pkl')
    # test_data = add_features(test_data, make_remarks = False)
    features, _ = add_features(test_data, vectorizers, make_remarks = False)
    if DEBUG: print(p_time(), 'End add Features:')
    # del attributes, resnet, text_and_bert

    # return test_data

    # tfidf_vectorizer = joblib.load('vectorizer.pkl')
    # text_data = test_data['text_1'] + ' ' + test_data['text_2']
    # text_embeddings = csr_matrix(tfidf_vectorizer.transform(text_data).toarray())
    # del text_data, tfidf_vectorizer

    #features = hstack([text_embeddings, features])
    # features = hstack([features])
    # vectorizers = joblib.load('vectorizers.pkl')
    features = prepare_data(test_data, features, vectorizers)
    
    # Test2
    #Lgbm
    model = joblib.load('lgbm.pkl')
    predictions_prob = model.predict(features)
    # Catboost
    # test_pool = Pool(features)
    # predictions_prob = model.predict_proba(test_pool)[:, 1]
    predictions = (predictions_prob >= 0.5).astype(int)

    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        #'target': predictions
        'target': predictions_prob
    })
    if IS_CHECK:
        #evaluate_predictions(test_data['target'].values, y_pred, y_pred_prob)
        evaluate_predictions(test_data['target'].values, predictions, predictions_prob)
        
    # submission.to_csv('./data/submission.csv', index=False)
    if DEBUG: print(p_time(), 'End:')
    # return
    # return predictions, predictions_prob
    
    del model, features

    # *********************************************
    if DEBUG: print(p_time(), 'Start new:')

    #tokenizer = joblib.load("tokenizer.pkl")
    tokenizer = BertTokenizer.from_pretrained('./rubert-tiny2')
    model = torch.load('bert.pkl')

    model.eval()
    model.to(device)

    if DEBUG: print(p_time(), 'Оценка модели:')
    # Получение предсказаний на тестовом наборе
    #test_data = test_data.sample(frac=0.01, random_state=42)

    test_data.reset_index(drop=True, inplace=True)
    train_texts = get_list_to_train_bert(test_data)
    
    # Устанавливаем размер batch_sz в 64, если графическая карта
    # и в 1, если cpu
    batch_sz = 64 if str(device) == 'cuda' else 1
    print('batch_size', batch_sz)
    y_pred, y_pred_prob = get_predictions_in_batches(model,
                                                     tokenizer,
                                                     train_texts,
                                                     batch_size=batch_sz)

    y_pred_prob = np.array(y_pred_prob)
    # Комбинируем предсказания LightGBM и Bert
    combined_prob = (predictions_prob*0.7 + y_pred_prob*0.3)
    combined_predictions = (combined_prob >= 0.5).astype(int)
    # return test_data['target'].values, predictions_prob, y_pred_prob
    # ! Только для проверки не для сабмита реального
    if IS_CHECK:
        #evaluate_predictions(test_data['target'].values, y_pred, y_pred_prob)
        evaluate_predictions(test_data['target'].values, combined_predictions, combined_prob)
    
    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        #'target': combined_predictions
        'target': combined_prob
    })
    
    # *********************************************
    
    submission.to_csv('./data/submission.csv', index=False)
    if DEBUG: print(p_time(), 'End:')
    print('dfg')
    return

if __name__ == "__main__":
    main()
