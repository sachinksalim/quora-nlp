import numpy as np
import pandas as pd

# Read Data

data_dir = 'data/'
df_train = pd.read_csv(data_dir + '_train.csv', encoding='utf-8', nrows = 5010)
df_train['id'] = df_train['id'].apply(str)

# df_test = pd.read_csv(data_dir + 'test.csv', encoding='utf-8', nrows = 3010)
# df_test['id'] = df_test['id'].apply(str)

# df_all = pd.concat((df_train, df_test))

df_all = df_train

df_all['question1'].fillna('', inplace=True)
df_all['question2'].fillna('', inplace=True)


# Create Vocab

from sklearn.feature_extraction.text import CountVectorizer
import itertools

counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))
other_index = len(counts_vectorizer.vocabulary_)


# Prepare Data

import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

words_tokenizer = re.compile(counts_vectorizer.token_pattern)

def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: 
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)

df_all = df_all.sample(2000) # Just for debugging

X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
    train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']), 
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.3, random_state=1989)


# Training

import keras.layers as lyr
from keras.models import Model

input1_tensor = lyr.Input(X1_train.shape[1:])
input2_tensor = lyr.Input(X2_train.shape[1:])

words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
seq_embedding_layer = lyr.LSTM(256, activation='tanh')

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

merge_layer = lyr.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

dense1_layer = lyr.Dense(16, activation='sigmoid')(merge_layer)
ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

model = Model([input1_tensor, input2_tensor], ouput_layer)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

model.fit([X1_train, X2_train], y_train, 
          validation_data=([X1_val, X2_val], y_val), 
          batch_size=128, epochs=6, verbose=2)

features_model = Model([input1_tensor, input2_tensor], merge_layer)
features_model.compile(loss='mse', optimizer='adam')

F_train = features_model.predict([X1_train, X2_train], batch_size=128)
F_val = features_model.predict([X1_val, X2_val], batch_size=128)

import xgboost as xgb

dTrain = xgb.DMatrix(F_train, label=y_train)
dVal = xgb.DMatrix(F_val, label=y_val)

xgb_params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eval_metric': 'logloss',
    'eta': 0.1, 
    'max_depth': 9,
    'subsample': 0.9,
    'colsample_bytree': 1 / F_train.shape[1]**0.5,
    'min_child_weight': 5,
    'silent': 1
}
bst = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dVal,'val')], 
                verbose_eval=10, early_stopping_rounds=10)


X1_test = create_padded_seqs(df_all[df_all['id'].notnull()]['question1'])
X2_test = create_padded_seqs(df_all[df_all['id'].notnull()]['question2'])

# print(X1_test)
# print(X2_test)
F_test = features_model.predict([X1_test, X2_test], batch_size=128)

dTest = xgb.DMatrix(F_test)

df_sub = pd.DataFrame({
        'id': df_all[df_all['id'].notnull()]['id'].values,
        'is_duplicate_actual':df_all[df_all['is_duplicate'].notnull()]['is_duplicate'].values, 
        'is_duplicate_predict': bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
    }).set_index('id')

# df_sub.sort_index(axis = 'id')

print(df_sub.head(50))
# df_sub['is_duplicate'].hist(bins=100)
