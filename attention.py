from __future__ import print_function
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime, time, json,csv
from keras.models import Model
from keras.layers import Input,Bidirectional,LSTM, dot , add, Reshape ,Flatten, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file

# MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 25
QUESTION_PAIRS_FILE = 'data/quora_duplicate_questions.tsv'
GLOVE_FILE = 'data/glove.6B/glove.6B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train.npy'
Q2_TRAINING_DATA_FILE = 'q2_train.npy'
LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25
DROPOUT = 0.1
BATCH_SIZE = 32

q1_list = []
q2_list = []
label   = []

with open(QUESTION_PAIRS_FILE, encoding = 'utf-8') as csvfile:
	readlist = csv.DictReader(csvfile,delimiter = '\t')
	for line in readlist:
		q1_list.append(line['question1'])
		q2_list.append(line['question2'])
		label.append(line['is_duplicate'])


# In[14]:

total_text = q1_list + q2_list
tokenizer = Tokenizer()
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS) 
tokenizer.fit_on_texts(total_text)

q1_sequence = tokenizer.texts_to_sequences(q1_list)
q2_sequence = tokenizer.texts_to_sequences(q2_list)
word_index = tokenizer.word_index
MAX_NB_WORDS = len(word_index)

embed_index = {}

with open(GLOVE_FILE,encoding='utf-8') as file:
    for line in file:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embed_index[word] = embedding


word_embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))

for word,i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embed_vector = embed_index.get(word)
    if embed_vector is not None:
        word_embedding_matrix[i] = embed_vector


q1_pad_sequence = pad_sequences(q1_sequence, maxlen = MAX_SEQUENCE_LENGTH)
q2_pad_sequence = pad_sequences(q2_sequence, maxlen = MAX_SEQUENCE_LENGTH)
labels = np.array(label, dtype=int)

q1_training = Q1_TRAINING_DATA_FILE
q2_training = Q2_TRAINING_DATA_FILE
train_labels = LABEL_TRAINING_DATA_FILE
train_word_embedding = WORD_EMBEDDING_MATRIX_FILE

np.save(open(q1_training,'wb'),q1_pad_sequence)
np.save(open(q2_training,'wb'),q2_pad_sequence)
np.save(open(train_labels,'wb'),labels)
np.save(open(train_word_embedding,'wb'),word_embedding_matrix)

word_data_file = 'num_words.json'
with open(word_data_file, 'w') as f:
    json.dump({'num_words':MAX_NB_WORDS},f)

q1_data = np.load(open(q1_training, 'rb'))
q2_data = np.load(open(q2_training, 'rb'))

q1_train = np.load(open(q1_training,'rb'))
q2_train = np.load(open(q2_training,'rb'))
label_train = np.load(open(train_labels,'rb'))
word_embed_train = np.load(open(train_word_embedding,'rb'))
with open(word_data_file,'r') as f:
    MAX_NB_WORDS = json.load(f)['num_words']

X = np.stack((q1_train, q2_train),axis = 1)
Y = labels

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = test_split, random_state = RNG_SEED)

q1_trainset = X_train[:,0]
q2_trainset = X_train[:,1]
q1_testset = X_test[:,0]
q2_testset = X_test[:,1]

question1 = Input(shape=(max_sequence_length,))
question2 = Input(shape=(max_sequence_length,))

sent_embed_dim = 128
q1 = Embedding(MAX_NB_WORDS + 1, 
                 EMBEDDING_DIM, 
                 weights=[word_embed_train], 
                 input_length=max_sequence_length, 
                 trainable=False)(question1)

q1 = Bidirectional(LSTM(sent_embed_dim, return_sequences = True), merge_mode = "sum")(q1)

q2 = Embedding(MAX_NB_WORDS + 1,
              EMBEDDING_DIM,
              weights = [word_embed_train],
              input_length = max_sequence_length,
              trainable = False)(question2)
q2 = Bidirectional(LSTM(sent_embed_dim, return_sequences = True), merge_mode = "sum")(q2)

#creating attention network

attn = dot([q1,q2],[1,1])
attn = Flatten()(attn)
attn = Dense((max_sequence_length * sent_embed_dim))(attn)
attn = Reshape((max_sequence_length , sent_embed_dim))(attn)

# creating model
model = add([q1,attn])
model = Flatten()(model)
model = Dense(300, activation='relu')(model)
model = Dropout(dropout)(model)
model = BatchNormalization()(model)
model = Dense(300, activation='relu')(model)
model = Dropout(dropout)(model)
model = BatchNormalization()(model)
model = Dense(300, activation='relu')(model)
model = Dropout(dropout)(model)
model = BatchNormalization()(model)
model = Dense(300, activation='relu')(model)
model = Dropout(dropout)(model)
model = BatchNormalization()(model)
model = Dense(300, activation='relu')(model)
model = Dropout(dropout)(model)
model = BatchNormalization()(model)

output_label = Dense(1, activation = 'sigmoid')(model)
model = Model(inputs = [question1,question2],outputs = output_label)
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
history = model.fit([q1_trainset,q2_trainset],
                   Y_train,
                   epochs = NB_EPOCHS,
                   validation_split = VALIDATION_SPLIT,
                    verbose =2,
                    batch_size = BATCH_SIZE,
                    callbacks = callbacks
                   )

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

model.save(MODEL_WEIGHTS_FILE)
model.load_weights(model_weight_file)
loss, accuracy = model.evaluate([q1_testset,q2_testset], Y_test, verbose = 0) 

print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))

