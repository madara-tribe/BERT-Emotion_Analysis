# https://github.com/CyberZHG/keras-self-attention/blob/master/README.md
!pip install keras-self-attention

!pip install -q keras-bert
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip -o uncased_L-12_H-768_A-12.zip

import codecs
from tqdm import tqdm
from keras_bert import Tokenizer
from keras_bert import get_custom_objects

import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import History, LearningRateScheduler, Callback
from keras import layers
from keras.models import Model, Sequential
from keras.layers import *
from functools import partial
import numpy as np
np.load = partial(np.load, allow_pickle=True)

pretrained_path = 'uncased_L-12_H-768_A-12'
#config_path = os.path.join(pretrained_path, 'bert_config.json')
#checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# load saved text
train_x = np.load('train_xs.npy')
train_y = np.load('train_label.npy')
test_x = np.load('test_xs.npy')
test_y = np.load('test_label.npy')
all_text = np.load('all_text.npy')
print(train_x.shape, train_x.max())

# one-hot
n_labels = len(np.unique(train_y))
train_y=np.eye(n_labels)[train_y] 
train_y = np.array(train_y)
print(train_y.shape)

"""
Self-Attention
input (query) と memory (key, value) すべてが同じ Tensor を使う Attention です。
"""
from keras_self_attention import SeqSelfAttention
import pandas as pd
from IPython.core.display import display, HTML

h_dim=356
seq_len = 691
vocab_size = 23569+1
inp = Input(batch_shape = [None, seq_len])
emb = Embedding(vocab_size, 300)(inp) # (?,128,32)
att_layer = SeqSelfAttention(name='attention')(emb)
out = Bidirectional(LSTM(h_dim))(att_layer)
output = Dense(2, activation='softmax')(out)  # shape=(?, 2)
model = Model(inp, output)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc', 'mse', 'mae'])
model.summary()

try:
  model.fit(train_x, train_y, epochs=1, batch_size=10)
finally:
  model.save_weights("model.h5")

predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
print(np.sum(test_y == predicts) / test_y.shape[0])

emodel = Model(inputs=model.input, outputs=[model.output, model.layers[2].output])
emodel.summary()

predict=emodel.predict(test_x)

# test_xの[0] == all_text[700] len()==175
token = all_text[700]
weight = [w.max() for w in predict[1][0][:175]]  # test_x[0][:176]

df = pd.DataFrame([token, weight], index=['token', 'weight'])
mean = np.array(weight).mean()
print(df.shape, mean)
df = df.T

df['rank'] = df['weight'].rank(ascending=False)
df['normalized'] = df['weight'].apply(lambda w: max(w - mean, 0))
df['weight'] = df['weight'].astype('float32')
df['attention'] = df['normalized'] > 0

# visualize attention by pandas
def pd_visualize_attention(df):
  dfs = df.style.background_gradient(cmap='Blues', subset=['normalized'])
  display(dfs)

# visualize attention by HTML
def html_visualize_attention(df):
  html = ""
  for i, v in df.iterrows():
    attn = v['normalized']*100  # ×100
    word = v['token']
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    html += ' ' + '<span style="background-color: {}">{}</span>'.format(html_color, word)
  display(HTML(html))

html_visualize_attention(df)

pd_visualize_attention(df)

