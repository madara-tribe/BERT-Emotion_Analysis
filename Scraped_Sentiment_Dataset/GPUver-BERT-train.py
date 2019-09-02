from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf
import numpy as np
from functools import partial
import tensorflow.keras.backend as K
from tensorflow.python import keras
from keras.callbacks import *
from HistoryCallbackLoss import HistoryCheckpoint
!pip install -q keras-bert
# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'



import codecs
from tqdm import tqdm
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint

SEQ_LEN = 350
BATCH_SIZE = 10
EPOCHS = 2
LR = 1e-4
pretrained_path = "drive/My Drive/bert-wiki-ja"
config_path = 'bert_config.json'
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-1400000')

# keras bert
bert = load_trained_model_from_checkpoint(config_path,
  checkpoint_path, training=True, trainable=True, seq_len=SEQ_LEN)
bert.summary()

print("load saved text")
# train text
train_x = np.load("drive/My Drive/train_text.npy")
train_y = np.load("drive/My Drive/train_label.npy")
print(train_x.shape, train_y.shape)

# test text
test_x = np.load("drive/My Drive/test_text.npy")
test_y = np.load("drive/My Drive/test_label.npy")

print(test_x.shape, test_y.shape)

# adjust data to bert
test_X= [test_x, np.zeros_like(test_x)]
X = [train_x, np.zeros_like(train_x)]

def bert_model():
  inputs = bert.inputs[:2]
  dense = bert.get_layer('NSP-Dense').output
  outputs = keras.layers.Dense(units=6, activation='softmax')(dense)

  decay_steps, warmup_steps = calc_train_steps(train_y.shape[0],
      batch_size=BATCH_SIZE, epochs=EPOCHS)

  model = keras.models.Model(inputs, outputs)
  # model.load_weights(weight_path)
  model.compile(AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
      loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
  model.summary()
  return model

model = bert_model()

# callback
callback=[]
callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period=1))
#callback.append(LearningRateScheduler(step_decay))
callback.append(TensorBoard(log_dir='tb/'))
callback.append(ModelCheckpoint('logss/{epoch:02d}_metric.hdf5', monitor='loss', verbose=1))

# train
model.fit(X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callback)

# predict
predicts = model.predict(test_X, verbose=True).argmax(axis=-1)
print(np.sum(test_y == predicts) / test_y.shape[0]) # 0.9429347826086957

