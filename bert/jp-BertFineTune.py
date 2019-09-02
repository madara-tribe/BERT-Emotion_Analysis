!pip uninstall tensorflow && pip install tensorflow==1.13.1
from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf
import numpy as np
from functools import partial
import tensorflow.keras.backend as K
from tensorflow.python import keras
!pip install -q keras-bert
# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'

import codecs
from tqdm import tqdm
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint


SEQ_LEN = 691
BATCH_SIZE = 100
EPOCHS = 2
LR = 1e-4


pretrained_path = "drive/My Drive/bert-wiki-ja"
config_path = 'bert_config.json'
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-1400000')

bert = load_trained_model_from_checkpoint(config_path,
    checkpoint_path, training=True, trainable=True, seq_len=SEQ_LEN)
bert.summary()

# load saved text
np.load = partial(np.load, allow_pickle=True)

# train text
train_x = np.load('train_xs.npy')
train_y = np.load('train_label.npy')
train_xs = [text[:512] for text in train_x]
train_xs = np.array(train_xs)
train_ys = train_y[:512]

# test text
test_x = np.load('test_xs.npy')
test_y = np.load('test_label.npy')
test_xs = [text[:512] for text in test_x]
test_xs = np.array(test_xs)
test_xs = test_xs[:368]
test_y = test_y[:368]

# all text data
all_text = np.load('all_text.npy')
print(train_xs.shape,  train_y.shape, train_xs.max())
print(test_xs.shape, test_y.shape)

# adjust data to bert
test_X= [test_xs, np.zeros_like(test_xs)]
X = [train_xs, np.zeros_like(train_xs)]

# rebuild BERT model
inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

decay_steps, warmup_steps = calc_train_steps(train_y.shape[0],
    batch_size=BATCH_SIZE, epochs=EPOCHS)

model = keras.models.Model(inputs, outputs)
model.compile(AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
    loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# Initialize Variables
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
sess.run(init_op)

# Convert to TPU Model
tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
strategy = tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address))

with tf.keras.utils.custom_object_scope(get_custom_objects()):
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

# train
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    tpu_model.fit(X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Predictiton and accuracy
with tf.keras.utils.custom_object_scope(get_custom_objects()):
    predicts = tpu_model.predict(test_X, verbose=True).argmax(axis=-1)
print(np.sum(test_y == predicts) / test_y.shape[0]) # 0.9429347826086957
