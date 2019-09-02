!pip install ConfigArgParse
import tensorflow as tf
import os
import numpy as np
import configargparse
from model import Model 
import gensim.models as word2vec
import time
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
learning_rate=0.01
max_article_length = 500
batch_size = 300
gpu_index = 0
epoch = 10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = Model()
classifier = tf.estimator.Estimator(model_fn=model.build, 
                                    config=tf.estimator.RunConfig(session_config=config,
                                                                  model_dir=os.path.join('ckpt', time.strftime("%m%d_%H%M%S"))),
                                    params={
                                        'feature_columns': [tf.feature_column.numeric_column(key='x')], \
                                        'kernels': [(3,512),(4,512),(5,512)], \
                                        'num_classes': 2, \
                                        'learning_rate': learning_rate, \
                                        'max_article_length': max_article_length})

# below is URL to download 'GoogleNews-vectors-negative300-SLIM.bin'

# https://www.kaggle.com/nareyko/googlenewsvectorsnegative300slim#GoogleNews-vectors-negative300-SLIM.bin.gz


class Word2vecEmbedder():
    def __init__(self, model, max_vocab_size, embedding_dim):
        self.w2v = model
        self.max_vocab_size = max_vocab_size
        self.embedding_dim = embedding_dim
        self._build()

    def _build(self):
        self.vocab_dict = self._get_vocab_dict(self.w2v, self.max_vocab_size)
        self.oov = [0 for _ in range(self.embedding_dim)]

    def _get_vocab_dict(self, model, max_vocab_size):
        assert model != None, "word2vec was not trained."
        vocab_dict = {}
        for key in sorted(model.wv.vocab):
            if model.wv.vocab[key].__dict__['index']<max_vocab_size :
                vocab_dict[key] = model.wv.vocab[key].__dict__['index']
        return vocab_dict

    def get_embedding(self, sentence):
        raise NotImplementedError


class Word2vecEnWordEmbedder(Word2vecEmbedder):
    def __init__(self):
        model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin', \
                                                           binary=True)
        max_vocab_size = 20000
        embedding_dim = 300
        super().__init__(model, max_vocab_size, embedding_dim)

    def get_embedding(self, sentence):
        embedding = []
        for word in sentence.split(' '):
            vocab_idx = self.vocab_dict.get(word, -1)
            if vocab_idx > 0:
                embedding.append(self.w2v.wv.vectors[vocab_idx])
            else:
                embedding.append(self.oov)
        return np.array(embedding)

class Dataset():
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.embedder = None
    
    def _generator(self, _x, _y=None):
        def _internal_generator():
            for idx, line in enumerate(_x):
                sentence_feature = self.embedder.get_embedding(line)
                if _y == None:
                    yield (sentence_feature, -1)
                else:
                    yield (sentence_feature, _y[idx])
        return _internal_generator
    
    def train_input_fn(self, batch_size, padded_size, epoch=20, shuffle=True):
        g = self._generator(self.train_x, self.train_y)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        if shuffle:
            dataset = dataset.shuffle(9876543)
        dataset = dataset.repeat(epoch)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}, label

    def eval_input_fn(self, batch_size, padded_size):
        g = self._generator(self.test_x, self.test_y)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        dataset = dataset.padded_batch(batch_size, padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}, label

    def predict_input_fn(self, _inputs: list, padded_size):
        g = self._generator(_inputs)
        dataset = tf.data.Dataset.from_generator(g, output_types=(tf.float32, tf.int32),
                                                 output_shapes=([None, self.embedder.embedding_dim], []))
        dataset = dataset.padded_batch(len(_inputs), padded_shapes=([padded_size, self.embedder.embedding_dim], []))
        iterator = dataset.make_one_shot_iterator()
        feature, label = iterator.get_next()
        return {"x": feature}
    
class SST(Dataset):
    def __init__(self, embed_cls):
        super().__init__()
        self.train_x, self.train_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/stanford-sentiment-dataset/master/stsa.binary.phrases.train')
        self.test_x, self.test_y = self._load_data('https://raw.githubusercontent.com/HaebinShin/stanford-sentiment-dataset/master/stsa.binary.test')
        self.embedder = embed_cls()

    def _maybe_download(self, _url):
        _path = tf.keras.utils.get_file(fname=_url.split('/')[-1], origin=_url)
        return _path

    def _load_data(self, url):
        path = self._maybe_download(url)
        contents = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line[0]
                docu = line[2:]
                if len(docu.strip())==0: continue
                contents.append(docu.strip())
                labels.append(int(label.strip()))
        return contents, labels

# dataset
data = SST(Word2vecEnWordEmbedder)

# train
classifier.train(input_fn=lambda: data.train_input_fn(batch_size=batch_size, padded_size=max_article_length, epoch=epoch))

# test
eval_val = classifier.evaluate(input_fn=lambda: data.eval_input_fn(batch_size=batch_size, padded_size=max_article_length))
print(eval_val)

# visualize text 
def _plot_score(vec, pred_text, xticks):
    _axis_fontsize=13
    fig=plt.figure(figsize = (14,10))
    plt.yticks([])
    plt.xticks(range(0,len(vec)), xticks, fontsize=_axis_fontsize)
    fig.add_subplot(1, 1, 1)
    plt.figtext(x=0.13, y=0.54, s='Prediction: {}'.format(pred_text), fontsize=15, fontname='sans-serif')
    img = plt.imshow([vec], vmin=0, vmax=1)
    plt.show()
    
def _get_text_xticks(sentence):
    tokens = [word_.strip() for word_ in sentence.split(' ')]
    return tokens

def visualize_gradcam_text(sentences, checkpoint_path, max_article_length):
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    model = Model()
    classifier = tf.estimator.Estimator(model_fn=model.build,
                                        config=tf.estimator.RunConfig(session_config=config),
                                        params={
                                            'feature_columns': [tf.feature_column.numeric_column(key='x')], \
                                            'kernels': [(3,512),(4,512),(5,512)], \
                                            'num_classes': 2, \
                                            'max_article_length': max_article_length})
    
    data = SST(Word2vecEnWordEmbedder)
    pred_val = classifier.predict(input_fn=lambda: data.predict_input_fn(sentences, padded_size=MAX_ARTICLE_LENGTH),
                                  checkpoint_path=checkpoint_path)
    for i, _val in enumerate(pred_val):
        pred_idx = _val['predict_index'][0]
        vec = _val['grad_cam'][pred_idx][:17]
        pred_text = "Negative" if pred_idx==0 else "Positive"
        _plot_score(vec=vec, pred_text=pred_text, xticks=_get_text_xticks(sentences[i]))

CKPT = './ckpt/0723_033050/model.ckpt-2566'
MAX_ARTICLE_LENGTH = 500
visualize_gradcam_text(['the movie exists for its soccer action and its fine acting .',
                      'the thrill is -lrb- long -rrb- gone .',
                      "now it 's just tired .",
                      'the cast is very excellent and relaxed .'], checkpoint_path=CKPT, 
                       max_article_length=MAX_ARTICLE_LENGTH )



"""
sentences=[]
for sent in train_x[:100]:
  if len(sent.split())>=129:
    sentences.append(sent.split()[:128])
  
def _get_text_xticks(sentence):
    tokens = [word_.strip() for word_ in sentence]
    return tokens
  
  
def _plot_score(vec, pred_text, xticks):
    _axis_fontsize=13
    fig=plt.figure(figsize = (14,10))
    plt.yticks([])
    plt.xticks(range(0,8), xticks, fontsize=_axis_fontsize)
    fig.add_subplot(1, 1, 1)
    plt.figtext(x=0.13, y=0.54, s='Prediction: {}'.format(pred_text), fontsize=15, fontname='sans-serif')
    img = plt.imshow([vec], vmin=0, vmax=1)
    plt.show()
    

for i, (pred_idx, val) in enumerate(zip(negpos, vec)):
  pred_text = "Negative" if pred_idx==0 else "Positive"
  _plot_score(vec=val, pred_text=pred_text, xticks=_get_text_xticks(sentences[i]))
"""
