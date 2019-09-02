# !pip install mojimoji mecab-python3
import pandas as pd
import os
import mojimoji
from pandas import Series, DataFrame
import numpy as np
import MeCab
import re

def mecab(document):
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(document).splitlines()
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    return words


def normalize_number(text):
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text


def delete_stopwords(doc):
    stopwords = ['@', '_', '!']
    sentence_words = []
    for word in doc:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if (re.compile(r"^.*[a-z]+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if word in stopwords: # ストップワードに含まれるものは除外
            continue
        sentence_words.append(word)        
    return sentence_words


# read csv
df=pd.read_csv("tweet_text.csv", engine='python')
print(df.shape)
df=df.dropna()
print(df.shape)


normalized_text=[]
labels=[]
for i, v in df.iterrows():
    if v["text"] is not None:
        sen = mojimoji.zen_to_han(v["text"])   # 1.半角＝＞全角
    else:
        continue
    if sen is not None:
        sen = normalize_number(sen)  # 2,数字の置き換え
        sen = mecab(sen)
    else:
        continue
    if len(sen)> 2:
        sen = delete_stopwords(sen)
        normalized_text.append(str(sen).replace("'", "").replace(",", " "))
        labels.append(int(v["label"]))
    else:
        continue




normalized_text=[s.split() for s in normalized_text]
print(len(normalized_text))


words = {}
for sentence in normalized_text:
  for word in sentence:
    if word not in words:
      words[word] = len(words)+1 # key to 0 padding is '+1'



# 文章を単語ID配列にする
data_x_vec = []
for sentence in normalized_text:
  sentence_ids = []
  for word in sentence:
    sentence_ids.append(words[word])
  data_x_vec.append(sentence_ids)
len(data_x_vec)


# 文章の長さを揃えるため、0でパディングする
max_sentence_size = 0
for sentence_vec in data_x_vec:
    if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)
print(max_sentence_size) # ==len(sentence_ids)

for sentence_ids in data_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.append(0) # 末尾に0を追加
print(len(sentence_ids))  



# arrayに変換
data_x_vecs = np.array(data_x_vec, dtype="int32")
print(data_x_vecs.shape)

total_label=np.array(labels)
print(total_label.shape)


# split to train and test
train_text=data_x_vecs[10000:]
test_text=data_x_vecs[:10000]

train_label = total_label[10000:]
test_label = total_label[:10000]

print(train_text.shape, test_text.shape)
print(train_label.shape, test_label.shape)


# save
np.save('train_text', train_text)
np.save('test_text', test_text)

np.save('train_label', train_label)
np.save('test_label', test_label)
