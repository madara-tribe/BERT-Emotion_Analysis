"""
・livedoor text file stracture
1 line =>text URL
2 line => data
3 line => title
4 line => text
"""
!pip install mecab-python3 && pip install mojimoji
from google.colab import drive
drive.mount('/content/drive')


import os
import mojimoji
import numpy as np
import MeCab
import re

"""1. load text data"""
# difine c at first
c=0
L={}
# load text sentence for deleting each text fileのN行目

text_path =  ["drive/My Drive/sports-watch", "drive/My Drive/topic-news"]
for i, path in enumerate(os.listdir(text_path[c]), len(L)):
  _list =[]
  with open(text_path[c]+'/'+path, "r") as f:
  #text_list = text_data.readlines()
     for _l in f:
        _l = _l.rstrip()
        if _l:
           _list.append(_l)
  L[i] = _list
  print(len(L))
c+=1
print(c, len(L))

"""2. divide words """
def mecab(document):
  mecab = MeCab.Tagger("-Ochasen")
  lines = mecab.parse(document).splitlines()
  words = []
  for line in lines:
      chunks = line.split('\t')
      if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
          words.append(chunks[0])
  return words

"""3. text normalization """
def normalize_number(text):
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

"""4. delete stop words & TF-IDF """
stopwords = ["——。", "！？", "0", "——", "\\u3000’", "\\u3000’"]
 
def delete_stopwords(doc):
  sentence_words = []
  for word in doc:
    if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None): # 数字が含まれるものは除外
      continue
    if word in stopwords: # ストップワードに含まれるものは除外
      continue
    #  特定の文字列(chr(92)=="\")を含む文字列削除
    if chr(92) in word or '——' in word or '″' in word:    
      continue
    sentence_words.append(word)        
  return sentence_words

S=[]
for i in range(0, len(L)):
  sen = str(L[i][2:]).strip().replace('\n', ' ')
  sen = mojimoji.han_to_zen(sen)   # 1.半角＝＞全角
  sen = normalize_number(sen)  # 2,数字の置き換え
  sen = mecab(sen)  # 単語の分割
  sen = delete_stopwords(sen) # delete stop words
  S.append(str(sen).replace("'", "").replace(",", " "))     # replace「'」and「,」 into blank
# print(len(S))  # S =>> '[オリックス  公式  アカウント  ヘディング  脳  発言  波紋  オリックス・バファローズ  公式  アカウント

"""4. TF-IDF or delete few words（　http://ailaby.com/tfidf/　）"""

"""
import collections
# make few words list
SS=[s.split() for s in S]
TT=[]
for s in SS:
  for word in s:
    TT.append(word)

count = collections.Counter(TT)
few_wordlist=[]
for word, amount in count.items():
  if amount <=1:
    continue
  few_wordlist.append(word)


def delete_sewwords(doc):
  sentence_words = []
  for word in doc:
    if word in few_wordlist: # ストップワードに含まれるものは除外
      continue
    sentence_words.append(word)        
  return sentence_words

# delete few words
train_x=[]
for docs in SS:
  docs = delete_sewwords(docs)
  train_x.append(docs)
"""

# reverse
SS=[s.split() for s in S]
print(len(SS))

"""5. convert text into id vector"""

# 単語のidx辞書
words = {}
for sentence in SS:
  for word in sentence:
    if word not in words:
      words[word] = len(words)+1 # key to 0 padding is '+1'
# print('len {}, min {}, max {},'.format(len(words.values()), min(words.values()), max(words.values())))

# 文章を単語ID配列にする
data_x_vec = []
for sentence in SS:
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

### make text data for train test ###

train0=data_x_vecs[:700]
test0=data_x_vecs[700:900]
trainsx = np.vstack((train0, train1))

train1=data_x_vecs[900:1500]
test1=data_x_vecs[1500:]

testsx = np.vstack((test0, test1))
print(trainsx.shape, testsx.shape)

# save 
np.save('train_xs', trainsx)
np.save('test_xs', testsx)

### make label for train test ###

shape=[900, 770]
label=[]
for N, shape in enumerate(shape):
  label.append([N]*shape)

L=[]
for l in label:
  for la in l:
    L.append(la)
Ls = np.array(L)
print(Ls.shape)

trl0=Ls[:700]
tel0=Ls[700:900]

trl1=Ls[900:1500]
tel1=Ls[1500:]

train_sy = np.hstack((trl0, trl1))
test_sy = np.hstack((tel0, tel1))
print(train_sy.shape, test_sy.shape)

# save
np.save('train_label', train_sy)
np.save('test_label', test_sy)

# save train and test text
np.save('all_text', np.array(SS))
