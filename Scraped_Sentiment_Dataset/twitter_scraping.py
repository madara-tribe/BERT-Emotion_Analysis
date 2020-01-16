import pandas as pd
import os
from os.path import join
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# get tweet text and emotion label
emotions = [["angry", "angry1", "angry2"], ["disgust", "disgust2"], ["fear", "fear2", "fear3"], ["happy", "happy2", "happy3"],
            "sad", ["surprise", "surprise2", "surprise3"]]
dir_path = "sentiment_twitterscraping"

size = 60000
df = []
for i, es in enumerate(emotions):
    if isinstance(es, list):
        for e in es:
            try:
                data = shuffle(pd.read_json(join(dir_path, "{}.json".format(e)))).iloc[:int(size/len(es))]
                data['label'] = i
                df.append(data)
            except ValueError:
                continue
                
    else:
        data = shuffle(pd.read_json(join(dir_path, "{}.json".format(es)))).iloc[:int(size)]
        data['label'] = i
        df.append(data)
        
df = pd.concat(df)
df = shuffle(df)
text_df = df['text']
label_df = df['label']
print(text_df.shape, label_df.shape)


twdf=pd.concat([text_df, label_df], axis=1)
# save to csv
twdf.to_csv('tweet_text.csv')


# read csv
df=pd.read_csv("tweet_text.csv", engine='python')
