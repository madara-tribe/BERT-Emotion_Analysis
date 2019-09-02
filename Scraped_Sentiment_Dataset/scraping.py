from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import Series, DataFrame
import time
import os
import itertools

# https://coco.to/movie/87205

rating = []
reviews =[]
first_url = 'https://filmarks.com/movies/82210'
next_urls = 'https://filmarks.com/movies/82210?page='
for i in range(1,200):
  if i==1:
    next_url = first_url
  else:
    next_url = next_urls+str(i)
    
  result = requests.get(next_url)
  c = result.content
  #HTMLを元に、オブジェクトを作る
  soup = BeautifulSoup(c, "lxml")
  #リストの部分を切り出し
  sums = soup.find("div",{'class':'l-main'})
  com = sums.find_all('div', {'class':'p-mark'})

  # get review
  for rev in com:
    reviews.append(rev.text)
  # get rating
  for crate in com:
    for rate in crate.find_all('div', {'class':'c-rating__score'}):
      rating.append(rate.text)
  # print(i)

# save review data as DataFrame
rev_list = Series(reviews)
rate_list = Series(rating)
print(len(rev_list), len(rate_list))

movie_df = pd.concat([rev_list, rate_list],axis=1)
movie_df.columns=['review','rating']
movie_df.to_csv('movie_review.csv', sep = '\t',encoding='utf-16')

