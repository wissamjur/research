import numpy as np

from sklearn.metrics import ndcg_score
from helpers import Helpers
s_helpers = Helpers()
import pandas as pd
from datetime import datetime as dt
from collections import OrderedDict 


'''
This script only runs on an environmet seperate from the one that has surprise since scikit-learn conflicts with sklearn
'''

neighbors_recs = s_helpers.load_dict('neighbors_dict')
neighbors2_recs = s_helpers.load_dict('neighbors2_dict')

neighbors_recs_updated = {}
# since the second dict might be smaller than the first and we need the keys (users) to be identical, then we have to update dict 1 to match dict 2
for key, value in neighbors_recs.items():
  for k, v in neighbors2_recs.items():
    if key == k:
      neighbors_recs_updated[key] = value

ds = [neighbors_recs_updated, neighbors2_recs]
final_dict = {}
for k in neighbors_recs_updated.keys():
  final_dict[k] = tuple(d[k] for d in ds)

results = {}
for key, item in final_dict.items():
    true_relevance = np.asarray([item[0]])
    scores = np.asarray([item[1]])
    ndcg = ndcg_score(true_relevance, scores)

    results[key] = [item,ndcg]

print(results[29])

'''
[
  (
    [131724, 5181, 5746, 5764, 5919, 6835, 7899, 3851, 4273, 187], 
    [131724, 5181, 5746, 5764, 5919, 6835, 7899, 3851, 187, 113275]
  ), 
  0.985341358883266
]
'''

items_df = pd.read_csv('../Research-old/datasets/ml_latest_small/movies.csv')
ratings_df = pd.read_csv('../Research-old/datasets/ml_latest_small/ratings.csv')
ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

final_df = ratings_df.set_index('movieId').\
                join(items_df.set_index('movieId'), how='left').reset_index()
u = final_df[final_df['userId'] == 29].sort_values(by=['date'])
# u[['userId','movieId','rating','date','genres']].to_csv('user_29_test.csv', index=False)

genres = []
genres_raw = u.genres.to_list()
for g in genres_raw:
  x = g.split('|')
  genres.extend(x)

def group_list(lst): 
  res =  [(el, lst.count(el)) for el in lst] 
  return list(OrderedDict(res).items())

print(group_list(['Action','Sci-Fi','IMAX','Action','Adventure','Sci-Fi','Crime','Drama','Action','Drama','War','Drama','War','Action','Romance','War','Western','Action','Horror','Sci-Fi','Action','Sci-Fi','IMAX']))




[131724, 5181, 5746, 5764, 5919, 6835, 7899, 3851, 4273, 187], 
[131724, 5181, 5746, 5764, 5919, 6835, 7899, 3851, 187, 113275]