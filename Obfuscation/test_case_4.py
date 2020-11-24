import sys
sys.path.append('./NaturalNoise/')
import pandas as pd
import numpy as np
import os.path

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from helpers import Helpers
from datetime import datetime as dt

ratings_path = '../Research-old/datasets/ml_latest_small/ratings-29-removed-days.csv'
s_helpers = Helpers()

# load the ratings csv
ratings_df = pd.read_csv(ratings_path).rename({'movieId': 'itemId'}, axis=1)

# calculate the neighbors of all users based on the original dataset
# get the neighbors_recs dictionary that has the list of neighboras for each user
if not os.path.exists('Obfuscation/output/neighbors_dict.pkl'):
    data = Dataset.load_from_df(ratings_df[['userId','itemId','rating']],Reader(rating_scale=(1,5)))
    # build the full trainset
    trainset = data.build_full_trainset()
    # select the recommender algorithm
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = s_helpers.get_top_n(predictions, n=10)

    neighbors_recs = {}
    for uid, user_ratings in top_n.items():
        neighbors_recs[uid] = [iid for (iid, _) in user_ratings]

    s_helpers.save_dict(neighbors_recs,'neighbors_dict')

else:
    neighbors_recs = s_helpers.load_dict('neighbors_dict')

if not os.path.exists('Obfuscation/output/neighbors2_dict.pkl'):
    # ## consider the users who have more than 5 rating days in the dataset
    # # get the total ratings per user and the total days for every user
    # ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())
    # ratings_per_day = ratings_df.groupby(['userId','date']).size().reset_index(name="ratings_per_day")
    # days_per_user = ratings_per_day.groupby(['userId']).size().reset_index(name="days_per_user")
    # m1 = days_per_user[days_per_user['days_per_user'] > 5].userId.to_list()
    # users_m1 = ratings_per_day[ratings_per_day['userId'].isin(m1)].sort_values(by=['date'])
    # # now remove last 5 *rating days* of all users and get the new neighbors
    # last_n_days_removed = users_m1.drop(users_m1.groupby(['userId']).tail(5).index, axis=0)
    # last_n_days_removed['user_date'] = list(zip(last_n_days_removed.userId, last_n_days_removed.date))
    # # apply tuple (user,date) to merge with the main df and get all the ratings without the last rating days
    # ratings_df2 = ratings_df[ratings_df[['userId', 'date']].apply(tuple, axis=1).isin(last_n_days_removed.user_date)]

    data = Dataset.load_from_df(ratings_df[['userId','itemId','rating']],Reader(rating_scale=(1,5)))
    # build the full trainset
    trainset = data.build_full_trainset()
    # select the recommender algorithm
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = s_helpers.get_top_n(predictions, n=10)

    neighbors2_recs = {}
    for uid, user_ratings in top_n.items():
        neighbors2_recs[uid] = [iid for (iid, _) in user_ratings]

    s_helpers.save_dict(neighbors2_recs,'neighbors2_dict')

else:
    neighbors2_recs = s_helpers.load_dict('neighbors_dict')

# ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())
# x = ratings_df[ratings_df['userId'] == 568]
# print(ratings_df)