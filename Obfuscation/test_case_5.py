import sys
sys.path.append('./NaturalNoise/')
import pandas as pd
import numpy as np
import os.path

# from surprise import KNNWitethMeans
# from surprise import accuracy
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from helpers import Helpers
from datetime import datetime as dt

s_helpers = Helpers()

ratings_path = '../Research-old/datasets/ml_latest_small/ratings.csv'

ratings_df = pd.read_csv(ratings_path).rename({'movieId': 'itemId'}, axis=1)
ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())
# ratings_df.sort_values(by=['userId','date']).to_csv('ratings_with_date_column.csv', index=False)

# calculate the top-10 item recommendations of all users based on the original dataset
# get the recommendations_recs dictionary that has the list of neighboras for each user
if not os.path.exists('Obfuscation/output/predictions_dict.pkl'):
    tuser = 599

    data = Dataset.load_from_df(ratings_df[['userId','itemId','rating']],Reader(rating_scale=(1,5)))
    # build the full trainset
    trainset = data.build_full_trainset()
    # select the recommender algorithm
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    ## Get the target user's KNN
    tuser_inner_id = algo.trainset.to_inner_uid(tuser)
    print("Target user_id *{}* and inner_id *{}*".format(tuser, tuser_inner_id))
    tuser_neighbors_iids = algo.get_neighbors(tuser_inner_id, k=10)
    tuser_neighbors = [algo.trainset.to_raw_uid(id) for id in tuser_neighbors_iids]
    print("Target user neighbors: ", tuser_neighbors)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = s_helpers.get_top_n(predictions, n=10)

    neighbors_recs = {}
    for uid, user_ratings in top_n.items():
        neighbors_recs[uid] = [iid for (iid, _) in user_ratings]

    
    # Then compute MAE and RMSE
    accuracy.mae(predictions)
    accuracy.rmse(predictions)

    s_helpers.save_dict(neighbors_recs,'predictions_dict')

else:
    neighbors_recs = s_helpers.load_dict('predictions_dict')

# get the total ratings per user and the total days for every user
ratings_per_day = ratings_df.groupby(['userId','date']).size().reset_index(name="ratings_per_day")
days_per_user = ratings_per_day.groupby(['userId']).size().reset_index(name="days_per_user")
# select only the users that have at least 20 rating days (metric 1)
m1 = days_per_user[days_per_user['days_per_user'] > 15].userId.to_list()
users_m1 = ratings_per_day[ratings_per_day['userId'].isin(m1)]

users_m1.to_csv('users.csv')