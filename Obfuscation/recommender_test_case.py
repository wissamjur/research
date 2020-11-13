import sys
sys.path.append('./NaturalNoise/')

import pandas as pd
from noise_filter_1 import NoiseFilter1
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import get_dataset_dir
from surprise import Reader
from datetime import datetime as dt
from surprise_helpers import SurpriseHelpers

ratings_path = '../Research/datasets/ml_latest_small/ratings.csv'
noise = NoiseFilter1()
s_helpers = SurpriseHelpers()
tuser = 179
tuser_neighbors = [291, 601, 147, 571, 88, 310, 311, 585, 139, 342]

# load the ratings csv
ratings_df = pd.read_csv(ratings_path).rename({'movieId': 'itemId'}, axis=1)
# call the first noise algo to get the dataset with natural noise
ratings_wndf = noise.get_dataset_with_noise(ratings_df)
ratings_wndf['date'] = ratings_wndf['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

rdf_surprise = ratings_wndf[['userId','itemId','rating']]
# ratings dataset without the opt-out user included. In this case we will be checking the effect of removing an opt-out user on his neighborhood
rdf_surprise_wot = rdf_surprise.drop(rdf_surprise[rdf_surprise.userId == tuser].index)
data = Dataset.load_from_df(rdf_surprise_wot,Reader(rating_scale=(1,5)))
# build the full trainset
trainset = data.build_full_trainset()
# select the recommender algorithm
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNWithMeans(sim_options=sim_options)
algo.fit(trainset)

# get the recommendations for the list of neighbors of the opt-out user
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = s_helpers.get_top_n(predictions, n=10)

neighbors_recs = {}
for uid, user_ratings in top_n.items():
    if uid in tuser_neighbors:
      neighbors_recs[uid] = [iid for (iid, _) in user_ratings]

print(neighbors_recs)