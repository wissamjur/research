import sys
sys.path.append('./NaturalNoise/')

import pandas as pd
from noise_filter_1 import NoiseFilter1
from obfuscation import Optout
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from datetime import datetime as dt
from helpers import Helpers


ratings_path = '../Research-old/datasets/ml_latest_small/ratings.csv'
noise = NoiseFilter1()
optout = Optout()
s_helpers = Helpers()

# load the ratings csv
ratings_df = pd.read_csv(ratings_path).rename({'movieId': 'itemId'}, axis=1)
# call the first noise algo to get the dataset with natural noise
ratings_wndf = noise.get_dataset_with_noise(ratings_df)
# call the optout function that users the natural noise dataset
opt_out_users = optout.get_opt_out_users(ratings_wndf)

'''In this section, we will analyze the users that were considered as opt-out by the previous algorithm that's based on the noise filter. 
To do that, we will first find the neighbors of one of those users and then check the recommendations his closest neighbor got. 
After that, we will remove the opt-out profile from the dataset and re-run the same method in order to verify that in fact 
the closest neighbor would get different ratings when the opt-out profile has been eliminated.'''
# target opt-out user
tuser = 483
ratings_wndf['date'] = ratings_wndf['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())
tuser_df = ratings_wndf.loc[ratings_wndf['userId'] == tuser].sort_values(by=['date'])

## Use surpriselib to build a full trainset and then select the recommender algorithm to be used for generating the recommendations
# load a ratings df compatible with surprise and another copy that doesn't include the opt-out user
rdf_surprise = ratings_wndf[['userId','itemId','rating']]
data = Dataset.load_from_df(rdf_surprise,Reader(rating_scale=(1,5)))
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

## Now, predict ratings for all pairs (u, i) that are NOT in the training set and get the neighbors top-n recommendations for 
## all the neighbors of the target user
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
top_n = s_helpers.get_top_n(predictions, n=10)

neighbors_recs = {}
for uid, user_ratings in top_n.items():
    if uid in tuser_neighbors:
      neighbors_recs[uid] = [iid for (iid, _) in user_ratings]

print("top-n recommendations for target user neighbors: ", neighbors_recs)