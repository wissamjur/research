'''
    Date: 7/8/2020

    Natural Noise filter class
    Paper: 5 - Detecting Noise in Recommender System Databases
    use it to apply noise filtering to a certain dataset of ratings


    Description: 
        1- We define the consistency c of a rating ru,v as the Mean Absolute Error between the actual and predicted rating
        2- if c is greater that the threshold th, then the rating is considered as noise

        In this class, we need to select a recommender and traing our data before we use it since it relies on the predicted ratings
'''

import swifter

from helpers.dataset import get_config_data, load_ratings
from surprise import KNNWithMeans, Dataset, Reader
import os
import pandas as pd
import numpy as np

# A reader is still needed but only the rating_scale param is requiered.
r_min = 1
r_max = 5
th = 0.4 # MAE rnd threshold used in coherence formula
reader = Reader(rating_scale=(r_min,r_max))

# The columns must correspond to user id, item id and ratings (in that order).
dataset_path = get_config_data()['dataset']
ratings_df = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)
data = Dataset.load_from_df(ratings_df, reader)


def compute_prediction(userId, itemId, rating):
    pred = algo.predict(userId, itemId, r_ui=rating, verbose=True)

    return pred

def compute_noise(userId, itemId, rating, prediction):
    estimate = prediction[54:]
    pred = estimate[:4]

    coherence = abs(rating - float(pred))/(r_max - r_min)

    if coherence > th:
        noise = 1
    else:
        noise = 0

    return noise

if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3.csv')):
    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # Build an algorithm, and train it.
    algo = KNNWithMeans()
    algo.fit(trainset)

    ratings_df['prediction'] = ratings_df.swifter.apply \
        (lambda x: compute_prediction(x.userId, x.itemId, x.rating), axis=1)

    ratings_df.to_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3.csv')

else:
    ratings_df = pd.read_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_predictions_nf3.csv')

appended_dataframes = []
splits = np.array_split(ratings_df, len(ratings_df)/18000)

for split in splits:
    split['coherence'] = split.swifter.apply \
            (lambda x: compute_noise(x.userId, x.itemId, x.rating, x.prediction), axis=1)
    
    appended_dataframes.append(split)

noise_df = pd.concat(appended_dataframes)
noise_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf3.csv', index=False)