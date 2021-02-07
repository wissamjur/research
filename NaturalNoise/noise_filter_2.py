'''
    Date: 7/5/2020

    Natural Noise filter class
    Papers: 12 - The Magic Barrier of Recommender Systems – No Magic, Just Ratings
            13 - A Novel Framework to Process the Quantity and Quality of User Behavior Data in Recommender Systems
    use it to apply noise filtering to a certain dataset of ratings


    Description: 
        1- we measure the coherence of a user c(u)
        2- use the coherence to categorize users into heavy and easy groups
        3- use the RND formula to determine the rating noise degree using a threshold v
                v is dependant on the user group (heavy, medium, or light)
'''

import pandas as pd
import numpy as np
import swifter
import math
import time
import os

from NoiseFilter2.Helpers import Helpers
from helpers.dataset import get_config_data, load_ratings, load_items
from NoiseFilter2.Coherence import Coherence

'''
    Find the rating noise degree of every rating in the dataset.
    The user groups is used to determine the threshold value that's set in the rnd formula:
        th = 0.075 for heavy and medium users
        th = 0.05 for light user groups
'''
class Noise:

    def __init__(self, ratings, movies):
        genres_list = Helpers().get_genres(movies) 
        # 1- call Coherence class to group users in the dataset
        if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_2.csv')):
            print("computing user groups...")
            t0 = time.time()
            users_categories_details = Coherence().compute_user_groups(ratings, movies, genres_list).reset_index()
            users_categories = users_categories_details[['userId', 'user_group']]
            t1 = time.time()
            print(users_categories, t1-t0)
        else:
            print("loading pre-computed groups for: " + get_config_data()['dataset_name'])
            users_categories_details = pd.read_csv('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_groups_protocol_2.csv')
            users_categories = users_categories_details[['userId', 'user_group']]

        ## 2- Calculate the rating noise degree for every rating in the dataset
        if(not os.path.exists('NaturalNoise/output/' + get_config_data()['dataset_name'] + '_ratings_rnd_protocol_2.csv')):
            print("computing noise in the dataset...")
            self.evaluate_noise(ratings, movies, genres_list, users_categories)
        else:
            print("noise already calculated for the dataset: " + get_config_data()['dataset_name'])


    def evaluate_noise(self, ratings_df, movies_df, genres_list, users_categories):
        # ratings_full_set = ratings_df.set_index('itemId').join(movies_df.set_index('itemId'), how='left').reset_index()
        user_features_dict = {}
        appended_dataframes = []

        # loop over all the ratings in the dataset to calculate the RND of every one
        user_ids = users_categories.userId.tolist()

        for user_id in user_ids:
            # to calculate RND, we need to get the feature_avg_rating of the user of the rating (same as we did with coherence)
            target_user = ratings_df.loc[ratings_df['userId'] == user_id]
            target_user_full = target_user.set_index('itemId').join(movies_df.set_index('itemId'), how='left').reset_index()
            target_user_full = target_user_full.set_index('userId').join(users_categories.set_index('userId'), how='left').reset_index()

            f = np.vectorize(lambda haystack, needle: needle in haystack)

            for genre in genres_list:
                # find all rows that conaint the Genre "Action", "Drama", etc.
                target_feature = target_user_full[f(target_user_full['genres'], genre)]

                # check whether the genre exists in at least one item row to avoid division by zero:
                if not target_feature.empty:
                    user_feature_ratings = target_feature.rating.tolist()
                    feature_avg_rating = sum(user_feature_ratings) / len(user_feature_ratings)

                    user_features_dict[genre] = feature_avg_rating

            target_user_full['rating_rnd'] = target_user_full.swifter.apply \
                    (lambda x: self.compute_rnd(x.rating, x.genres, x.user_group, user_features_dict), axis=1)

            # since we are inside a loop, save the dataframe into a list of dataframe (for every user) then use pd.concat to cobine them all into one big dataframe
            appended_dataframes.append(target_user_full)
        
        noise_df = pd.concat(appended_dataframes)
        noise_df.to_csv(r'NaturalNoise/output/' + get_config_data()['dataset_name'] + '_user_ratings_nf2.csv', index=False)
        return noise_df

    def compute_rnd(self, rating, genres, user_group, user_features_dict):
        # if(user_group == 'HEUG' or user_group == 'HDUG' \
        #     or user_group == 'MEUG' or user_group == 'MDUG'):
        #     rnd_threshold = 0.075
        # else:
        #     rnd_threshold = 0.05

        rnd_threshold = 0.075
        total_feature_condition = 0

        try:
            genres_list = genres.split('|')
        except:
            genres_list = genres

        if not '(no genres listed)' in genres_list:
            for genre in genres_list:
                feature_formula = abs(rating - user_features_dict[genre])/user_features_dict[genre]
                feature_condition = 1 if feature_formula > rnd_threshold else 0
                total_feature_condition += feature_condition

            rnd = total_feature_condition/len(genres_list)
        else:
            rnd = 0

        return rnd


dataset_path = get_config_data()['dataset']
ratings = load_ratings(dataset_path)[['userId','movieId','rating']].rename({'movieId': 'itemId'}, axis=1)
items = load_items(dataset_path)[['movieId','title','genres']].rename({'movieId': 'itemId'}, axis=1)
Noise(ratings, items)
