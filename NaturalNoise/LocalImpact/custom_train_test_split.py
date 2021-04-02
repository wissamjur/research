# Custom train-test split script to override the deafault Surprise function
# Modify testset and trainset to simulate local neighborhood effect after removing target ratings (user 599 in our case)
# returns: trainset, testset (Surprise compatible)

import pandas as pd
from surprise import Dataset
from surprise import Reader
from datetime import datetime as dt

def train_test_split(raw_dataset):
    
    testset_df = pd.read_csv('NaturalNoise/LocalImpact/testset_local_eval.csv')

    testset_details_df = testset_df.merge(raw_dataset, how='inner', on=['userId', 'itemId'])
    # get the target ratings (to be removed) from the testset
    u599_testset_t_ratings = testset_details_df[ 
                                    (testset_details_df['userId'] == 599) 
                                    & ( 
                                        (testset_details_df['date'] == dt.date(dt(2017, 6, 27)))
                                        | (testset_details_df['date'] == dt.date(dt(2017, 6, 26)))
                                        | (testset_details_df['date'] == dt.date(dt(2018, 2, 20)))
                                        | (testset_details_df['date'] == dt.date(dt(2018, 2, 21)))
                                        )
                                ]
    testset_df_u599_update = testset_details_df.drop(u599_testset_t_ratings.index.to_list()).rename({'rating_x': 'rating'}, axis=1)
    testset = list(testset_df_u599_update[['userId', 'itemId', 'rating']].to_records(index=False))

    # define the trainset: raw_dataset - testset (updated one)
    trainset_df = pd.merge(raw_dataset, testset_df_u599_update[['userId', 'itemId', 'rating','timestamp','date']], how='outer', indicator=True)\
        .query('_merge == "left_only"')\
        .drop('_merge', 1)

    # temp code (remove user 599 target ratings from the trainset_df) This is done to measure the local-eval-after
    # we did 3 versions of the removal (1- removed all target dates below, then decreased the number of removed ratings in the peak selected days)
    trainset_df_u599_t_removed = trainset_df[~(
                                        (trainset_df['userId'] == 599) & (
                                            (trainset_df['date'] == dt.date(dt(2017, 6, 27)))
                                            # | (trainset_df['date'] == dt.date(dt(2017, 6, 26)))
                                            # | (trainset_df['date'] == dt.date(dt(2018, 2, 20)))
                                            # | (trainset_df['date'] == dt.date(dt(2018, 2, 21)))
                                        )
                                    )]
    # trainset_df_u599_t_removed was initially trainset_df -- original trainset
    new_train_data = Dataset.load_from_df(trainset_df_u599_t_removed[['userId', 'itemId', 'rating']], Reader(rating_scale=(1,5)))
    trainset = new_train_data.build_full_trainset()

    return trainset, testset