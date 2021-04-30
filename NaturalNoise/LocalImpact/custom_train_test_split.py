# Custom train-test split script to override the deafault Surprise function
# Modify testset and trainset to simulate local neighborhood effect after removing target ratings 
#       user 599 for ml-latest and user 5376 for ml-1m
# returns: trainset, testset (Surprise compatible)

import pandas as pd
from surprise import Dataset
from surprise import Reader
from datetime import datetime as dt

def mod_train_test_split(raw_dataset, remove_ratings=False):
    
    # testset_df = pd.read_csv('NaturalNoise/LocalImpact/testset_local_eval_ml_latest_small.csv')
    testset_df = pd.read_csv('NaturalNoise/LocalImpact/testset_local_eval_ml_1m.csv')

    testset_details_df = testset_df.merge(raw_dataset, how='inner', on=['userId', 'itemId'])
    
    # get the ratings of the target user (to be removed) from the testset
    # for ml-latest": user 599
    # tuser_testset_t_ratings = testset_details_df[ 
    #                                 (testset_details_df['userId'] == 599) 
    #                                 & ( 
    #                                     (testset_details_df['date'] == dt.date(dt(2017, 6, 27)))
    #                                     | (testset_details_df['date'] == dt.date(dt(2017, 6, 26)))
    #                                     | (testset_details_df['date'] == dt.date(dt(2018, 2, 20)))
    #                                     | (testset_details_df['date'] == dt.date(dt(2018, 2, 21)))
    #                                     )
    #                             ]

    tuser_testset_t_ratings = testset_details_df[ 
                                    (testset_details_df['userId'] == 387) 
                                    & ( 
                                        (testset_details_df['date'] == dt.date(dt(2004, 9, 12)))
                                        )
                                ]

    testset_df_tuser_update = testset_details_df.drop(tuser_testset_t_ratings.index.to_list()).rename({'rating_x': 'rating'}, axis=1)
    testset = list(testset_df_tuser_update[['userId', 'itemId', 'rating']].to_records(index=False))

    # define the trainset: raw_dataset - testset (updated one)
    trainset_df = pd.merge(raw_dataset, testset_df_tuser_update[['userId', 'itemId', 'rating','timestamp','date']], how='outer', indicator=True)\
        .query('_merge == "left_only"')\
        .drop('_merge', 1)

    # temp code (remove user 599 target ratings from the trainset_df) This is done to measure the local-eval-after
    # we did 3 versions of the removal (1- removed all target dates below, then decreased the number of removed ratings in the peak selected days)
    if remove_ratings:

        # for ml-latest": user 599
        # ttrainset_df_tuser_t_removed = trainset_df[~(
        #                                     (trainset_df['userId'] == 599) & (
        #                                         (trainset_df['date'] == dt.date(dt(2017, 6, 27)))
        #                                         | (trainset_df['date'] == dt.date(dt(2017, 6, 26)))
        #                                         | (trainset_df['date'] == dt.date(dt(2018, 2, 20)))
        #                                         | (trainset_df['date'] == dt.date(dt(2018, 2, 21)))
        #                                     )
        #                                 )]

        # for ml-latest": user 5376
        ttrainset_df_tuser_t_removed = trainset_df[~(
                                            (trainset_df['userId'] == 387) & (
                                                (trainset_df['date'] == dt.date(dt(2004, 9, 12)))
                                            )
                                        )]

        # ttrainset_df_tuser_t_removed was initially trainset_df -- original trainset
        train_data = Dataset.load_from_df(ttrainset_df_tuser_t_removed[['userId', 'itemId', 'rating']], Reader(rating_scale=(1,5)))
        trainset = train_data.build_full_trainset()
    else:
        train_data = Dataset.load_from_df(trainset_df[['userId', 'itemId', 'rating']], Reader(rating_scale=(1,5)))
        trainset = train_data.build_full_trainset()

    return trainset, testset