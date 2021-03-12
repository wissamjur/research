from surprise import SVD, KNNWithMeans, NMF
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from collections import defaultdict
import pandas as pd
import numpy as np

## import from custom scripts
from helpers.metrics import precision_recall_at_k
from noise_filter_1 import NoiseFilter1
from helpers.dataset import get_config_data, load_ratings
##

# call noise filter 1 to get the clusters and the noisy ratings it identifies
nf1 = NoiseFilter1()
dataset_path = get_config_data()['dataset']
raw_ratings_df = load_ratings(dataset_path)[['userId','movieId','rating','timestamp']].rename({'movieId': 'itemId'}, axis=1)
ratings_df_noise = nf1.get_dataset_with_noise(raw_ratings_df)
# load the ratings df without the noisy ratings that were identified by NF 1
ratings_df_without_noise = ratings_df_noise[ratings_df_noise['noise'] < 1]

### testing with recommender algorithms
data = Dataset.load_from_df(
                ratings_df_without_noise[['userId','itemId','rating']],
                Reader(rating_scale=(1,5))
            )
# sample random trainset and testset
trainset, testset = train_test_split(data, test_size=.15, shuffle=False)
df_test = pd.DataFrame(testset, columns=['userId','itemId','rating'])
df_test.to_csv('testset.csv')
print(df_test)

# load a recommender algorithm: SVD, NMF or KNNWithMeans algorithm
# algo = SVD()
# algo = NMF()
sim_options = {
    'name': 'pearson',
    'user_based': True
    }
algo = KNNWithMeans(sim_options=sim_options)

# train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# compute MAE and RMSE
accuracy.fcp(predictions)
accuracy.mae(predictions)
accuracy.rmse(predictions)
# 5-fold cross validation (to avoid bias)
# cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=5, verbose=True)

# comput Precision and Recall
# avg_precision = []
# avg_recall = []
# kf = KFold(n_splits=5)
# for trainset, testset in kf.split(data):
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

#     # Precision and recall can then be averaged over all users
#     avg_precision.append(round(sum(prec for prec in precisions.values()) / len(precisions), 4))
#     avg_recall.append(round(sum(rec for rec in recalls.values()) / len(recalls), 4))

# print("Precision: ", avg_precision)
# print("Recall: ", avg_recall)
### end testing

####################
# calculate accuracy at the neighborhood level
# map the predictions to each user.
mae_at_user = dict()
user_est_true = defaultdict(list)
for uid, iid, true_r, est, _ in predictions:
    user_est_true[uid].append((iid, true_r, est))

for uid, user_ratings in user_est_true.items():

    # calculate the mae for every user
    mae = (sum(abs(est - true_r) for (_, true_r, est) in user_ratings)) / len(user_ratings)

    mae_at_user[uid] = mae
    
mae_at_user_df = pd.DataFrame.from_dict(mae_at_user, orient='index').reset_index().sort_values(by=['index'])
# print(mae_at_user_df)
mae_at_user_df.to_csv('local-eval-before.csv', index=False)
######################