from collections import defaultdict
import pandas as pd
import math

def compute_ndcg_at_user(testset, predictions, n=10):

    dcg_at_user = dict()
    idcg_at_user = dict()
    map_users = defaultdict(list)
    top_n = defaultdict(list)
    top_n_real = defaultdict(list)

    # map the predictions to each user
    for uid, iid, true_r, est, _ in predictions:
        map_users[uid].append((iid, true_r, est))

    # Then sort the predictions for each user and retrieve the k highest ones
    # This will give us the detailed prediction data for every user already sorted in a top-n manner
    for uid, user_ratings in map_users.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True) # sort by order of highest prediction
        top_n[uid] = user_ratings[:n]

    for uid, user_ratings in map_users.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True) # sort by order of highest prediction
        top_n_real[uid] = user_ratings[:n]

    for uid, user_ratings in top_n.items():

        dcg = 0
        idcg = 0

        i = 1 # used to define the index (rank of the item in the list of recommended items)
        for (iid, true_r, est) in user_ratings:

            # DCG calculation
            utility = pow(2, true_r) - 1
            discount_factor = math.log2(i + 1)

            dcg += utility / discount_factor
            i += 1

            # IDCG calculation
            search_top_n_real = [index + 1 for index, v in enumerate(top_n_real[uid]) if v[0] == iid] # find the location of the item in the ground-truth rankings
            if not search_top_n_real:
                # print("item not found in the ground-truth rankings, using lowest values instead")
                search_top_n_real = n + 1
                utility = 1
                item_index_in_gt = search_top_n_real
            else:
                item_index_in_gt = search_top_n_real[0]

            discount_factor_real = math.log2(item_index_in_gt + 1) # an additional one is added since the search_top_n_real index starts at 0
            idcg += utility / discount_factor_real

        dcg_at_user[uid] = dcg
        idcg_at_user[uid] = idcg

    dcg_at_user_df = pd.DataFrame.from_dict(dcg_at_user, orient='index').reset_index().sort_values(by=['index'])
    idcg_at_user_df = pd.DataFrame.from_dict(idcg_at_user, orient='index').reset_index().sort_values(by=['index'])
    dcg_at_user_df.to_csv('NaturalNoise/LocalImpact/output/dcg_at_user.csv', index=False)
    idcg_at_user_df.to_csv('NaturalNoise/LocalImpact/output/idcg_at_user.csv', index=False)