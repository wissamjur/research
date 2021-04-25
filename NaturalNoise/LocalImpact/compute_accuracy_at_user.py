# custom script to calculate accuracy at the neighborhood level (user-level)
# input: surpriselib predictions (after training a model)

from collections import defaultdict
import pandas as pd
import numpy as np

def compute_mae_at_user(predictions, neighbors):

    mae_at_user = dict()
    map_users = defaultdict(list)

    # map the predictions to each user
    for uid, iid, true_r, est, _ in predictions:
        map_users[uid].append((iid, true_r, est))

    # append the neighborhood ratings to every user and calculate the mae at each iteration
    for uid, user_ratings in list(map_users.items()):

        # work on a copy of the user_ratings list rather to avoid overwriting it
        # the copy won't work unless we use list.copy() method
        user_neighbors_ratings = user_ratings.copy()

        for neighbor in neighbors[uid]:
            neighbor_ratings = map_users[neighbor]
            user_neighbors_ratings.extend(neighbor_ratings)

        # calculate the mae for every user in the testset (neighborhood cetered at the user if the neighborhood list is not empty)
        mae = np.mean([float(abs(true_r - est))
                    for (_, true_r, est) in user_neighbors_ratings])
                    
        mae_at_user[uid] = mae

    mae_at_user_df = pd.DataFrame.from_dict(mae_at_user, orient='index').reset_index().sort_values(by=['index'])
    mae_at_user_df.to_csv('NaturalNoise/LocalImpact/output/local-eval.csv', index=False)