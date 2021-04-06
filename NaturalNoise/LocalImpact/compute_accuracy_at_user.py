# custom script to calculate accuracy at the neighborhood level (user-level)
# input: surpriselib predictions (after training a model)

from collections import defaultdict
import pandas as pd

def compute_mae_at_user(predictions, neighbors):

    mae_at_user = dict()
    user_est_true = defaultdict(list)

    # map the predictions to each user
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, true_r, est))

    for uid, user_ratings in list(user_est_true.items()): # had to add list() areound user_est_true.items() to avoid "dict changed size" error

        # if there's a neighborhood (k>1), append the ratings of the nieghbors in total_user_ratings
        for neighbor in neighbors[uid]:
            neighbor_ratings = user_est_true[neighbor]
            user_ratings += neighbor_ratings

        # calculate the mae for every user in the testset (neighborhood cetered at the user if the neighborhood list is not empty)
        mae = (sum(abs(est - true_r) for (_, true_r, est) in user_ratings)) / len(user_ratings)

        mae_at_user[uid] = mae

    mae_at_user_df = pd.DataFrame.from_dict(mae_at_user, orient='index').reset_index().sort_values(by=['index'])
    mae_at_user_df.to_csv('NaturalNoise/LocalImpact/output/local-eval.csv', index=False)