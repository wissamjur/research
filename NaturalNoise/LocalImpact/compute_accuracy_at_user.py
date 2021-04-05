# custom script to calculate accuracy at the neighborhood level (user-level)
# input: surpriselib predictions (after training a model)

from collections import defaultdict
import pandas as pd

def compute_mae_at_user(predictions):

    mae_at_user = dict()
    user_est_true = defaultdict(list)

    # map the predictions to each user
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, true_r, est))

    for uid, user_ratings in user_est_true.items():
        # calculate the mae for every user in the testset
        mae = (sum(abs(est - true_r) for (_, true_r, est) in user_ratings)) / len(user_ratings)

        mae_at_user[uid] = mae

    mae_at_user_df = pd.DataFrame.from_dict(mae_at_user, orient='index').reset_index().sort_values(by=['index'])
    mae_at_user_df.to_csv('NaturalNoise/LocalImpact/output/local-eval.csv', index=False)