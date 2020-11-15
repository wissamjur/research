'''
This script measures the effect of an ofuscation formula that was purely developed by us. Rather than relying on the noise 
algorithm to detect an opt-out user (as we did in our previous test cases 1 and 2), we will be setting our own metrics
and cases for an opt-out user.

The metrics:
    1- Have a reasonable amount of rating days (at least 20 rating days)
    2- Have a significant amount of ratings (huge spike) in the last two days as opposed to the initial 10 to 15 days
    3- The neighbors of the items on the "spike day" (list called z) are do not match the highly rated items in the initial days:
        matches-found / len[z] < 0.2

        where matches-found is the number of matches between the neighbors of the "spike day" items and the highly rated (>=4)
        items of the initial days. len[z] is the total number of neighbors for all the items of the spike day

After figuring out the opt-out-user based on our formulation above, we run the same scenarios as in the previous test cases.
'''

import pandas as pd
from datetime import datetime as dt

ratings_path = '../Research-old/datasets/ml_1m/ratings.csv'

# load in the dataset and change the timestamps to dates
ratings_df = pd.read_csv(ratings_path).rename({'movieId': 'itemId'}, axis=1)
ratings_df['date'] = ratings_df['timestamp'].apply(lambda x: dt.fromtimestamp(x).date())

# get the total ratings per user and the total days for every user
ratings_per_day = ratings_df.groupby(['userId','date']).size().reset_index(name="ratings_per_day")
days_per_user = ratings_per_day.groupby(['userId']).size().reset_index(name="days_per_user")
# select only the users that have at least 20 rating days (metric 1)
m1 = days_per_user[days_per_user['days_per_user'] > 20].userId.to_list()
users_m1 = ratings_per_day[ratings_per_day['userId'].isin(m1)]

# further, from user_m1 filter out only those that have a sum of ratings in the last 3 days that's more than the avergae 
# ratings per day in the initial days days (metric 2)
users_m2 = []
for user in users_m1.userId.drop_duplicates().to_list():
    x = users_m1[users_m1['userId'] == user]
    last_days = x[x['date'].isin(x.date.tail(3).to_list())]
    first_days = x[x['date'].isin(x.date.head(len(x)-3).to_list())]

    if( (last_days.ratings_per_day.sum()) > (first_days.ratings_per_day.max()) ):
        users_m2.append(user)
