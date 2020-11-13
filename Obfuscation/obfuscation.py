from datetime import datetime as dt

class Optout:

    def get_opt_out_users(self, ratings_df):

        users = ratings_df.userId.drop_duplicates().to_list()
        optout_dict = {}

        for user in users:
            # get the max, min dates plus the total number of rating days
            user_df = ratings_df[ratings_df['userId'] == user]
            dates_list = user_df.timestamp.to_list()
            max_date = dt.fromtimestamp(max(dates_list))
            min_date = dt.fromtimestamp(min(dates_list))

            date_diff = (max_date - min_date).days

            optout_dict[user] = 0
            if date_diff >= 1:
                # get overall total noise + total noise in last day of rating
                total_noise = sum(user_df.noise.to_list())

                if total_noise > 0:
                    noise_last_day = sum(ratings_df[ratings_df['timestamp'] == max(dates_list)] \
                                    .noise.to_list())
                
                    if (noise_last_day / total_noise) >= 0.5:
                        optout_dict[user] = 1
                    else:
                        optout_dict[user] = 0

        opt_out_users = dict((k, v) for k, v in optout_dict.items() if v >= 1)

        return opt_out_users