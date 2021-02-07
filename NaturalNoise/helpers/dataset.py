'''
    Helper functions
    ----------------
'''
import numpy as np, csv
import pandas as pd



'''
    function that loads a recommender dataset
    @param folder location
    @return dictionary
'''
def load_ratings(path):
    ratings_path = path + '/ratings.csv'

    ratings = pd.read_csv(ratings_path)

    return ratings

def load_items(path):
    itens_path = path + '/movies.csv'

    items = pd.read_csv(itens_path)

    return items

'''
    function that retrieves the main dataset files location from the config file
'''
def get_config_data():
    myObject = {}
    with open("config.txt") as f:
        for line in f.readlines():
            key, value = line.rstrip("\n").split("=")
            if(not key in myObject):
                myObject[key] = value
            else:
                print("Duplicate assignment of key '%s'" % key)

    return myObject