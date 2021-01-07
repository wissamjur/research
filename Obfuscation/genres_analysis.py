
from collections import OrderedDict 

def group_list(lst): 
  res =  [(el, lst.count(el)) for el in lst] 
  return list(OrderedDict(res).items())


user_276_l1 = ['Documentary', 'Adventure', 'Animation', 'Comedy', 'Comedy',	'Drama', 'Musical',	'Romance', 'Drama',	
                'War', 'Documentary', 'Drama', 'Comedy', 'Drama', 'Romance', 'Film-Noir', 'Romance', 'Thriller',
                'Drama', 'Romance',	'Drama', 'Western']		

user_276_l2 = ['Crime', 'Drama', 'Thriller', 'Documentary', 'Drama', 'Romance', 'Drama', 'Adventure', 'Animation', 'Comedy',
                'Comedy', 'Drama', 'Musical', 'Romance', 'Comedy', 'Documentary', 'Comedy', 'Drama', 'Crime',
                'Film-Noir', 'Mystery', 'Thriller']

user_154_l1 = ['Crime', 'Drama', 'Documentary', 'Adventure', 'Animation', 'Comedy', 'Drama', 'War', 'Comedy', 'Drama',
                'Comedy', 'Drama', 'Romance', 'Comedy', 'Musical', 'Romance', 'Mystery', 'Thriller', 
                'Film-Noir', 'Romance', 'Thriller']

user_154_l2 = ['Crime', 'Drama', 'Documentary', 'Adventure', 'Animation', 'Comedy', 'Drama', 'War', 'Comedy', 
                'Musical', 'Romance', 'Drama', 'Western', 'Drama', 'Romance', 'Action', 'Adventure', 'Drama', 
                'Crime', 'Drama', 'Thriller', 'Comedy']

print(group_list(user_154_l2))

