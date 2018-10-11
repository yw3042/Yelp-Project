import sys
import random
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

sys.path.append('../code')
baseDataDir= 'D:/CU/5293/group/yelp-dataset/'
review = pd.read_csv(baseDataDir+'yelp_review.csv')
business = pd.read_csv(baseDataDir+'yelp_business.csv')
user = pd.read_csv(baseDataDir + 'yelp_user.csv')
random.seed(1)


# Extract business data of Las Vegas
business1 = business[business.city == 'Las Vegas']

# Drop users and businesses which have less than 500 reviews
user1 = user[user.review_count > 500]
business1 = business1[business1.review_count > 500]
business1 = business1.groupby('state').filter(lambda r: len(r) > 20)

# Merger review, business and user
rev_biz_usr = pd.merge(pd.merge(review, business1, on='business_id'), user1, on='user_id')
col_drop = rev_biz_usr.columns.difference(['business_id', 'user_id', 'stars_x'])
rev_biz_usr.drop(col_drop, axis=1, inplace=True)

# Drop users and businesses with less than 20 reviews
for _ in range(8):
    rev_biz_usr = rev_biz_usr.groupby('business_id').filter(lambda r: len(r) >= 20)
    rev_biz_usr = rev_biz_usr.groupby('user_id').filter(lambda r: len(r) >= 20)
rev_biz_usr.stars_x = rev_biz_usr.stars_x.astype(int)
X = pd.pivot_table(rev_biz_usr, index='business_id', columns='user_id', values='stars_x')#, fill_value=0)

# Store the rating matrix 
X.to_csv('rating.csv')




###parameters setting
city_name = 'Las Vegas'
business_filter = 500
customer_filter = 500
train_size = 0.75

city_business = business[business.city == city_name]
cat = np.array(city_business.categories,dtype=np.str)
food = np.empty(len(cat))
foodDescription = set(['Food','food','Restaurants','Cafe','bar','Bar','cafe','restaurants'])
for i in range(len(cat)):
    thisSet = set(cat[i].split(';'))
    union = len(thisSet.intersection(foodDescription))
    food[i] = union >=1
foodIndex = np.array(food,dtype=np.bool)
city_business = city_business[foodIndex]
CityFoodIndex = city_business[city_business.review_count >= business_filter].business_id

############ Use vegasFoodBusiness to filter review
CityFoodFilter = set(CityFoodIndex)
reviewIndex = review.business_id.isin(CityFoodIndex)
city_review = review[reviewIndex]

user_reduce = user[user.review_count >= customer_filter]
user_reduce = user_reduce[['user_id','review_count']]
city_data = pd.merge(city_review, user_reduce,on = 'user_id')
n_customer = len(set(city_data.user_id))
n_business = len(set(city_data.business_id))

###split train/test sets
###split train/test sets
n = city_data.shape[0]
train_index = np.zeros(n,dtype=bool)
train_index[random.sample(range(n),int(n*train_size))] = 1
test_index = np.logical_not(train_index)
stars = city_data[['user_id','business_id','stars']]
train_stars = stars[train_index]
test_stars = stars[test_index]
text = city_data[['user_id','business_id','text','stars']]

train_text = text[train_index]
test_text = text[test_index]
user_list = train_stars['user_id'].unique()
business_list = train_stars['business_id'].unique()
allUser_list = stars['user_id'].unique()
test_stars = test_stars.loc[test_stars['user_id'].isin(user_list),:]
test_stars = test_stars.loc[test_stars['business_id'].isin(business_list),:]

print("The proportion of test set is %f." % (test_stars.shape[0]/(test_stars.shape[0]+train_stars.shape[0])))

####Construct train matrix
n_user = len(user_list)
n_business = len(business_list)
train_matrix = np.zeros((n_user, n_business))
train_matrix = DataFrame(train_matrix, columns = business_list, index = user_list)
for line in train_stars.itertuples():
	train_matrix.loc[line[1],line[2]]= line[3]

all_matrix = np.zeros((n_customer, n_business))
all_matrix = DataFrame(all_matrix, columns = business_list, index = allUser_list)
for line in stars.itertuples():
	all_matrix.loc[line[1],line[2]]= line[3]

train_matrix.to_csv("train_matrix.csv")
test_stars.to_csv("test_stars.csv")
train_stars.to_csv("train_stars.csv")
train_text.to_csv('train_text.csv')
test_text.to_csv('test_stars.csv')


#######













