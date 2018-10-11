import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.metrics import pairwise
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from data_preparation import *

def base_line(data,test,train):
    data[data==0]=np.nan
    nuser = data.shape[0]
    nbusiness = data.shape[1]
    business_mean = data.mean(axis = 0)
    user_mean = data.mean(axis = 1)
    baseline = np.zeros((nuser,nbusiness))
    for i in range(nuser):
        for j in range(nbusiness):
            baseline[i,j] = (business_mean[j]+user_mean[i])/2
    baseline = DataFrame(baseline, columns = data.columns, index = data.index)
    baseline = round(baseline*2)/2
    prediction= []
    for line in train.itertuples():
        prediction.append(baseline.loc[line[1],line[2]])
    y = np.array(train.iloc[:,2])
    RMSE= []
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    prediction= []
    for line in test.itertuples():
        prediction.append(baseline.loc[line[1],line[2]])
    y = np.array(test.iloc[:,2])
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    success_rate = np.mean((np.array(prediction)>=4) == (y >=4))
    return([RMSE,success_rate])
            

def memory_based(data,test,train):
    data[data==0]=np.nan
    nuser = data.shape[0]
    nbusiness = data.shape[1]
    user_mean = data.mean(axis = 1)
    matrix = data.fillna(0)
    matrix = np.array(matrix)
    user_sim = pairwise.cosine_similarity(matrix)
    business_sim = pairwise.cosine_similarity(matrix.T)
    user_based = np.zeros((nuser,nbusiness))
    for i in range(nuser):
        for j in range(nbusiness):
            None_index = matrix[:,j]==0
            if sum(abs(user_sim[i,~None_index])) < 0.0000001:
                user_based[i,j] = user_mean[i]
            else:
                dif = matrix[:,j]-user_mean
                dif[None_index] = 0
                user_based[i,j] = user_mean[i] + np.dot(dif,user_sim[i,:])/sum(abs(user_sim[i,~None_index]))
    user_based_rating = DataFrame(user_based, columns = data.columns, index = data.index)
    user_based_rating[user_based_rating<0] = 0
    user_based_rating = round(user_based_rating*2)/2
    user_based_rating[user_based_rating >5] = 5
    prediction= []
    for line in train.itertuples():
        prediction.append(user_based_rating.loc[line[1],line[2]])
    y = np.array(train.iloc[:,2])
    RMSE = []
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    prediction= []
    for line in test.itertuples():
        prediction.append(user_based_rating.loc[line[1],line[2]])
    y = np.array(test.iloc[:,2])
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    success_rate = np.mean((np.array(prediction)>=4) == (y >=4))
    return([RMSE,success_rate])
    
    
def item_based(data,test,train):
    data[data==0]=np.nan
    nuser = data.shape[0]
    nbusiness = data.shape[1]
    user_mean = data.mean(axis = 1)

    matrix = data.fillna(0)
    matrix = np.array(matrix)
    business_sim = pairwise.cosine_similarity(matrix.T)
    item_based = np.zeros((nuser,nbusiness))
    for i in range(nuser):
        for j in range(nbusiness):
            None_index = matrix[i,:]==0
            item_based[i,j] = np.dot(matrix[i,:],business_sim[:,j])/sum(abs(business_sim[~None_index,j]))
    item_based_rating = DataFrame(item_based, columns = data.columns, index = data.index)
    item_based_rating[item_based_rating<0] = 0
    item_based_rating = round(item_based_rating*2)/2
    item_based_rating[item_based_rating >5] = 5
    prediction= []
    for line in train.itertuples():
        prediction.append(item_based_rating.loc[line[1],line[2]])
    y = np.array(train.iloc[:,2])
    RMSE =[]
    RMSE.append(np.sqrt(np.nanmean((np.array(prediction) - y)**2)))
    prediction= []
    for line in test.itertuples():
        prediction.append(item_based_rating.loc[line[1],line[2]])
    y = np.array(test.iloc[:,2])
    RMSE.append(np.sqrt(np.nanmean((np.array(prediction) - y)**2)))
    success_rate = np.mean((np.array(prediction)>=4) == (y >=4))
    return([RMSE,success_rate])


def text_cf(data, test,train, text):
    textData = text

    # Concate all reviews by each customer and each business
    userText = textData.groupby('user_id').text.sum()
    businessText = textData.groupby('business_id').text.sum()

    # Build a TfidVectorizer Model and Train the model using the previous data
    model = TfidfVectorizer(min_df = 100, stop_words = 'english')
    restaurantsVec = model.fit_transform(businessText)
    
    data[data==0]=np.nan
    nuser = data.shape[0]
    nbusiness = data.shape[1]
    matrix = data.fillna(0)
    matrix = np.array(matrix)
    business_sim = pairwise.cosine_similarity(restaurantsVec)
    item_based = np.zeros((nuser,nbusiness))
    for i in range(nuser):
        for j in range(nbusiness):
            None_index = matrix[i,:]==0
            item_based[i,j] = np.dot(matrix[i,:],business_sim[:,j])/sum(abs(business_sim[~None_index,j]))
    item_based_rating = DataFrame(item_based, columns = data.columns, index = data.index)
    item_based_rating[item_based_rating<0] = 0
    item_based_rating = round(item_based_rating*2)/2
    item_based_rating[item_based_rating >5] = 5
    prediction= []
    for line in train.itertuples():
        prediction.append(item_based_rating.loc[line[1],line[2]])
    y = np.array(train.iloc[:,2])
    RMSE =[]
    RMSE.append(np.sqrt(np.nanmean((np.array(prediction) - y)**2)))
    prediction= []
    for line in test.itertuples():
        prediction.append(item_based_rating.loc[line[1],line[2]])
    y = np.array(test.iloc[:,2])
    RMSE.append(np.sqrt(np.nanmean((np.array(prediction) - y)**2)))
    success_rate = np.mean((np.array(prediction)>=4) == (y >=4))
    return([RMSE,success_rate])
    
def text_knn(data, test,train, text):
    textData = text

# Concate all reviews by each customer and each business
    userText = textData.groupby('user_id').text.sum()
    businessText = textData.groupby('business_id').text.sum()
#    Build a TfidVectorizer Model and Train the model using the previous data
    model = TfidfVectorizer(min_df = 100, stop_words = 'english')
    restaurantsVec = model.fit_transform(businessText)
    userVec = model.transform(userText)
    data[data==0]=np.nan
    nuser = data.shape[0]
    nbusiness = data.shape[1]
    user_mean = data.mean(axis = 1)
    matrix = data.fillna(0)
    matrix = np.array(matrix)
    user_sim = pairwise.cosine_similarity(userVec)
    user_based = np.zeros((nuser,nbusiness))
    for i in range(nuser):
        for j in range(nbusiness):
            None_index = matrix[:,j]==0
            if sum(abs(user_sim[i,~None_index])) < 0.0000001:
                user_based[i,j] = user_mean[i]
            else:
                dif = matrix[:,j]-user_mean
                dif[None_index] = 0
                user_based[i,j] = user_mean[i] + np.dot(dif,user_sim[i,:])/sum(abs(user_sim[i,~None_index]))
    user_based_rating = DataFrame(user_based, columns = data.columns, index = data.index)
    user_based_rating[user_based_rating<0] = 0
    user_based_rating = round(user_based_rating*2)/2
    user_based_rating[user_based_rating >5] = 5
    prediction= []
    for line in train.itertuples():
        prediction.append(user_based_rating.loc[line[1],line[2]])
    y = np.array(train.iloc[:,2])
    RMSE = []
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    prediction= []
    for line in test.itertuples():
        prediction.append(user_based_rating.loc[line[1],line[2]])
    y = np.array(test.iloc[:,2])
    RMSE.append(np.sqrt(np.mean((np.array(prediction) - y)**2)))
    success_rate = np.mean((np.array(prediction)>=4) == (y >=4))
    return([RMSE,success_rate])



