import numpy as np
import pandas as pd
import scipy
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import linear_model

######## import data
from data_preparation import *
textData = text

# Concate all reviews by each customer and each business
userText = textData.groupby('user_id').text.sum()
businessText = textData.groupby('business_id').text.sum()

# Build a TfidVectorizer Model and Train the model using the previous data
model = TfidfVectorizer(min_df = 1, stop_words = 'english')
restaurantsVec = model.fit_transform(businessText)
userVec = model.transform(userText)

# Calculate cosine similarity
cosine_similarities = linear_kernel(userVec, restaurantsVec)
cosine = pd.DataFrame(cosine_similarities)

# Define the list of user_id and business_id
usersName = userText.index
businessName = businessText.index
cosine.index = userText.index
cosine.columns = businessText.index
testUserName = np.array(usersName)
testBusinessName = np.array(businessName)

# Here we create a new vector to store out prediction with rows as user_id and
# columns as business_id
ans = []

for thisUser in usersName:
    thisMean = np.nanmean(all_matrix.loc[thisUser,:])
    #print(thisUser)
    nUser = cosine.shape[0]
    nBusiness = cosine.shape[1]
    aRow = train_stars[train_stars.user_id==thisUser]
    lr = linear_model.LinearRegression()
    bus = aRow.business_id.values
    if (len(bus) <=0):
        ans.append(thisMean)

    else:
        x = np.array(cosine.loc[thisUser,bus]).reshape(len(bus),1)
        y = np.array(aRow.stars).reshape(len(aRow),1)
        lr.fit(x,y)
        pred = lr.predict( np.array(cosine.loc[thisUser,:]).reshape(nBusiness,1) )
        pred = np.maximum(1, np.minimum(pred, 5))
        #textMatrix.loc[thisUser,:] = pred
        ans.append(pred)



# Calculate test RMSE 

prediction= []
for line in test_stars.itertuples():
    userNumber = np.where(testUserName==line[1])[0][0]
    businessNumber = np.where(testBusinessName==line[2])[0][0]
    prediction.append(ans[userNumber][businessNumber])
y = np.array(test_stars.iloc[:,2])
RMSETest = np.sqrt(np.mean((np.array(prediction) - y)**2))


# Calculate traning RMSE
prediction= []
for line in train_stars.itertuples():
    userNumber = np.where(testUserName==line[1])[0][0]
    businessNumber = np.where(testBusinessName==line[2])[0][0]
    prediction.append(ans[userNumber][businessNumber])
y = np.array(train_stars.iloc[:,2])
RMSETrain = np.sqrt(np.mean((np.array(prediction) - y)**2))
