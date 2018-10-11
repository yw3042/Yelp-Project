
import numpy as np
# from scipy.sparse.linalg import svds
# from math import sqrt
# from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
# import utils
from scipy.optimize import minimize
from numpy import linalg as LA
import random 
random.seed(3)


X = pd.read_csv('rating.csv')
X.index = X.iloc[:,0]
X = X.iloc[:,1:]

# Split the data as training and test sets:
test_split_ratio = 0.3
test_size = int(X.shape[1] * test_split_ratio)
train_size = X.shape[1] - test_size
# rand_test_user_mask = random.sample(range(Y.shape[1]), test_size)
rand_column_mask = np.random.choice(X.shape[1], test_size, replace=False)
# Since number of reviews by each user is > 20, select 5 ratings as test per test user
X_test = X.iloc[:,rand_column_mask].copy()

for col in X_test:
    mask_size = 5
    mask = np.random.choice(X_test[col].notnull().nonzero()[0], mask_size, replace=False)
    X_test[col][mask] = np.nan

value_locations_premask = X.iloc[:,rand_column_mask][X.iloc[:,rand_column_mask].notnull()].stack().index.tolist()
value_locations_masked = X_test[X_test.notnull()].stack().index.tolist()
test_values_locations = list(set(value_locations_premask) - set(value_locations_masked))
# Convert to dataframe in order to be able to use during lookup operation, which requires list of rows, and list of columns
test_values_locations = pd.DataFrame.from_records(test_values_locations, columns=['business_id', 'user_id'])
# Get location positions:
query_rows = test_values_locations.business_id
rows = X.index.values
sidx = np.argsort(rows)
row_ids = sidx[np.searchsorted(rows,query_rows,sorter=sidx)]
# Get location positions:
query_cols = test_values_locations.user_id
cols = X.columns.values
sidx = np.argsort(cols)
col_ids = sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

X_test_original = X.iloc[:,rand_column_mask].copy()
X.iloc[:,rand_column_mask] = X_test.copy()

R = X.notnull()

# Define loss function
def cost(params, X, R, num_business, num_user, num_features, lamda):
    # lamda is the regularization coefficient lambda 
    # Convert the dataframe to ndarray, fill nans with zeros, and leave the answer array for easier linear algebra operations
#     Y_mat = np.nan_to_num(Y.as_matrix())
#     R_mat = np.nan_to_num(R.as_matrix())
    # unfold X and theta from the 1D params array
    P = np.reshape(params[:num_business*num_features], (num_business, num_features))
    Q = np.reshape(params[num_business*num_features:], (num_user, num_features))    
    
    J = 0.5*np.sum(pow((P@Q.T - X)*R,2)) + lamda/2*(np.sum(pow(Q,2)) + np.sum(pow(P,2)))
    
    P_grad = (P@Q.T - X)*R@Q + lamda*P
    Q_grad = (P@Q.T - X).T * R.T@P + lamda*Q
    
    grad = np.concatenate((np.ravel(P_grad), np.ravel(Q_grad)))
    print('The cost is currently equal to.........', J)
    return J, grad

# Parameters
num_features = 110
num_business = X.shape[0]
num_user = X.shape[1]    

n = 60
lamda = 10
P = np.random.randn(num_business, num_features)
Q = np.random.randn(num_user, num_features)
params = np.concatenate((np.ravel(P), np.ravel(Q)))

# Normalization
X_mat = X.as_matrix()
Xmean = np.nanmean(X_mat, axis=1, keepdims=True)

X_mat = np.nan_to_num(X_mat)
R_mat = np.nan_to_num(R.as_matrix())

Xnorm = np.nan_to_num(X.subtract(X.mean(axis=1), axis=0).as_matrix())

# Train the model

fmin = minimize(fun=cost, x0=params, args=(Xnorm, R_mat, num_business, num_user, num_features, lamda),  
                method='CG', jac=True, options={'maxiter': n})
P = np.matrix(np.reshape(fmin.x[:num_business * num_features], (num_business, num_features)))  
Q = np.matrix(np.reshape(fmin.x[num_business * num_features:], (num_user, num_features)))
predictions = P * Q.T + Xmean

a = pd.DataFrame(predictions - X)
a = a.fillna(0)
n = (a!=0).sum()
n = n.sum()

# Train RMSE
train_rmse = np.sqrt(LA.norm(a, 'fro')/n)

# Test RMSE
test_rmse = np.sqrt(np.mean(np.power((predictions[row_ids, col_ids].clip(1,5) - 
                  X_test_original.lookup(test_values_locations.business_id, test_values_locations.user_id)),2)))

nuser = X.shape[0]
nbusiness = X.shape[1]
business_mean = X.mean(axis = 0)
user_mean = X.mean(axis = 1)
baseline = np.zeros((nuser,nbusiness))
for i in range(nuser):
    for j in range(nbusiness):
        baseline[i,j] = (business_mean[j]+user_mean[i])/2
baseline = DataFrame(baseline, columns = X.columns, index = X.index)
baseline = round(baseline*2)/2
prediction= []
for line in test_values_locations.itertuples():
    prediction.append(baseline.loc[line[1],line[2]])
y = X_test_original.lookup(test_values_locations.business_id, test_values_locations.user_id)
# Baseline RMSE
base_rmse = np.sqrt(np.mean((np.array(prediction) - y)**2))

i = 0
plt.plot(predictions[np.argsort(np.ravel(predictions[:,i]))[::-1], i].clip(1,5))
plt.title('Collaborative Filtering by Conjugate Gradient Prediction')
plt.xlabel('User1');
plt.ylabel('Rating');
plt.show()



