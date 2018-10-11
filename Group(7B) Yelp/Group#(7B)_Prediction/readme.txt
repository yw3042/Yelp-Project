Files preparation

Save all the code files in a same folder.
Download raw data set from Yelp Dataset Challenge (https://www.kaggle.com/yelp-dataset/yelp-dataset). This dataset contains seven CSV files. In our project, we use three of them: "yelp_business.csv", "yelp_user.csv", "yelp_review.csv". All of them should be saved in a parallel folder named "yelp-dataset" with code file.It is such a large data set, so we cannot submit together with the other files.

Code Usage
1. 
data_preparation.py convert raw data into the format can be put into further use. 
Several parameters are defined to extract target business and users. 
city_name defines the city we're currently dealing with. 
business_filter and customer_filter are used to define reviews requirment.
train_size defines the proportion of training set. It generates training matrix and test set and origin training set.

2. 
cf_method.py defines the baseline, memory-based and item-based collaborative filtering algorithms. 
Each algorithm is programmed into a function. 
Each function takes train data matrix, train data set and test data set as input. 
And the output contains RMSE and prediction error for rating larger than 4.

3. 
main.py is the main function of the first three method. 
When want to duplicate the result of first three method, we just need to run this file and it will automatically invoke the above two programs. 
And the outcome will be stored. 

4. 
ConjugateGradient.py runs the algorithm for collaborative filtering by conjugate gradient.
First read in the rating matrix X, then split the data into training and test sets randomly.
test_values_locations record the locations for test sample, with row_ids and col_ids storing the indices.
Loss function and its derivatives are defined in the function cost.
After setting the parameters, train the model by function minimize.
Predictions are stored in the variable predictions.
Training, test and baseline RMSE are stored in train_rmse, test_rmse and base_rmse respectively.

5.TextBased.py is the file for text based model. Simply run this file would run data_preparation.py first and then build the text based model and make prediction. The file would save the predicted values as ‘and’ and also save the training and test error as ‘RMSETrain’ and ‘RMSETest’. 

Group Member:
yaxin wang yw3042
Minmin Zhu mz2656
Jiaxi Wu   jw3588
Zonghao Li zl2613
Tianze Yue ty2369
