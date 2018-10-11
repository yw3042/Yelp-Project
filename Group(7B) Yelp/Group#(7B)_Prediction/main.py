# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:44:46 2018

@author: YTZzz
"""
import sys
sys.path.append('D:/CU/5293/group/submit/Group#(7A)_Prediction')
import pandas as pd
import cf_method as cf
from data_preparation import *

RMSE_baseline = cf.base_line(train_matrix, test_stars,train_stars)
RMSE_memory_based = cf.memory_based(train_matrix, test_stars, train_stars)
RMSE_item_based = cf.item_based(train_matrix, test_stars, train_stars)

RMSE_text_cf= cf.text_cf(train_matrix, test_stars, train_stars, train_text)
RMSE_text_knn = cf.text_knn(train_matrix, test_stars, train_stars, train_text)

