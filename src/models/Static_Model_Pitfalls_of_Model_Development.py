
# coding: utf-8

# * Build baseline model
# * Example of leaky variables:
# * Missing Data
# * Example of New categorical variables
# * Features not available in production, only in training
# * Outliers
# * Blacklist variables
#
#
# * Example of overfitting
# * Multi-collinearity of variables in linear & NN models
# * Variance Analysis
# * Feature Engineering transformation compute time

# ## Define Business Objective
#
# Help an airline carrier & it's passengers know whether a flight will be delayed 24 hours in advance.
#
# ## Define Success
#
# Manual or automated decision making system?
#
# How to represent expected output (True: flight delayed / False: flight not delayed)?
#
# ## Define Cost of Errors (Bad Prediction)?
#
# ![alt text](../assets/images/Flight_Confusion_Matrix.png)

# ## Airline On-Time Performance Data Dictionary
#
# |Column   | Description  | Type  |Questions/Comments |
# |:-:|---|---|---|
# | Year  | year of the flight  | Integer  |    |
# | Month  | month of the flight | Integer  |    |
# | DayofMonth  | day of the month (1 to 31)  | Integer  |    |
# | DayOfWeek  | day of the week  |  Integer |    |
# | DepTime  | actual departure time |  Float | Is this available 24 hours prior to departure (i.e. time of prediction)?   |
# | CRSDepTime  | scheduled departure time  | Integer  | Is this available 24 hours prior to departure (i.e. time of prediction)?   |
# | ArrTime  | actual arrival time   | Float  |  Is this info available during time of prediction?     |
# | CRSArrTime  | scheduled arrival time   | Integer  |  Is this info available during time of prediction?  How likely is it to change?    |
# | UniqueCarrier  | carrier ID  | String  |  Why would this matter?  |
# | FlightNum  | flight number   |  Integer |  How are flight numbers assigned?  |
# | TailNum  | plane's tail number   | String  |  How are tail numbers assigned & why would that matter? What happens if this plane is decomissioned?   |
# | ActualElapsedTime  | actual elapsed time of the flight, in minutes   | Float  | Is this info available during time of prediction?  What happens if we include this variable in the model?   |
# | CRSElapsedTime  | scheduled elapsed time of the flight, in minutes   |  Float |   Is this info available during time of prediction?  How likely is it to change?  |
# | AirTime  | airborne time for the flight, in minutes   | Float  |   Is this info available during time of prediction?   |
# | ArrDelay  | arrival delay, in minutes  |  Float |   Is this info available during time of prediction?   |
# | DepDelay  |  departure delay, in minutes   | Float  |  Is this info available during time of prediction?    |
# | Origin  | originating airport    | String   |  How likely is this to change?   |
# | Dest  | destination airport  | String  |  How likely is this to change?   |
# | Distance  | flight distance    | Float  |  How likely is this to change?  |
# | TaxiIn  | taxi time from wheels down to arrival at the gate, in minutes    |  Float |  Is this info available during time of prediction?  |
# | TaxiOut  | taxi time from departure from the gate to wheels up, in minutes  | Float  | Is this info available during time of prediction?   |
# | Cancelled  | cancellation status (stored as logical).   | Integer  | Should we bother predicting whether flight is delayed or not for a cancelled flight?   |
# | CancellationCode  | cancellation code, if applicable   | String  | Should we bother predicting whether flight is delayed or not for a cancelled flight?     |
# | Diverted  | diversion status  | Integer   | Is this info available during time of prediction?    |
# | CarrierDelay  | delay, in minutes, attributable to the carrier   |  Float |    |
# | WeatherDelay  | delay, in minutes, attributable to weather factors  | Float  |  Weather predictions are available 24 hour in advance. Will you still include this variable if the model is expected run 48 hours instead of 24 hours in advance? How about if model expected to run 4 hours instead of 24 hours in advance? |
# | NASDelay  | delay, in minutes, attributable to the National Aviation System   | Float  | How far in advance do we know about national aviation delays? Consult domain expert.    |
# | SecurityDelay  | delay, in minutes, attributable to security factors    | Float  | How far in advance do we know about security delays? Consult domain expert.   |
# | LateAircraftDelay  | delay, in minutes, attributable to late-arriving aircraft  | Float  |   How far in advance do we know about security delays? Consult domain expert.  |
# | IsArrDelayed  | represents whether flight arrival was delayed or not  | String  |  How was this generated? How is delayed define (in terms of mins)? Should you trust this? |
# | IsDepDelayed  | represents whether flight departure was delayed or not  | String  | How was this generated? How is delayed define (in terms of mins)? Should you trust this?   |
#
#
# *note*: Determine what unit time is representd in? Local (PST, CT, EST) or Universal (UTC)? If not universal, we'll have to normalize time to a universal standard.

# ### Variables Not to be used for training a ML model: todo
# Not all variables available in the dataset should be used during training. Here is a list of questions to help you figure out which variables to exclude from the training production.
#
#
# 1. Is the variable available during time of inference (i.e. production prediction)? You'll want to first know when you'll be making a prediction?
#  1. Do you know if a plane will arrive late prior to taking off?
#
#
# 2. In some regulated industries, some variables are illegal to use for predictive modeling.
#  1. For example, personally identifiable information (PII) is one such example.
#
#
# 3. How likely is the variable available in production?
#  1. Determine a threshold for how available you expect a variable to be available during time of inference and remove variables which exceed that threshold.

# ## Supervised Learning Pipeline
# Here is a general end to end pipeline for a data science project.
#
# 1. Define Business Objective & Criteria for Success
#     + Experimental Design
#         + Identify the business/product objective
#         + Identify & hypothesize goals and criteria for success
#         + Create a set of questions for identifying correct data set
#         + Define which machine learning evaluation metric will be used to quantify quality of predictions
#         + Identify data sources, time window of data collected, data formats, data dictionary, features, target & evaluation metric
# 2. Data Aquisition
#     + Define what/how much data we need, where it lives, what format it's in & load dataset
#     + Import data from local or remote data source & determine the most approperiate tools to work with the data
#         + Pandas has functions for common open source data formats including data base connectors for MySQL & PostgreSQL
#         + Use Spark for Big Data
#     + Gather/Read any documentation available for the data (schema, data dictionary)
#     + Load and pre-process the data into a representation which is ready for model training
#         + If the data is available in an open source data format (JSON, CSV, XML, EXCEL), you'll be able to leverage open source tools
#         + If the data is available in a closed source format(fixed formatted row) then you will need to develop a parser to format the data into approperiate columns
#         + Ensure correct data types are imputed
#         + Look at the values. Ensure they make sense in the context of each column
#         + Look for missing/empty values
#         + For categorical fields, what are the unique values in the field?
#         + For numeric fields, are all values numbers?
#         + Split-out validation dataset
# 3. Exploratory Data Analysis
#     + Gather insights by using exploratory methods, descriptive & inferential statistics
#         + Find median, mode, std dev, min, max, average for each column. Do these make sense in the context of the column?
#         + Do financial values have reasonable upper bounds?
#         + Univariate feature distributions (to observe stability & other patterns of a given feature like skew)
#         + Feature & target correlations
#         + Target analysis (plot of feature vs target)
#         + Are there any outliers?
#         + Do the column values seem to follow a normal distribution? Uniform? Exponential (i.e. long tail)? If exponential, taking log(X) may be beneficial for linear regerssion.
# 4. Feature Engineering
#     + Perform feature scaling / normalization
#     + Inject domain knowledge (structure) into the data by adding or modifying existing columns
#         + Linear combinations of two or more features (ratios or other arithmetic variations)
#         + Adding new columns for day of year, hour of day from a datetime column
#     + Convert categorical data into numerical values using one-hot encoding
# 5. Feature Selection
#     + Drop highly correlated features (see correlation section above)
#     + PCA
#     + Recusive Feature Elimination
#     + Regularization method using LASSO
# 6. Select, build & evaluate the model
#     + Establish a baseline model for comparison
#     + Spot Check & Compare Algorithms
#     + Run a spot check of single model performance & tune the top 3 best performing learners
#         + Evaluate Algorithms with Standardization
#         + Improve accuracy
#     + You may generally find ensemble Methods (such as Bagging and Boosting, Gradient Boosting) to be quite useful
# 7. Refine the model (Hyper-parameter tuning)
#     + Use GridSearch to search & tune hyper-parameters
# 9. Finalize Model (use all training data and confirm using validation dataset)
#     + Save model binary along with model training results
#     + Predictions on validation dataset
# 10. Communicate the results
#     + Summarize findings with narrative, storytelling techniques
#     + Present limitations, assumptions of your analysis
#     + Identify follow-up problems and questions for future analysis

# In[1]:


#load libraries
from __future__ import print_function

import math
import numpy as np
from IPython import display
print('numpy: {}'.format(np.__version__))

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
print('pandas: {}'.format(pd.__version__))

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

print('sklearn: {}'.format(sklearn.__version__))

import xgboost as xgb
from xgboost import plot_importance
print('xgboost: {}'.format(xgb.__version__))

import joblib
import pickle

pd.options.display.max_rows = 40
pd.options.display.float_format = '{:.1f}'.format

seed = 7
OUTPUT_DIR="../data/processed/"


# In[2]:


# load data
def load_data(location, data_format="csv"):
    if(data_format=="csv"):
        df = pd.read_csv(location, encoding="ISO-8859-1", low_memory=False)
        df = df.reindex(
            np.random.permutation(df.index))
    else:
        print("{} format not currently supported".format(data_format))
    return df

airlines_df = load_data("https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv")

# preview data
airlines_df.head()


# Observe columns available in the dataset...

# In[3]:


airlines_df.columns


# In[4]:


airlines_df.describe()


# In[5]:


# dataset size
airlines_df.shape


# #### Target Analysis
# Check if any instances don't contain a label...

# In[6]:


airlines_df["IsDepDelayed"].isnull().sum()


# In[7]:


airlines_df["IsDepDelayed"].value_counts()


# In[8]:


y = airlines_df.IsDepDelayed
y.head()


# In[9]:


cols_not_to_use = ["DepTime", "ArrTime", "TailNum", "ActualElapsedTime", "AirTime", "ArrDelay",
                   "DepDelay", "TaxiIn", "TaxiOut",  "CancellationCode", "Diverted", "CarrierDelay",
                   "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "IsArrDelayed", "IsDepDelayed"]

cols_to_use = ["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime",
                                          "CRSArrTime", "UniqueCarrier", "FlightNum", "CRSElapsedTime",
                                          "Origin", "Dest", "Distance", "Cancelled"]

assert(len(cols_not_to_use) + len(cols_to_use) == airlines_df.shape[1])

def get_training_data(df, cols_not_to_use):
    print("\nGet Training Data...")
    print("Original shape: {}".format(df.shape))
    df = df.drop(cols_not_to_use, axis=1, errors='ignore')
    print("After columns dropped shape: {}".format(df.shape))
    return df

def label_encode_target(df, target):
    print("\nLabel Encode Target into Integers...")
    # encode string class values as integers
    y = df[target]
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    label_encoded_y = label_encoder.transform(y)
    return label_encoded_y

def naive_one_hot_encode(df, cols_to_encode=[]):
    print("\nNaive One-Hot-Encode for features: {}".format(cols_to_encode))
    print("\nTotal number of features before encoding: {}".format(df.shape[1]))
    for col in cols_to_encode:
        # use pd.concat to join the new columns with your original dataframe
        df = pd.concat([df,pd.get_dummies(df[col], prefix=col+"_")],axis=1)
        df = df.drop(col,axis=1)
    print("\nTotal number of features after encoding: {}".format(df.shape[1]))
    return df


# In[10]:


# this method call is not idempotent (can't delete target more than once)
label_encoded_y = label_encode_target(airlines_df, "IsDepDelayed")

X = get_training_data(airlines_df, cols_not_to_use)
assert(len(X.columns) == len(cols_to_use))

X = naive_one_hot_encode(X, ['UniqueCarrier','Dest','Origin'])

# train / test split
training_examples, test_examples, training_targets, test_targets = train_test_split(X, label_encoded_y, test_size=0.30)


# In[11]:


columns_of_interest = ["DayofMonth", "Year", "DayOfWeek", "Month", "Distance", "FlightNum", "Origin", "Dest", "UniqueCarrier"]
X = airlines_df[columns_of_interest]
# use pd.concat to join the new columns with your original dataframe
X = pd.concat([X,pd.get_dummies(X['UniqueCarrier'], prefix='carrier_')],axis=1)
X = pd.concat([X,pd.get_dummies(X['Dest'], prefix='dest_')],axis=1)
X = pd.concat([X,pd.get_dummies(X['Origin'], prefix='origin_')],axis=1)
X.drop(['UniqueCarrier','Dest','Origin'],axis=1, inplace=True)
training_examples, test_examples, training_targets, test_targets = train_test_split(X, label_encoded_y, test_size=0.30)


# In[12]:


X.shape


# In[13]:


# no non-numeric columns
#pd.set_option('display.max_rows', 40)
X.dtypes


# In[14]:


# check how many missing values each feature contains
training_examples.isnull().sum(axis = 0)


# In[15]:


def train(raw_df, target):
    # Create a training matrix.
    label_encoded_y = label_encode_target(raw_df, target)
    X = get_training_data(airlines_df, cols_not_to_use)
    assert(len(X.columns) == len(cols_to_use))
    X = naive_one_hot_encode(X, ['UniqueCarrier','Dest','Origin'])
    training_examples, test_examples, training_targets, test_targets = train_test_split(X, label_encoded_y, test_size=0.30)
    dtrain = xgb.DMatrix(training_examples, training_targets)
    param = {'objective': 'binary:logistic', 'seed':0,'nround': 1000,
              'max_depth': 5, 'eta': 0.01, 'subsample': 0.5,
              'min_child_weight': 1}
    watchlist = [(dtrain, 'train')]
    num_round = 2
    bst = xgb.train(param, dtrain, num_round, watchlist)
    print("\nFeature Importances...")
    importances = bst.get_fscore()
    print("\nFeature Importances {}".format(importances))
    return (bst, training_examples, test_examples, training_targets, test_targets)

model, training_examples, test_examples, training_targets, test_targets = train(airlines_df, "IsDepDelayed")
xgb.plot_importance(model);


# In[16]:


def predict(model, test_examples, test_targets):
    # XGBoost outputs probabilities by default and not actual class labels. To calculate accuracy we
    # need to convert these to a 0/1 label. We will set 0.5 probability as our threshold.
    predictions = model.predict(xgb.DMatrix(test_examples))
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    accuracy = accuracy_score(test_targets, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("\nConfusion Matrix...")
    print(confusion_matrix(test_targets, predictions))
    print("\nClassification Report...")
    print(classification_report(test_targets, predictions))
    return predictions

predictions = predict(model, test_examples, test_targets)


# Binary Classification Evaluation Metrics todo
# add cv
# https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/
# https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/

# In[17]:


def persist_model(model, dataset_name, training_examples, test_examples, training_targets, test_targets):
    import os
    # save model artifacts
    cwd = os.getcwd()
    train_X = OUTPUT_DIR + dataset_name + "_training_examples.pkl"
    train_y = OUTPUT_DIR + dataset_name + "_training_targets.pkl"
    test_X = OUTPUT_DIR + dataset_name + "_test_examples.pkl"
    test_y = OUTPUT_DIR + dataset_name + "_test_targets.pkl"
    joblib.dump(model, OUTPUT_DIR + "model_xgboost")
    training_examples.to_pickle(train_X)
    print("Finished saving training design matrix: {}".format(train_X))

    f1=open(train_y,'wb')
    pickle.dump(training_targets,f1)
    print("Finished saving training targets: {}".format(train_y))

    test_examples.to_pickle(test_X)
    print("Finished saving test design matrix: {}".format(train_X))

    f2=open(test_y,'wb')
    pickle.dump(test_targets, f2)
    print("Finished saving test targets: {}".format(train_y))

    f1.close()
    f2.close()


# In[18]:


persist_model(model, "airlines", training_examples, test_examples, training_targets, test_targets)
print("Pipeline Completed")


# ## Overfitting Example
# You have 2 ways to control overfitting in XGBoost:
#
# 1. Early stopping is an approach to training complex machine learning models to avoid overfitting.
# > It works by monitoring the performance of the model that is being trained on a separate test dataset and stopping the training procedure once the performance on the test dataset has not improved after a fixed number of training iterations.
#
# > It avoids overfitting by attempting to automatically select the inflection point where performance on the test dataset starts to decrease while performance on the training dataset continues to improve as the model starts to overfit.
#
# https://xgboost.readthedocs.io/en/latest//parameter.html
#
# ```
# max_depth : int
#     Maximum tree depth for base learners.
# learning_rate : float
#     Boosting learning rate (XGBoost's "eta")
# n_estimators : int
#     Number of boosted trees to fit.
# silent : boolean
#     Whether to print messages while running boosting.
# objective : string
#     Specify the learning task and the corresponding learning objective.
# nthread : int
#     Number of parallel threads used to run XGBoost.
# gamma : float
#     Minimum loss reduction required to make a further partition
#     on a leaf node of the tree.
# min_child_weight : int
#     Minimum sum of instance weight(hessian) needed in a child.
# max_delta_step : int
#     Maximum delta step we allow each tree's weight estimation to be.
# subsample : float
#     Subsample ratio of the training instance.
# colsample_bytree : float
#     Subsample ratio of columns when constructing each tree.
# base_score:
#     The initial prediction score of all instances, global bias.
# seed : int
#     Random number seed.
# missing : float, optional
#     Value in the data which needs to be present as a missing value.
#     If None, defaults to np.nan.
# ```
