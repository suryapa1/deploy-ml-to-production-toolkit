
# coding: utf-8

# ## Dynamic Inference:
# Inference is the term used to describe the process of using a pre-trained model to make predictions for unseen data.
# Dynamic Inference is the term used to describe making predictions on demand, using a server. 
# 
# The tutorial below demonstrates how to serve our Lending Club model trained earlier using a low latency prediction servering system called **clipper** ([docs](http://clipper.ai/), [implementation](https://github.com/ucbrise/clipper)). **clipper** can be hosted on your favorite cloud provider or on-prem.

# In[1]:


import logging, xgboost as xgb, numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
from datetime import datetime
import pickle
import time
import matplotlib.pyplot as plt
plt.show(block=True)

from clipper_admin import ClipperConnection, DockerContainerManager
clipper_conn = ClipperConnection(DockerContainerManager())
print("Start Clipper...")
clipper_conn.start_clipper()
print("Register Clipper application...")
clipper_conn.register_application('xgboost-airlines', 'doubles', 'default_pred', 100000)


# In[17]:


training_examples = pd.read_pickle("../data/processed/airlines_training_examples.pkl")
f1=open("../data/processed/airlines_training_targets.pkl",'rb')
training_targets = pickle.load(f1) 
f1.close()
test_examples = pd.read_pickle("../data/processed/airlines_test_examples.pkl")

def get_train_points():
     return training_examples.values.tolist()

def get_test_points(start_row_index,end_row_index):
    return test_examples.iloc[start_row_index:end_row_index].values.tolist()

def get_test_point(row_index):
     return test_examples.iloc[row_index].tolist()


# In[18]:


# Create a training matrix.
dtrain = xgb.DMatrix(get_train_points(), label=training_targets)
# We then create parameters, watchlist, and specify the number of rounds
# This is code that we use to build our XGBoost Model, and your code may differ.
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)


# In[19]:


def predict(xs):
    result = bst.predict(xgb.DMatrix(xs))
    return result 
# make predictions
predictions = predict(test_examples.values)
print("Predict instances in test set using custom defined scoring function...")
predictions


# In[7]:


from clipper_admin.deployers import python as python_deployer
# We specify which packages to install in the pkgs_to_install arg.
# For example, if we wanted to install xgboost and psycopg2, we would use
# pkgs_to_install = ['xgboost', 'psycopg2']
print("Deploy predict function closure using Clipper...")
python_deployer.deploy_python_closure(clipper_conn, name='xgboost-model', version=1,
    input_type="doubles", func=predict, pkgs_to_install=['xgboost'])

time.sleep(5)

# In[8]:


print("Link Clipper connection to model application...")
clipper_conn.link_model_to_app('xgboost-airlines', 'xgboost-model')


# In[22]:


import requests, json
# Get Address
addr = clipper_conn.get_query_addr()
print("Model predict for a single instance via Python requests POST request & parse response...")

# Post Query
response = requests.post(
     "http://%s/%s/predict" % (addr, 'xgboost-airlines'),
     headers={"Content-type": "application/json"},
     data=json.dumps({
         'input': get_test_point(0)
     }))
result = response.json() 
print(result)


# In[23]:


# import requests, json, numpy as np
# print("Model predict for a single instance via Python requests POST request...")
# headers = {"Content-type": "application/json"}
# requests.post("http://localhost:1337/xgboost-airlines/predict", headers=headers, data=json.dumps({"input": get_test_point(0)})).json()


# # In[25]:


import requests, json, numpy as np
print("Model predict for a batch of instances via Python requests POST request...")
headers = {"Content-type": "application/json"}
response = requests.post("http://localhost:1337/xgboost-airlines/predict", headers=headers, data=json.dumps({"input_batch": get_test_points(0,2)})).json()
print(response)

# # In[27]:


# get_test_point(0)
# print("Model predict for a single instance via curl...")
# # get_ipython().system('curl -X POST --header "Content-Type:application/json" -d \'{"input": [16.0, 1995.0, 1.0, 1.0, 257.0, 1670.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}\' 127.0.0.1:1337/xgboost-airlines/predict')


# # If you want to get details...

# # In[ ]:


# # todo: insert link to clipper troubleshooting
# # clipper_conn.inspect_instance()
# # clipper_conn.get_clipper_logs()


# # In[2]:


print("Shutting down Clipper connection.")
clipper_conn.stop_all()
import sys
sys.exit

# # In[3]:


# # stop all containers:
# # get_ipython().system('docker rm $(docker ps -a -q)')


# # In[5]:


# # get_ipython().system('docker ps')


# # In[28]:


# # stop all containers:
# # docker kill $(docker ps -q)

# # remove all containers
# # !docker rm $(docker ps -a -q)

# # remove all docker images
# # docker rmi $(docker images -q)

