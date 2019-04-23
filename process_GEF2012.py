#merge and process the GEF2012 data and labels

import pandas as pd
import numpy as np

def process_GEF2012(data, labels):

    #merge data and labels for further processing
    data_labels_set=pd.merge(labels,data,left_on = ['wf','date'], right_on = ['wf','date_pred'], how = 'inner')

    #select the most actual weather prediction
    data_labels_set = pd.merge(data_labels_set.groupby(['wf','date']).agg({'date_iss':'max'}), data_labels_set, on = ['wf','date','date_iss'], how = 'inner')

    #add categorical, time-base features
    data_labels_set['dow'] = data_labels_set.loc(axis=1)['date'].dt.weekday
    data_labels_set['dom'] = data_labels_set.loc(axis=1)['date'].dt.day
    data_labels_set['moy'] = data_labels_set.loc(axis=1)['date'].dt.month
    data_labels_set['hod'] = data_labels_set.loc(axis=1)['date'].dt.hour
    data_labels_set['td_ip'] = (data_labels_set.loc(axis=1)['date']-data_labels_set.loc(axis=1)['date_iss']).astype('timedelta64[h]')

    #add dependence on windspeed of t-1 and t+1
    data_dict = {}
    for i in range(1,8):
        data_dict['wf'+str(i)] =data_labels_set.loc['wf'+str(i)].loc(axis=1)['ws'].shift(periods=1)
    data_labels_set['wsm1'] = pd.concat(data_dict, axis = 0).values

    data_dict = {}
    for i in range(1,8):
        data_dict['wf'+str(i)] =data_labels_set.loc['wf'+str(i)].loc(axis=1)['ws'].shift(periods=-1)
    data_labels_set['wsp1'] = pd.concat(data_dict, axis = 0).values
    
    #date_pred and date_iss info already in date and td_ip respectively
    data_labels_set.drop(['date_pred', 'date_iss'], axis = 1, inplace = True)
    
    #reset indices
    data_labels_set.reset_index(inplace = True)
    data_labels_set.set_index(['wf','date'], inplace = True)
    
    return data_labels_set
    
############################################################################################################   
    
#sequential train_test_split
def train_test_split_seq(data_labels_set, test_size = 0.1):
    train_length=int(len(data_labels_set)*(1-test_size))
    return data_labels_set.iloc[0:train_length], data_labels_set.iloc[train_length:]
    
############################################################################################################      
    
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        return x[self.attribute_names].values
       
############################################################################################################
        
#convert to nparray, encode and impute data
#imputer needed since there are some missing values fpr the attributes wsm1 and wsp1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion

num_attribs = ['ws','wsm1','wsp1','wd']
num_pipeline = make_pipeline(DataFrameSelector(num_attribs)
                             ,SimpleImputer(missing_values=np.nan, strategy='median')
                             ,StandardScaler())

cat_attribs = ['wf','dow','dom','moy','hod','td_ip']
cat_pipeline = make_pipeline(DataFrameSelector(cat_attribs)
                             ,OneHotEncoder(categories='auto'))


comb_pipeline = FeatureUnion(transformer_list = [
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])