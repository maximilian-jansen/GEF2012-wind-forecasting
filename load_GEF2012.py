#load the GEF2012 data and labels

import pandas as pd

def load_GEF2012():
    
    #load and preprocess the wind forecasts
    data_dict = {}
    for i in range(1,8):
        data_dict['wf'+str(i)] = pd.read_csv("datasets/windforecasts_wf"+str(i)+".csv")
    data = pd.concat(data_dict, axis = 0)
    
    #define new indices
    data.reset_index(inplace=True)
    data.rename(index=str, columns = {'level_0': 'wf', 'level_1': 'id'},inplace=True)
    data = data.astype({'id': int})
    data.set_index(['wf','id'],inplace=True)
    data.sort_index(axis=0,inplace=True)

    #get the date for which the wind is forecasted
    data.loc(axis=1)['date'] = pd.to_datetime(data.loc(axis=1)['date'], format="%Y%m%d%H")
    data.loc(axis=1)['date_pred'] = data.loc(axis=1)['date']+pd.to_timedelta(data.loc(axis=1)['hors'], unit='h')

    data = data.rename(index=str, columns = {'date': 'date_iss'})
    data.drop('hors', axis = 1, inplace = True)
    
    
    #load and preprocess the labels
    labels = pd.read_csv("datasets/train.csv")

    labels['id'] = range(0,len(labels))

    for i in range(1,8):
        labels = labels.rename(index=str, columns = {'wp'+str(i): 'wf'+str(i)})
    
    #rearrange the dataframe
    labels=pd.melt(labels, id_vars=['id','date'], value_vars = ['wf1', 'wf2', 'wf3', 'wf4', 'wf5', 'wf6', 'wf7'], var_name='wf', value_name='wp')
    
    #reset indices
    labels.set_index(['wf','id'], inplace=True)

    labels.loc(axis=1)['date']=pd.to_datetime(labels.loc(axis=1)['date'], format="%Y%m%d%H")
    
    return data, labels