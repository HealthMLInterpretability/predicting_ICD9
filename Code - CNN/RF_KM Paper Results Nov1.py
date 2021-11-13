#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# In[8]:


df_TFIDF40 = pd.read_csv('input_ICD9_TFIDF_40.csv')
df_TM5 = pd.read_csv('input_ICD9_TM_5.csv')
df_TM20 = pd.read_csv('input_ICD9_TM_20.csv')
df_TM30 = pd.read_csv('input_ICD9_TM_30.csv')
df_TM30.rename(columns={'top_icd': 'ICD9'}, inplace=True)
df_TM39 = pd.read_csv('input_ICD9_TM_39.csv')

df_list = [df_TFIDF40, df_TM5, df_TM20, df_TM30, df_TM39]
files_list = ['TFIDF_40', 'TM_5', 'TM_20', 'TM_30', 'TM_39']


# In[32]:


# Some data quality checks
# Label is consistent
print([True for df in df_list if 'ICD9' in df.columns])
print([df.shape for df in df_list])
print([df['ICD9'].value_counts() for df in df_list])


# In[34]:


def get_classification_metrics_rf(df: pd.DataFrame, label_col:str):
    '''
    Get accuracy and F1 metrics from Random Forest
    '''
    # Train test split
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Classifer
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    rf_acc = balanced_accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average = 'weighted')
    rf_auc = roc_auc_score(y_test, rf_prob, multi_class='ovr', average='macro')
    
    # Construct results
    results = dict()
    results['pred'], results['pred_prob'] = rf_pred, rf_prob
    results['acc'], results['f1'], results['auc'] = rf_acc, rf_f1, rf_auc
    results['model'] = rf
        
    return results


# In[48]:


def F1(pred, true, clabel): # Accuracy / F1 / Precision / Recall Output
    TP,FP,FN=0,0,0 
    for i in range(len(pred)):
        if pred[i] == true[i] and pred[i] == clabel: # only for minority class.
            TP+=1
        if pred[i] == clabel and true[i] != clabel:
            FP+=1
        if pred[i] != clabel and true[i] == clabel:
            FN+=1
    if TP==0:
        precision=0
        recall=0
        f1=0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*TP/(2*TP+FP+FN)

    return precision,recall,f1


def get_classification_metrics_km(df: pd.DataFrame, label_col:str): #PY Double check
    '''
    Get accuracy and F1 metrics from Kmeans
    '''
    # Train test split
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Kmeans Clustering
    km = KMeans()
    km.fit(X_train)
    km_pred = km.predict(X_test)
    
    ICD9_CODE_map = {
    '414':  0, #chronic heart
    '38':  1, #sepsis
    '410': 2, #heart attack
    '424': 3, #diseases of endocardium
    }
    
    y_train_km = y_train.map(ICD9_CODE_map)
    y_test_km = y_test.map(ICD9_CODE_map)
    pred = kmeans.labels_
    true = y_test_km
    
    acc=0
    max = 0
    predTemp=[-1,-1,-1,-1]
    predNew=[-1]*len(pred)
    predAssign = pd.Series(pred)
    predFinal=[-1]*len(pred)

    for i in range(4):
        predTemp[i]=0
        for j in range(4):
            if j!=i:
                predTemp[j]=1
            else:
                continue
            for k in range(4):
                if k!=i and k!=j:
                    predTemp[k]=2
                else:
                    continue
                for l in range(4):
                    if l!=i and l!=k and l!=j:
                        predTemp[l]=3
                        pred_map = {
                            0: predTemp[0],
                            1: predTemp[1],
                            2: predTemp[2],
                            3: predTemp[3],
                        }
                        predNew = predAssign.map(pred_map)
                        predNew = predNew.values
                        acc = accuracy_score(true, predNew)
                        if acc > max: 
                            max = acc
                            predFinal = predNew  
                    else:
                        continue

    #Assign new class to pred.
    precision_c0,recall_c0,f1_c0=F1(predFinal,true.tolist(),0)
    precision_c1,recall_c1,f1_c1=F1(predFinal,true.tolist(),1)
    precision_c2,recall_c2,f1_c2=F1(predFinal,true.tolist(),2)
    precision_c3,recall_c3,f1_c3=F1(predFinal,true.tolist(),3)

    preAvg=(precision_c0+precision_c1+precision_c2+precision_c3)/4
    reAvg=(recall_c0+recall_c1+recall_c2+recall_c3)/4
    f1Avg=(f1_c0+f1_c1+f1_c2+f1_c3)/4
    
    # Weighted F1 (PY double check)
    f1Weighted = np.average([f1_c0, f1_c1, f1_c2, f1_c3], weights=[len(predFinal==0), len(predFinal==1), len(predFinal==2), len(predFinal==3)])
    
    # Accuracy
    bal_acc = balanced_accuracy_score(true, predFinal)
    
    # Results
    results = dict()
    results['pred'] = rf_pred, pred
    results['acc'], results['f1'], results['f1_weighted'] = bal_acc, f1Avg, f1Weighted
    results['model'] = km
    
    return results    


# In[38]:


rf_results_list = [get_classification_metrics_rf(df, 'ICD9') for df in df_list]


# In[49]:


km_results_list = [get_classification_metrics_km(df, 'ICD9') for df in df_list]


# In[ ]:





# In[ ]:





# In[ ]:




