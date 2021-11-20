import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

d = pd.read_csv('input-tm.csv')

ICD9_CODE_map = {
    4019:  0,
    5849:  1,
    53081: 2,
    51881: 3,
}

d["ICD9_CODE"] = d["ICD9_CODE"].map(ICD9_CODE_map)

X = d.values
X = X[:,1:9]
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

pred = kmeans.labels_
true = d["ICD9_CODE"].values
pd.DataFrame(kmeans.labels_).to_csv("kmeanlabels.csv")

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

'''
#Try only 2 classes
acc=0
max = 0
for i in range (2):
    count = 0
    for n in range(len(pred)):
        if pred[n]==i and true[n]==0:
            count=count+1
    if count>max:
        max=count
        c0=i
acc=acc+max

max = 0
for k in range (2):
    count = 0
    for n3 in range(len(pred)):
        if k!=c0 and pred[n3]==k and true[n3]==1:
            count=count+1
    if count>max:
        max=count
        c1=k

acc=acc+max
#Accuracy!!
acc = acc/2170
print(acc)

#change pred
for p in range(len(pred)):
    if pred[p] == 0:
        pred[p] = c0
    elif pred[p] == 1:
        pred[p] =c1

precision_c0,recall_c0,f1_c0=F1(pred,true,0)
precision_c1,recall_c1,f1_c1=F1(pred,true,1)

preAvg=(precision_c0+precision_c1)/2
reAvg=(recall_c0+recall_c1)/2
f1Avg=(f1_c0+f1_c1)/2

print(preAvg,reAvg,f1Avg)

'''
#4 classes
#pred - kmeans.labels_ - 0/1/2/3
#true - ICD9 

acc=0
max = 0
predTemp=[-1,-1,-1,-1]
predNew=[-1]*len(pred)
predAssign = pd.Series(pred)
predFinal=[-1]*len(pred)

from sklearn.metrics import accuracy_score

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
 


''' previous version. incorrect. 
for i in range (4):
    count = 0
    for n in range(len(pred)):
        if pred[n]==i and true[n]==0:
            count=count+1
    if count>max:
        max=count
        c0=i

acc=acc+max
max = 0
for j in range (4):
    count = 0
    for n2 in range(len(pred)):
        if j!=c0 and pred[n2]==j and true[n2]==3:
            count=count+1
    if count>max:
        max=count
        c1=j

acc=acc+max
max = 0
for k in range (4):
    count = 0
    for n3 in range(len(pred)):
        if k!=c0 and k!=c1 and pred[n3]==k and true[n3]==1:
            count=count+1
    if count>max:
        max=count
        c2=k

acc=acc+max
for l in range (4):
    count=0
    for n4 in range(len(pred)):
        if l!=c0 and l!=c1 and l!=c2:
            count=count+1
        c3=l
acc=acc+max
'''

#Accuracy!
from sklearn.metrics import accuracy_score
print("acc_sklearn:",accuracy_score(true, predFinal))

#Assign new class to pred.

precision_c0,recall_c0,f1_c0=F1(predFinal,true,0)
precision_c1,recall_c1,f1_c1=F1(predFinal,true,1)
precision_c2,recall_c2,f1_c2=F1(predFinal,true,2)
precision_c3,recall_c3,f1_c3=F1(predFinal,true,3)

preAvg=(precision_c0+precision_c1+precision_c2+precision_c3)/4
reAvg=(recall_c0+recall_c1+recall_c2+recall_c3)/4
f1Avg=(f1_c0+f1_c1+f1_c2+f1_c3)/4

print(preAvg,reAvg,f1Avg)