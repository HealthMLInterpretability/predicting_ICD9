from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.initializers import random_uniform
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Convolution1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import BatchNormalization
from keras.layers.pooling import AveragePooling1D
from sklearn.model_selection import KFold
from keras.layers import Input
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.utils.np_utils import to_categorical
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from typing import Optional
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
import os
import time
import keras
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
tqdm.pandas()


def build_model_conv1d(x_train, num_classes: int):
    model = Sequential()
    model.add(Convolution1D(32, 3, activation='relu',
              input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer='adam', metrics=["accuracy"])
    return model


def lf_schedule(epoch, lr):
    if epoch % 2 == 0:
        lr = lr - 0.02*lr  # -->i.e. 0.95*lr (2% decay)
        print(f'New learning rate for epoch={epoch} is {lr}')
        return lr
    else:
        return lr


def make_callback_earlystop():

    # use early stopping to exit training if validation loss is not decreasing even after certain epochs
    callback1 = EarlyStopping(monitor="val_accuracy",
                              patience=2, verbose=0, mode="auto", min_delta=0)
    # If validation accuracy at current epoch is less than previous epoch accuracy, decrease the learning rate by factor of 2%.
    callback2 = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.02, patience=1, verbose=0, mode="auto")
    # For every 3rd epoch, decay learning rate by 5%
    callback3 = LearningRateScheduler(schedule=lf_schedule, verbose=0)
    return callback1, callback2, callback3


def F1(pred, true, clabel):  # Accuracy / F1 / Precision / Recall Output
    TP, FP, FN = 0, 0, 0
    for i in range(len(pred)):
        # only for minority class.
        if pred[i] == true[i] and pred[i] == clabel:

            TP += 1
        if pred[i] == clabel and true[i] != clabel:
            FP += 1
        if pred[i] != clabel and true[i] == clabel:
            FN += 1
    if TP == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*TP/(2*TP+FP+FN)

    return precision, recall, f1


def get_avg_PRF(label_count: int, true, pred):

    acc = 0
    max = 0
    predTemp = [-1]*label_count
    predNew = [-1]*len(pred)
    predAssign = pd.Series(pred)
    predFinal = [-1]*len(pred)

    from sklearn.metrics import accuracy_score

    if(label_count == 2):
        for i in range(label_count):
            predTemp[i] = 0
            for j in range(label_count):
                if j != i:
                    predTemp[j] = 1
                else:
                    continue
                for l in range(label_count):

                    pred_map = {
                        0: predTemp[0],
                        1: predTemp[1],
                    }
                    predNew = predAssign.map(pred_map)
                    predNew = predNew.values
                    acc = accuracy_score(true, predNew)
                    if acc > max:
                        max = acc
                        predFinal = predNew
                    else:
                        continue

    if (label_count == 4):
        for i in range(label_count):
            predTemp[i] = 0
            for j in range(label_count):
                if j != i:
                    predTemp[j] = 1
                else:
                    continue
                for k in range(label_count):
                    if k != i and k != j:
                        predTemp[k] = 2
                    else:
                        continue
                    for l in range(label_count):
                        if l != i and l != k and l != j:
                            predTemp[l] = label_count-1
                            pred_map = dict()
                            for x in range(0, label_count):
                                pred_map[x] = predTemp[x]

                            predNew = predAssign.map(pred_map)
                            predNew = predNew.values
                            acc = accuracy_score(true, predNew)
                            if acc > max:
                                max = acc
                                predFinal = predNew
                        else:
                            continue

    # Assign new class to pred.
    RPF_list = []

    for x in range(0, label_count):
        precision, recall, f1 = F1(predFinal, true.tolist(), x)
        RPF_list.append([precision, recall, f1])

    preAvg = 0
    reAvg = 0
    f1Avg = 0
    f1_scores = []
    for x in range(len(RPF_list)):
        preAvg += RPF_list[x][0]
        reAvg += RPF_list[x][1]
        f1Avg += RPF_list[x][2]
        f1_scores.append(RPF_list[x][2])

    preAvg = round((preAvg/label_count)*100, 3)
    reAvg = round((reAvg/label_count)*100, 3)
    f1Avg = round((f1Avg/label_count)*100, 3)
    acc = round((accuracy_score(true, predFinal))*100, 3)
    bal_acc = round(balanced_accuracy_score(true, predFinal)*100, 3)
    predFinalLen = []

    for x in range(0, label_count):
        predFinalLen.append(len(predFinal == x))

    f1Weighted = round((np.average(f1_scores, weights=predFinalLen))*100, 3)

    return acc, bal_acc, preAvg, reAvg, f1Avg, f1Weighted


def Classification_Metrics(**kwargs):
    test = kwargs['test']
    preds = kwargs['preds']
    model_name = kwargs['model_name']
    pred_prob = kwargs['pred_prob'] if 'pred_prob' in kwargs else []
    labels_count = kwargs['num_classes']

    accuracy, bal_acc, precision, recall, f1_average, f1_weighted = get_avg_PRF(
        labels_count, test, preds)

    if (len(pred_prob) > 0):
        if labels_count == 2:
            if np.array(pred_prob).ndim > 1:
                pos_pred_prob = pred_prob[:, 1]
            else:
                pos_pred_prob = pred_prob
            auc = round(roc_auc_score(test, pos_pred_prob,
                                      multi_class='ovo', average='weighted')*100, 3)
        else:
            auc = round(roc_auc_score(
                test, pred_prob, multi_class='ovr', average='weighted')*100, 3)

    else:
        auc = "None"

    result = [model_name.upper(), accuracy, bal_acc, recall,
              f1_weighted, f1_average, precision, auc]

    return result


def Random_Forest_EXP(x_train, x_test, y_train, y_test, num_classes, result: list):

    clf_RF = RandomForestClassifier(random_state=32)
    clf_RF.fit(x_train, y_train)
    preds = clf_RF.predict(x_test)
    pred_prob = clf_RF.predict_proba(x_test)
    result_clf_RF = Classification_Metrics(
        test=y_test, preds=preds, pred_prob=pred_prob,  model_name="Random Forest", num_classes=num_classes)
    result.append(result_clf_RF)


def KMEANS_EXP(X: pd.DataFrame, Y: pd.DataFrame, num_classes, result: list):

    kmeans = KMeans(n_clusters=num_classes,
                    random_state=0, max_iter=1000, algorithm='auto')
    kmeans.fit(X)
    pred = kmeans.labels_
    correct_labels = sum(Y == pred)
    result_k_means = Classification_Metrics(
        test=Y, preds=pred, model_name="K-Means", num_classes=num_classes)
    result.append(result_k_means)


def Conv1D(X_train, X_test, Y_train, Y_test, num_classes, result: list):
    model_conv1d = build_model_conv1d(X_train, num_classes)
    Y_train = to_categorical(Y_train, num_classes)
    callback1, callback2, callback3 = make_callback_earlystop()
    history = model_conv1d.fit(X_train, Y_train, validation_split=0.15, epochs=10,
                               batch_size=8, shuffle=True, verbose=0, callbacks=[callback1, callback2, callback3])
    predictions = model_conv1d.predict(X_test)
    Y_pred = predictions.argmax(axis=1)
    Y_test_max = Y_test
    result.append(Classification_Metrics(
        test=Y_test_max,  preds=Y_pred, pred_prob=predictions, model_name="CNN", num_classes=num_classes))


def baseline_models_experimental(df_experiments: pd.DataFrame, model_type: Optional[str] = "",  num_classes: Optional[int] = 2, top_icd_map: Optional[bool] = False):

    df_temp = df_experiments.copy()
    Y = df_temp.label

    if (top_icd_map == True):
        Y = Y.map({414: 0, 38: 1, 410: 2, 424: 3})

    df_temp.drop(columns={'label'}, inplace=True)
    X = df_temp
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, stratify=Y)

    result_baseline = []

    Random_Forest_EXP(x_train, x_test, y_train, y_test,
                      num_classes, result_baseline)
    KMEANS_EXP(X, Y, num_classes, result_baseline)
    Conv1D(x_train, x_test, y_train, y_test, num_classes, result_baseline)

    Pretty_Table = PrettyTable(["Model", "Accuracy", "Balanced Accuracy",
                                "Recall", "F1-Weighted", "F1-Average", "Precision", "AUC"])
    Pretty_Table.title = model_type.upper()
    for x in result_baseline:
        Pretty_Table.add_row(x)

    print(Pretty_Table, end="\n")


class BaselineModels():
    """
    SUMMARY:
    Loads and run Random Forest, Kmeans and CNN 1D model for our project Interpretability of Clinal Text
    - CNN models runs on 10 epochs as default and for train test split is test_size is 30% 
    :param dataset: Dataset should be a pandas dataframe with numerical features (numerical embeddings) and label column name should be "label"
    :param dataset_name: This parameter is optional however it aspects dataset name in string format which is going to be title of the final result table
    :param num_classes: num_classes aspects integer value denoting number of classes, default value is 2
    :param top_icd_map: If the dataset contains labels as ICD9 codes, inorder to convert them into labels for classification top_icd_map can be marked as True       
    """

    def __init__(self, dataset:pd.DataFrame, dataset_name: Optional[str] = "", num_classes: Optional[int] = 2, top_icd_map: Optional[bool] = False):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.top_icd_map = top_icd_map
        baseline_models_experimental(df_experiments=self.dataset, model_type=self.dataset_name,
                                     num_classes=self.num_classes, top_icd_map=self.top_icd_map)

