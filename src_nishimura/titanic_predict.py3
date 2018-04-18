# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Kaggleのタイタニックタスクを解くやつ

t-nishimura 2018
"""

import argparse

import numpy as np

import pandas as pd
import seaborn as sn
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import random

import sys
sys.path.append('lib')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ChainerのFutureWarningが結果の一覧性を下げているので消す

import pickle
import os
import csv

def load_train_data():
    train_data = pd.read_csv('input/train.csv')
    # feature_data = train_data.loc[:,['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
    feature_data = train_data.loc[:,['Age','SibSp','Parch','Fare']]
    feature_data = feature_data.fillna(0)
    label_data = train_data.loc[:,['Survived']]
    return feature_data, label_data

def output_prediction(clf):
    # PassengerId,Survived
    # test_feature = pd.read_csv('input/test.csv').loc[:, ['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
    test_feature = pd.read_csv('input/test.csv').loc[:,['Age','SibSp','Parch','Fare']]
    test_feature = test_feature.fillna(0)
    predictions = clf.predict(test_feature)

    passenger_id = pd.read_csv('input/test.csv').loc[:,['PassengerId']]
    f = open('output_nishimura/sample_predictions.csv', 'w')
    print('PassengerId,Survived', file=f)
    for passenger_id_in_test, predict in enumerate(predictions):
        print('%s,%s' % (passenger_id.loc[passenger_id_in_test][0], predict), file=f)

def main():
    feature_data, label_data = load_train_data()
    clf = ensemble.RandomForestClassifier()
    clf.fit(feature_data,label_data)
    output_prediction(clf)

if __name__ == "__main__":
    """ コマンドエラー時に表示する文字列 """
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--env_str',
        type = str,
        dest = 'env_str',
        default = 'WMCAI',
        help = 'DB environment setting.'
    )
    # main(args.env_str)
    main()
