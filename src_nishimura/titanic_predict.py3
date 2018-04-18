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
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import random

import sys
sys.path.append('lib')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ChainerのFutureWarningが結果の一覧性を下げているので消す

import pickle
import os

def load_train_data():
    pass

def output_prediction():
    # PassengerId,Survived
    pass

def main():
    # 特徴量
    pass


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
