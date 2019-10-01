from feature_engineer import *
from optimize_params import init_model

from pathlib import Path

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

from hyperopt import fmin, tpe, space_eval, hp, Trials

from sklearn.model_selection import GroupKFold


TRAIN_PATH = Path('./')

main_train = pd.read_csv(TRAIN_PATH/'train.csv')
transaction = pd.read_csv(TRAIN_PATH/'customer_transaction_data.csv')
coupon_map = pd.read_csv(TRAIN_PATH/'coupon_item_mapping.csv')
demographics = pd.read_csv(TRAIN_PATH/'customer_demographics.csv')
item_data = pd.read_csv(TRAIN_PATH/'item_data.csv')
test_df = pd.read_csv('test.csv')
campaign = pd.read_csv(TRAIN_PATH/'campaign_data.csv')

transaction = modify_transaction_df(transaction)
campaign = modify_campaign_df(campaign)

coupon_map = coupon_map.merge(item_data, on = 'item_id', how = 'left')


modified_train, modified_test, features = get_feat_engineered_df(main_train, test_df, transaction, campaign, demographics, coupon_map, item_data)


if __name__=='__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='xgboost')
    args = vars(parser.parse_args())
    model_type = args['model_type']

    with open('{}_params.pkl'.format(model_type), 'rb') as f :
        params, scale = pickle.load(f)

    n_folds = 8
    kf = GroupKFold(n_folds)
    trials = Trials()
    kf_score = 0
    pred = np.zeros(len(test_df))

    for i , (train_idx, val_idx) in enumerate(kf.split(modified_train, modified_train['redemption_status'], modified_train['customer_id'])) :
        scale_const = 1/modified_train['redemption_status'].iloc[train_idx].mean() if scale else 1

        model = init_model(model_type, scale_const, params)
        if 'catboost' in model_type :
            eval_set = (modified_train.iloc[val_idx][features], modified_train.iloc[val_idx]['redemption_status'])
        elif 'xgboost' in model_type :  
            eval_set = [(modified_train.iloc[val_idx][features], modified_train.iloc[val_idx]['redemption_status'])]

        model.fit(modified_train.iloc[train_idx][features], 
                  modified_train.iloc[train_idx]['redemption_status'],
                  verbose = False, 
                  eval_set=eval_set)

        pred_train = model.predict_proba(modified_train.iloc[train_idx][features])[:,1]
        roc_train = roc_auc_score(modified_train.iloc[train_idx]['redemption_status'], pred_train)

        pred_val = model.predict_proba(modified_train.iloc[val_idx][features])[:,1]
        roc_val = roc_auc_score(modified_train.iloc[val_idx]['redemption_status'], pred_val)

        kf_score += roc_val/n_folds

        pred_test = model.predict_proba(modified_test[features])[:,1]
        pred += pred_test/n_folds

        print('Fold {} roc_auc train : {}             roc_auc val : {}'.format(i+1,roc_train, roc_val))
        print()

    print('KFold AUC :', kf_score)

    subm = test_df[['id']].copy()
    subm['redemption_status'] = pred
    subm.to_csv('subm_{}.csv'.format(model_type), index= False)

