from feature_engineer import *

from pathlib import Path

import os
from copy import deepcopy
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

from functools import partial

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

n_folds = 8
kf = GroupKFold(n_folds)
trials = Trials()

space = {
    'xgboost' : {'max_depth' : hp.quniform('max_depth', 2,8,1),
                 'n_estimators' : hp.quniform('n_estimators', 1000, 2000,50),
                 'lambda' : hp.uniform('lambda', 0.1, 3),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                 'eta' : hp.uniform('eta', 0.1,0.6),
                 'alpha' : hp.uniform('alpha',0,1),
                 'subsample' : hp.uniform('subsample', 0.6,1),
                 'scale_pos_weight' : hp.choice('scale_pos_weight', [False, True])},

    'xgboost_dart' : {'max_depth' : hp.quniform('max_depth', 2,8,1),
                      'n_estimators' : hp.quniform('n_estimators', 100, 800,50),
                      'lambda' : hp.uniform('lambda', 0.1, 3),
                      'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                      'eta' : hp.uniform('eta', 0.1,0.6),
                      'alpha' : hp.uniform('alpha',0,1),
                      'subsample' : hp.uniform('subsample', 0.6,1),
                      'scale_pos_weight' : hp.choice('scale_pos_weight', [False, True])},

    'catboost' : {'max_depth' : hp.quniform('max_depth', 5,11,1),
                  'n_estimators' : hp.quniform('n_estimators', 1000, 2000,50),
                  'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 1, 5),
                  'eta' : hp.uniform('eta', 0.01,0.6),
                  'bagging_temperature' : hp.uniform('bagging_temperature', 0.67,3),
                  'scale_pos_weight' : hp.choice('scale_pos_weight', [False, True])}
}

def init_model(model_type='xgboost', scale_const = 1, params=None) :
    if 'xgboost' in model_type  :
        from xgboost import XGBClassifier
        if 'dart' in model_type :
            return XGBClassifier(booster = 'dart', random_state = 100, n_jobs = -1, scale_pos_weight = scale_const, eval_metric='auc', **params)
        else :
            return XGBClassifier(random_state = 100, n_jobs = -1, scale_pos_weight = scale_const, **params)
    elif 'catboost' in model_type :
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_state = 100, eval_metric='AUC', scale_pos_weight = scale_const, verbose = False, use_best_model = True, **params)

def eval_model(params, model_type) :
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    scale = params.pop('scale_pos_weight')

    kf_score = 0
    for i, (train_idx,val_idx) in enumerate(kf.split(modified_train, modified_train['redemption_status'], modified_train['customer_id'])) :
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
        pred_val = model.predict_proba(modified_train.iloc[val_idx][features])[:,1]
        roc_val = roc_auc_score(modified_train.iloc[val_idx]['redemption_status'], pred_val)
        kf_score += roc_val/n_folds
    return 1 - kf_score

def get_best_params(trials) :
    trials = deepcopy(trials.trials)
    def check_val(x) :
        try :
            return  x['result']['loss']
        except :
            return 100
    best = min(trials, key = lambda x : check_val(x))
    best_params = best['misc']['vals']
    for param in best_params.keys() :
        param_value = best_params[param][0]
        if param in ['n_estimators', 'max_depth'] :
            param_value = int(param_value)
        best_params[param] = param_value
    scale = best_params.pop('scale_pos_weight')
    if scale == 1 :
        return best_params, True
    else :
        return best_params, False
    
if __name__=='__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='xgboost')
    parser.add_argument('--max_evals', default='50', type=int)
    
    args = vars(parser.parse_args())
    model_type = args['model_type']
    fixed_eval_model =partial(eval_model, model_type = model_type)
    best = fmin(fixed_eval_model, space=space[model_type], algo=tpe.suggest, max_evals=args['max_evals'], trials = trials)
    params, scale = get_best_params(trials)

    with open('{}_params.pkl'.format(model_type), 'wb') as f :
        pickle.dump((params, scale) , f)





                                                        