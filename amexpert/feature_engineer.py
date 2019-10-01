import numpy as np
import pandas as pd

__all__ = ['modify_transaction_df','modify_campaign_df','get_feat_engineered_df']

def modify_transaction_df(df) :
    df = df.copy()
    df['year'] = df['date'].apply(lambda x : int(x.split('-')[0]))
    df['month'] = df['date'].apply(lambda x : int(x.split('-')[1]))
    df['cal_date'] = df['date'].apply(lambda x : int(x.split('-')[2]))
    df['net_pay'] = df['selling_price'] + df['other_discount'] + df['coupon_discount']
    df['discount_rate'] = 1 - df['net_pay']/df['selling_price']
    df['use_coupon'] = df['coupon_discount'] < 0
    df['use_other_discount'] = df['other_discount'] < 0
    df['use_any_discount'] = df['use_coupon'] | df['use_other_discount']
    return df

def modify_campaign_df(df) :
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], format='%d/%m/%y')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%d/%m/%y')
    df['length'] = (df['end_date']-df['start_date']).dt.days
    return df

def count_unique_brand(x) :
    return x.nunique()

def counttype(cond) :
    def fun(x) :
        return (x == cond).sum()
    return fun

def get_feat_engineered_df(train_df, 
                           test_df, 
                           transaction_df, 
                           campaign_df, 
                           demographics_df,
                           coupon_map_df,
                           item_data_df
                           ) :
    modified_train = train_df.copy()
    modified_test = test_df.copy()
    features = []

    gb_dict = transaction_df.groupby(['customer_id','year','month']).agg({'net_pay' : 'sum'}).groupby('customer_id').mean().to_dict()['net_pay']
    modified_train['mean_monthly_expenditure'] = train_df['customer_id'].map(gb_dict)
    modified_test['mean_monthly_expenditure'] = test_df['customer_id'].map(gb_dict)
    features.append('mean_monthly_expenditure')

    gb_dict = transaction_df.groupby(['customer_id']).agg({'discount_rate' : 'median'}).to_dict()['discount_rate']
    modified_train['median_discount_rate'] = train_df['customer_id'].map(gb_dict)
    modified_test['median_discount_rate'] = test_df['customer_id'].map(gb_dict)
    features.append('median_discount_rate')

    agg_transaction_feat = ['use_any_discount', 'use_coupon', 'use_other_discount']
    for feat in agg_transaction_feat :
        gb_dict = transaction_df.groupby(['customer_id']).agg({feat : 'mean'}).to_dict()[feat]
        feat_name = '{}_rate'.format(feat)
        modified_train[feat_name] = train_df['customer_id'].map(gb_dict)
        modified_test[feat_name] = test_df['customer_id'].map(gb_dict)
        features.append(feat_name)

    campaign_feat_list = ['campaign_type']

    for feat in campaign_feat_list :
        map_dict = campaign_df[['campaign_id',feat]].set_index('campaign_id').to_dict()[feat]
        modified_train['campaign_type'] = train_df['campaign_id'].map(map_dict)
        modified_test['campaign_type'] = test_df['campaign_id'].map(map_dict)
        features.append(feat)

    num_map = {'X' : 0, 'Y' : 1} 
    modified_train['campaign_type'] = modified_train['campaign_type'].map(num_map)
    modified_test['campaign_type'] = modified_test['campaign_type'].map(num_map)

    brand_types = ['Established', 'Local']

    for bt in brand_types :
        gb_dict = coupon_map_df.groupby('coupon_id').agg({'brand_type' : counttype(bt)}).to_dict()['brand_type']
        feat_name = 'total_{}_brand_items'.format(bt.lower())
        modified_train[feat_name] =  train_df['coupon_id'].map(gb_dict)
        modified_test[feat_name] = test_df['coupon_id'].map(gb_dict)
        features.append(feat_name)

    modified_train['total_items'] =  modified_train['total_established_brand_items'] + modified_train['total_local_brand_items']
    modified_test['total_items'] = modified_test['total_established_brand_items'] + modified_test['total_local_brand_items']
    features.append('total_items')

    gb_dict = coupon_map_df.groupby('coupon_id').agg({'brand' : count_unique_brand}).to_dict()['brand']
    modified_train['nunique_brand'] = train_df['coupon_id'].map(gb_dict)
    modified_test['nunique_brand'] = test_df['coupon_id'].map(gb_dict)
    features.append('nunique_brand')

    demo_feats = ['income_bracket','age_range']
    for feat in demo_feats :
        map_dict = demographics_df[['customer_id',feat]].set_index('customer_id').to_dict()[feat]
        modified_train[feat] = train_df['customer_id'].map(map_dict)
        modified_test[feat] = test_df['customer_id'].map(map_dict)

    map_dict = demographics_df[['customer_id','family_size']].set_index('customer_id').replace('5+',5,).astype(int).to_dict()['family_size']
    modified_train['family_size'] = train_df['customer_id'].map(map_dict)
    modified_test['family_size'] = test_df['customer_id'].map(map_dict)

    age_map = {'70+' : 5, '56-70': 4, '46-55' : 3, '36-45' : 2, '26-35': 1, '18-25' : 0}
    modified_train['age_range'] = modified_train['age_range'].map(age_map)
    modified_test['age_range'] = modified_test['age_range'].map(age_map)
    
    fillna_feats = ['family_size', 'age_range', 'income_bracket']
    for feat in fillna_feats :
        modified_train[feat] = modified_train[feat].fillna(-999)
        modified_test[feat] = modified_test[feat].fillna(-999)
        if feat not in features :
            features.append(feat)

    return modified_train, modified_test, features

if __name__=='__main__' :
    pass
