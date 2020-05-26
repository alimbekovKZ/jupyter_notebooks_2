# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 500)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, hp, tpe, space_eval

from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
from time import time
from tqdm import tqdm_notebook

from xgboost import XGBClassifier
import os

from sklearn.model_selection import KFold

import gc
import warnings
warnings.filterwarnings('ignore')


# 迭代次数传参
# python label_lgb2.py 1.2

import argparse
ap = argparse.ArgumentParser(description='label_lgb2.py')
ap.add_argument('nround_times', nargs='*', action="store", default=1.0, type = float)
pa = ap.parse_args()
nround_times = pa.nround_times[0]

# # 导入数据

# In[2]:


NROWS = None
# NROWS = 5000


# 使用原始数据
train = pd.read_csv('../temp/train_label.csv', nrows=NROWS)

# 使用增加样本后的数据
# train = pd.read_csv('../temp/train_label_50.csv', nrows=NROWS)

test = pd.read_csv('../temp/test_label.csv', nrows=NROWS)
test = test.drop('isFraud',axis=1)
sub = pd.read_csv('../temp/sample_submission_label.csv', nrows=NROWS)

print("train.shape:", train.shape)
print("test.shape:", test.shape)
# train.head(3)

target = "isFraud"

# ## 内存优化

# In[4]:


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[: 3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min  and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum()/ 1024**2
    print('Memory usage after optimization is: {:.2f} MB, {:.1f}% reduction'.          format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[5]:


train = reduce_mem_usage(train)
test  = reduce_mem_usage(test)


# # 特征工程
print("feature engineer")


# ## 特征分类: 按特征名分类

# ### 缺失值的数量

# shift特征
train['card_addr1_P_emaildomain'] = train["card1"].apply(lambda x: str(x)) + "_" + train["card2"].apply(
    lambda x: str(x)) + \
                                    "_" + train["card3"].apply(lambda x: str(x)) + "_" + train["card4"].apply(
    lambda x: str(x)) + \
                                    "_" + train["card5"].apply(lambda x: str(x)) + "_" + train["card6"].apply(
    lambda x: str(x)) + \
                                    "_" + train["addr1"].apply(lambda x: str(x)) + "_" + train["P_emaildomain"].apply(
    lambda x: str(x))

test['card_addr1_P_emaildomain'] = test["card1"].apply(lambda x: str(x)) + "_" + test["card2"].apply(lambda x: str(x)) + \
                                   "_" + test["card3"].apply(lambda x: str(x)) + "_" + test["card4"].apply(
    lambda x: str(x)) + \
                                   "_" + test["card5"].apply(lambda x: str(x)) + "_" + test["card6"].apply(
    lambda x: str(x)) + \
                                   "_" + test["addr1"].apply(lambda x: str(x)) + "_" + test["P_emaildomain"].apply(
    lambda x: str(x))

shift_feature = []
for i in range(1, 51):
    train["card_addr1_P_emaildomain_" + str(i) + "before"] = train["card_addr1_P_emaildomain"].shift(i)
    test["card_addr1_P_emaildomain_" + str(i) + "before"] = test["card_addr1_P_emaildomain"].shift(i)
    shift_feature.append("card_addr1_P_emaildomain_" + str(i) + "before")

for i in range(-1, -51, -1):
    train["card_addr1_P_emaildomain_" + str(-i) + "after"] = train["card_addr1_P_emaildomain"].shift(i)
    test["card_addr1_P_emaildomain_" + str(-i) + "after"] = test["card_addr1_P_emaildomain"].shift(i)
    shift_feature.append("card_addr1_P_emaildomain_" + str(-i) + "after")

def cur_in_window(x):
    cur = x[0]
    window = x[1:]
    cnt = 0
    for item in window:
        if cur == item:
            cnt += 1
    return cnt

train["shift_100_cnt"] = train[["card_addr1_P_emaildomain"] + shift_feature].apply(lambda x: cur_in_window(x), axis=1)
test["shift_100_cnt"] = test[["card_addr1_P_emaildomain"] + shift_feature].apply(lambda x: cur_in_window(x), axis=1)

train = train.drop(["card_addr1_P_emaildomain"] + shift_feature, axis=1)
test = test.drop(["card_addr1_P_emaildomain"] + shift_feature, axis=1)
# shift特征end



train['null'] = train.isna().sum(axis=1)
test['null'] = test.isna().sum(axis=1)


# # ### target_encoder
#
# print("target_encoder")
# np.random.seed(13)
#
# def impact_coding(data, feature, target='y'):
#     '''
#     In this implementation we get the values and the dictionary as two different steps.
#     This is just because initially we were ignoring the dictionary as a result variable.
#
#     In this implementation the KFolds use shuffling. If you want reproducibility the cv
#     could be moved to a parameter.
#     '''
#     n_folds = 5 #20
#     n_inner_folds = 5 #10
#     impact_coded = pd.Series()
#
#     oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
#     kf = KFold(n_splits=n_folds, shuffle=True)
#     oof_mean_cv = pd.DataFrame()
#     split = 0
#     for infold, oof in kf.split(data[feature]):
#             impact_coded_cv = pd.Series()
#             kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
#             inner_split = 0
#             inner_oof_mean_cv = pd.DataFrame()
#             oof_default_inner_mean = data.iloc[infold][target].mean()
#             for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
#                 # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
#                 oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
#                 impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
#                             lambda x: oof_mean[x[feature]]
#                                if x[feature] in oof_mean.index else oof_default_inner_mean, axis=1))
#
#                 # Also populate mapping (this has all group -> mean for all inner CV folds)
#                 inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
#                 inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
#                 inner_split += 1
#
#             # Also populate mapping
#             oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
#             oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
#             split += 1
#
#             impact_coded = impact_coded.append(data.iloc[oof].apply(
#                             lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
#                                if x[feature] in inner_oof_mean_cv.index else oof_default_mean, axis=1))
#
#     return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean
#
#
# # In[8]:
#
#
# # train.card5.nunique()
#
#
# # In[9]:
#
#
# target = "isFraud"
#
#
# # In[10]:
# # Apply the encoding to training and test data, and preserve the mapping
# categorical_features = ["card2", "card5"]
# impact_coding_map = {}
# for f in categorical_features:
#     print("Impact coding for {}".format(f))
#     train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f, target)
#     impact_coding_map[f] = (impact_coding_mapping, default_coding)
#     mapping, default_mean = impact_coding_map[f]
#     test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]]
#                                                     if x[f] in mapping else default_mean, axis=1)
#
# # ### target_encoder end

# ### targetEncoder保存中间变量


# ### 从targetEncoder运行

# ### 时间相关特征(TransactionDT)

# In[6]:

def transform_TransactionDT(df):
    START_DATE = '2017-12-01'
    start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['Date'] = df['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))

    df['Weekday'] = df['Date'].dt.dayofweek
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Morning'] = (df['Hour'] >= 7) & (df['Hour'] <= 11).astype('int')
    df['Noon'] = (df['Hour'] >=12) & (df['Hour'] <= 18).astype('int')
    df['Evening'] = (df['Hour'] >= 19) & (df['Hour'] <=23).astype('int')
    df['Midnight'] = (df['Hour'] >= 0) & (df['Hour'] <=6).astype('int')
    
    del df['Date']
    
    return df


# In[7]:


train = transform_TransactionDT(train)
test = transform_TransactionDT(test)


# ### 金额(TransactionAmt)

# In[8]:


train['TransactionAmt'] = train['TransactionAmt'].astype(float)
train['TransAmtLog'] = np.log(train['TransactionAmt'])
train['TransAmtDemical'] = train['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()

test['TransactionAmt'] = test['TransactionAmt'].astype(float)
test['TransAmtLog'] = np.log(test['TransactionAmt'])
test['TransAmtDemical'] = test['TransactionAmt'].astype('str').str.split('.', expand=True)[1].str.len()


# ### 金额(TransactionAmt)是否整除的特征

# In[9]:


def mod_m(x, m):
    if x%m == 0:
        return 1
    else:
        return 0
    
train['TransactionAmt_mod_1'] = train['TransactionAmt'].apply(lambda x: mod_m(x, 1))
train['TransactionAmt_mod_10'] = train['TransactionAmt'].apply(lambda x: mod_m(x, 10))
train['TransactionAmt_mod_50'] = train['TransactionAmt'].apply(lambda x: mod_m(x, 50))
train['TransactionAmt_mod_100'] = train['TransactionAmt'].apply(lambda x: mod_m(x, 100))

test['TransactionAmt_mod_1'] = test['TransactionAmt'].apply(lambda x: mod_m(x, 1))
test['TransactionAmt_mod_10'] = test['TransactionAmt'].apply(lambda x: mod_m(x, 10))
test['TransactionAmt_mod_50'] = test['TransactionAmt'].apply(lambda x: mod_m(x, 50))
test['TransactionAmt_mod_100'] = test['TransactionAmt'].apply(lambda x: mod_m(x, 100))


# ### ProductCD



# ### card特征提取

def get_sub(x, idx):
    try:
        return str(x)[idx]
    except:
        return "-1"

for idx in [-1, -2, -3, -4, -5]:
    train["card1" + "_sub_" + str(idx)] = train["card1"].apply(lambda x: get_sub(x, idx))
    test["card1" + "_sub_" + str(idx)]  = test["card1"].apply(lambda x: get_sub(x, idx))


# feature = 'card1'
# temp = train.groupby([feature])[target].sum().reset_index()
# temp.index = temp[feature]
# temp = temp.drop(feature, axis=1)
# faeture_map = temp.to_dict()[target]
# train[feature + "_target_cnt"] = train[feature].map(faeture_map)
# test[feature + "_target_cnt"] = test[feature].map(faeture_map)

# In[12]:


# card1是类别型特征,对card1的字段进行提取

train["card1_len"] = train["card1"].apply(lambda x: len(str(x)))
test["card1_len"]  = test["card1"].apply(lambda x: len(str(x)))

train["card1_first"] = train["card1"].apply(lambda x: str(x)[0])
test["card1_first"]  = test["card1"].apply(lambda x: str(x)[0])

# card2特征提取
# train["card2_first"] = train["card2"].apply(lambda x: str(x)[0])
# test["card2_first"]  = test["card2"].apply(lambda x: str(x)[0])

# train["card2_second"] = train["card2"].apply(lambda x: str(x)[1])
# test["card2_second"]  = test["card2"].apply(lambda x: str(x)[1])

# train["card2_last"] = train["card2"].apply(lambda x: str(x)[2])
# test["card2_last"]  = test["card2"].apply(lambda x: str(x)[2])


# In[13]:


# 是否缺失的标记
train["card1_na"] = 0
train.loc[train["card1"].isna(), "card1_na"] = 1
test["card1_na"] = 0
test.loc[test["card1"].isna(), "card1_na"] = 1

train["card2_na"] = 0
train.loc[train["card2"].isna(), "card2_na"] = 1
test["card2_na"] = 0
test.loc[test["card2"].isna(), "card2_na"] = 1

# train["card3_na"] = 0
# train.loc[train["card3"].isna(), "card3_na"] = 1
# test["card3_na"] = 0
# test.loc[test["card3"].isna(), "card3_na"] = 1

# train["card4_na"] = 0
# train.loc[train["card4"].isna(), "card4_na"] = 1
# test["card4_na"] = 0
# test.loc[test["card4"].isna(), "card4_na"] = 1

train["card5_na"] = 0
train.loc[train["card5"].isna(), "card5_na"] = 1
test["card5_na"] = 0
test.loc[test["card5"].isna(), "card5_na"] = 1

# train["card6_na"] = 0
# train.loc[train["card6"].isna(), "card6_na"] = 1
# test["card6_na"] = 0
# test.loc[test["card6"].isna(), "card6_na"] = 1


# In[14]:


# card字段拼接的统计

train['card_str'] = train["card1"].apply(lambda x: str(x)) + "_" + train["card2"].apply(lambda x: str(x)) +                "_" + train["card3"].apply(lambda x: str(x)) + "_" + train["card4"].apply(lambda x: str(x)) +                "_" + train["card5"].apply(lambda x: str(x)) + "_" + train["card6"].apply(lambda x: str(x))

test['card_str'] = test["card1"].apply(lambda x: str(x)) + "_" + test["card2"].apply(lambda x: str(x)) +                "_" + test["card3"].apply(lambda x: str(x)) + "_" + test["card4"].apply(lambda x: str(x)) +                "_" + test["card5"].apply(lambda x: str(x)) + "_" + test["card6"].apply(lambda x: str(x))

train['card_count_full'] = train['card_str'].map(pd.concat([train['card_str'], test['card_str']], ignore_index=True).value_counts(dropna=False))
test['card_count_full'] = test['card_str'].map(pd.concat([test['card_str'], test['card_str']], ignore_index=True).value_counts(dropna=False))


# In[15]:


train['TransactionAmt_to_std_card_str'] = train['TransactionAmt'] / train.groupby(['card_str'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card_str'] = test['TransactionAmt'] / test.groupby(['card_str'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_mean_card_str'] = train['TransactionAmt'] / train.groupby(['card_str'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card_str'] = test['TransactionAmt'] / test.groupby(['card_str'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_sum_card_str'] = train['TransactionAmt'] / train.groupby(['card_str'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card_str'] = test['TransactionAmt'] / test.groupby(['card_str'])['TransactionAmt'].transform('sum')


# In[16]:


train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

train['card2_count_full'] = train['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))
test['card2_count_full'] = test['card2'].map(pd.concat([train['card2'], test['card2']], ignore_index=True).value_counts(dropna=False))

train['card3_count_full'] = train['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))
test['card3_count_full'] = test['card3'].map(pd.concat([train['card3'], test['card3']], ignore_index=True).value_counts(dropna=False))

train['card4_count_full'] = train['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))
test['card4_count_full'] = test['card4'].map(pd.concat([train['card4'], test['card4']], ignore_index=True).value_counts(dropna=False))

train['card5_count_full'] = train['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))
test['card5_count_full'] = test['card5'].map(pd.concat([train['card5'], test['card5']], ignore_index=True).value_counts(dropna=False))

train['card6_count_full'] = train['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))
test['card6_count_full'] = test['card6'].map(pd.concat([train['card6'], test['card6']], ignore_index=True).value_counts(dropna=False))


# In[17]:


train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card6'] = train['TransactionAmt'] / train.groupby(['card6'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card6'] = test['TransactionAmt'] / test.groupby(['card6'])['TransactionAmt'].transform('mean')


# In[18]:


train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card6'] = train['TransactionAmt'] / train.groupby(['card6'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card6'] = test['TransactionAmt'] / test.groupby(['card6'])['TransactionAmt'].transform('std')


# In[19]:


train['TransactionAmt_to_sum_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('sum')
train['TransactionAmt_to_sum_card2'] = train['TransactionAmt'] / train.groupby(['card2'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card2'] = test['TransactionAmt'] / test.groupby(['card2'])['TransactionAmt'].transform('sum')

train['TransactionAmt_to_sum_card3'] = train['TransactionAmt'] / train.groupby(['card3'])['TransactionAmt'].transform('sum')
train['TransactionAmt_to_sum_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card3'] = test['TransactionAmt'] / test.groupby(['card3'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('sum')

train['TransactionAmt_to_sum_card5'] = train['TransactionAmt'] / train.groupby(['card5'])['TransactionAmt'].transform('sum')
train['TransactionAmt_to_sum_card6'] = train['TransactionAmt'] / train.groupby(['card6'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card5'] = test['TransactionAmt'] / test.groupby(['card5'])['TransactionAmt'].transform('sum')
test['TransactionAmt_to_sum_card6'] = test['TransactionAmt'] / test.groupby(['card6'])['TransactionAmt'].transform('sum')


# In[20]:


train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')


# In[21]:


train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')


# ### card svd特征

# In[22]:


from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

def get_dc_feature(df_train, df_test, n_comp=12, used_features=None):
    """
    构造分解特征
    """
    if not used_features:
        used_features = df_test.columns
        
    train = df_train.copy()
    test  = df_test.copy()    
        
    # tSVD
#     tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
#     tsvd_results_train = tsvd.fit_transform(train[used_features])
#     tsvd_results_test = tsvd.transform(test[used_features])

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train[used_features])
    pca2_results_test = pca.transform(test[used_features])

#     # ICAz
#     ica = FastICA(n_components=n_comp, random_state=420)
#     ica2_results_train = ica.fit_transform(train[used_features])
#     ica2_results_test = ica.transform(test[used_features])

#     # GRP
#     grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
#     grp_results_train = grp.fit_transform(train[used_features])
#     grp_results_test = grp.transform(test[used_features])

#     # SRP
#     srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
#     srp_results_train = srp.fit_transform(train[used_features])
#     srp_results_test = srp.transform(test[used_features])
    
    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

#         train['ica_' + str(i)] = ica2_results_train[:, i - 1]
#         test['ica_' + str(i)] = ica2_results_test[:, i - 1]

#         train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#         test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

#         train['grp_' + str(i)] = grp_results_train[:, i - 1]
#         test['grp_' + str(i)] = grp_results_test[:, i - 1]

#         train['srp_' + str(i)] = srp_results_train[:, i - 1]
#         test['srp_' + str(i)] = srp_results_test[:, i - 1]

    return train, test

print(train.shape, test.shape)
used_features = ['card1', 'card2','card3', 'card5']

train[used_features] = train[used_features].fillna(-1.0)
test[used_features]  = test[used_features].fillna(-1.0)

train, test = get_dc_feature(train, test, n_comp=3, used_features=used_features)
print(train.shape, test.shape)


# ### address
# both are for purchaser, addr1 as billing region, addr2 as billing country

# In[23]:


train['addr1_count_full'] = train['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))
test['addr1_count_full'] = test['addr1'].map(pd.concat([train['addr1'], test['addr1']], ignore_index=True).value_counts(dropna=False))

train['addr2_count_full'] = train['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))
test['addr2_count_full'] = test['addr2'].map(pd.concat([train['addr2'], test['addr2']], ignore_index=True).value_counts(dropna=False))

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')


# ### distance

# In[24]:


train["dist1_plus_dist2"]  = train["dist1"] + train["dist2"]
train["dist1_minus_dist2"] = train["dist1"] - train["dist2"]
train["dist1_times_dist2"]  = train["dist1"] * train["dist2"]
train["dist1_divides_dist2"] = train["dist1"] / train["dist2"]

test["dist1_plus_dist2"]  = test["dist1"] + test["dist2"]
test["dist1_minus_dist2"] = test["dist1"] - test["dist2"]
test["dist1_times_dist2"]  = test["dist1"] * test["dist2"]
test["dist1_divides_dist2"] = test["dist1"] / test["dist2"]


# ### 邮箱

# In[25]:


def transform_email(df):
    for col in ['P_emaildomain', 'R_emaildomain']:
        col1 = col.replace('domain', '_suffix')
        df[col1] = df[col].str.rsplit('.', expand=True).iloc[:, -1]
        
        col2 = col.replace('domain', 'Corp')
        df[col2] = df[col]
        df.loc[df[col].isin(['gmail.com', 'gmail']), col2] = 'Google'
        df.loc[df[col].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk', 'yahoo.co.jp', 
                             'yahoo.de', 'yahoo.fr', 'yahoo.es', 'yahoo.com.mx', 
                             'ymail.com']), col2] = 'Yahoo'
        df.loc[df[col].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es', 
                             'hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr', 
                             'hotmail.fr']), col2] = 'Microsoft'
        df.loc[df[col].isin(['aol.com', 'verizon.net']), col2] = 'Verizon'
        df.loc[df[col].isin(['att.net', 'sbcglobal.net', 'bellsouth.net']), col2] = 'AT&T'
        df.loc[df[col].isin(['icloud.com', 'mac.com', 'me.com']), col2] = 'Apple'
        df.loc[df[col2].isin(df[col2].value_counts()[df[col2].value_counts() <= 1000].index), col2] = 'Others'
    
    return df


# In[26]:


train = transform_email(train)
test = transform_email(test)


# In[27]:


# train['P_email']=(train['P_emaildomain']=='xmail.com')
# train['R_email']=(train['R_emaildomain']=='xmail.com')
# test['P_email']=(test['P_emaildomain']=='xmail.com')
# test['R_email']=(test['R_emaildomain']=='xmail.com')


# ### C1-C14
# 
# C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
# 

# ### D1-D15
# timedelta, such as days between previous transaction, etc.

# ### M1-M9
# match, such as names on card and address, etc.

# In[28]:


MFeatures = ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]
for feature in MFeatures:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


# ### Vxxx
# Vesta engineered rich features, including ranking, counting, and other entity relations.

# ### id相关特征

# In[29]:


# # lastest_browser

# a = np.zeros(train.shape[0])
# train["lastest_browser"] = a
# a = np.zeros(test.shape[0])
# test["lastest_browser"] = a

# def browser(df):
#     df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
#     df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
#     return df

# train=browser(train)
# test=browser(test)


# In[30]:


def transform_id_cols(df):
#     df['id_01_cut'] = pd.cut(df['id_01'], bins=[-100, -30, -20, -10, -5, 0])
    
    df['id_05_d'] = df['id_05']
    df['id_05_d'].where(df[df['id_05_d'].notnull()]['id_05_d'] == 0, 1, inplace=True)
    
#     df['id_06_cut'] = pd.cut(df['id_06'], bins=[-100, -10, -5, 0])
    df['id_06_d'] = df['id_06']
    df['id_06_d'].where(df[df['id_06_d'].notnull()]['id_06_d'] == 0, 1, inplace=True)
    
    # Dealing with id_30
    df['id_30_count'] = df['id_30'].map(df['id_30'].value_counts(dropna=False))
    df['System'] = df['id_30'].astype('str').str.split('.', expand=True)[0].str.split('_', expand=True)[0]
    df['SystemCorp'] = df['System'].str.split(expand=True)[0]
    
    # Dealing with id_31
    df['LastestBrowser'] = df['id_31']
    df.loc[df['LastestBrowser'].isin(['samsung browser 7.0', 'opera 53.0', 'mobile safari 10.0', 'chrome 63.0 for android', 
                                       'google search application 49.0', 'firefox 60.0', 'edge 17.0', 'chrome 69.0', 
                                       'chrome 67.0 for android', 'chrome 64.0', 'chrome 63.0 for ios', 'chrome 65.0', 
                                       'chrome 64.0 for android', 'chrome 64.0 for ios', 'chrome 66.0', 
                                       'chrome 65.0 for android', 'chrome 65.0 for ios', 'chrome 66.0 for android', 
                                       'chrome 66.0 for ios']), 'LastestBrowser'] = 1
    df.loc[df['LastestBrowser'].str.len() > 1, 'LastestBrowser'] = 0
    
    df['id_31_count'] = df['id_31'].map(df['id_31'].value_counts(dropna=False))
    
    df['MSBrowser'] = df['id_31'].str.contains('edge|ie|microsoft', case=False) * 1
    df['AppleBrowser'] = df['id_31'].str.contains('safari', case=False) * 1
    df['GoogleBrowser'] = df['id_31'].str.contains('chrome', case=False) * 1
    
    df['BrowserType'] = df['id_31']
    df.loc[df['BrowserType'].str.contains('samsung', case=False, na=False), 'BrowserType'] = 'Samsung'
    df.loc[df['BrowserType'].str.contains('safari', case=False, na=False), 'BrowserType'] = 'Apple'
    df.loc[df['BrowserType'].str.contains('chrome|google', case=False, na=False), 'BrowserType'] = 'Google'
    df.loc[df['BrowserType'].str.contains('firefox', case=False, na=False), 'BrowserType'] = 'Mozilla'
    df.loc[df['BrowserType'].str.contains('edge|ie|microsoft', case=False, na=False, regex=True), 'BrowserType'] = 'Microsoft'
    df.loc[df['BrowserType'].isin(df['BrowserType'].value_counts()                                      [df['BrowserType'].value_counts()< 1000].index), ['BrowserType']] = 'other'
    
    # Dealing with id_33
    df['id_33_count'] = df['id_33'].map(df['id_33'].value_counts(dropna=False))
    df['DisplaySize'] = df['id_33'].str.split('x', expand=True)[0].astype('float')                            * df['id_33'].str.split('x', expand=True)[1].astype('float')
    df['DisplaySize'].replace(0, np.nan, inplace=True)
    df['DisplaySize'] = (df['DisplaySize'] / df['DisplaySize'].min()).round(0)
    
    # Try easy combining
    for feature in ['id_02__id_20', 'id_13__id_17', 'id_02__D8', 'D11__DeviceInfo', 
                    'DeviceInfo__P_emaildomain', 'card2__dist1', 'card1__card5', 
                    'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:
        f1, f2 = feature.split('__')
        df[feature] = df[f1].astype(str) + '_' + df[f2].astype(str)
    for col in ['id_30', 'id_31', 'id_33', 'DeviceInfo']:
        df[col + '_DeviceTpye'] = train[col] + '_' + train['DeviceType']
           
    return df


# In[31]:


train = transform_id_cols(train)
test  = transform_id_cols(test)


# ### DeviceType

# In[32]:


# ### DeviceInfo

# In[33]:


def transform_DeviceInfo(df):
    df['DeviceCorp'] = df['DeviceInfo']
    df.loc[df['DeviceInfo'].str.contains('HUAWEI|HONOR', case=False, na=False, regex=True), 'DeviceCorp'] = 'HUAWEI'
    df.loc[df['DeviceInfo'].str.contains('OS', na=False, regex=False), 'DeviceCorp'] = 'APPLE'
    df.loc[df['DeviceInfo'].str.contains('Idea|TA', case=False, na=False), 'DeviceCorp'] = 'Lenovo'
    df.loc[df['DeviceInfo'].str.contains('Moto|XT|Edison', case=False, na=False), 'DeviceCorp'] = 'Moto'
    df.loc[df['DeviceInfo'].str.contains('MI|Mi|Redmi', na=False), 'DeviceCorp'] = 'Mi'
    df.loc[df['DeviceInfo'].str.contains('VS|LG|EGO', na=False), 'DeviceCorp'] = 'LG'
    df.loc[df['DeviceInfo'].str.contains('ONE TOUCH|ALCATEL', case=False, na=False, regex=False), 'DeviceCorp'] = 'ALCATEL'
    df.loc[df['DeviceInfo'].str.contains('ONE A', na=False, regex=False), 'DeviceCorp'] = 'ONEPLUS'
    df.loc[df['DeviceInfo'].str.contains('OPR6', na=False, regex=False), 'DeviceCorp'] = 'HTC'
    df.loc[df['DeviceInfo'].str.contains('Nexus|Pixel', case=False, na=False, regex=True), 'DeviceCorp'] = 'google'
    df.loc[df['DeviceInfo'].str.contains('STV', na=False, regex=False), 'DeviceCorp'] = 'blackberry'
    df.loc[df['DeviceInfo'].str.contains('ASUS', case=False, na=False, regex=False), 'DeviceCorp'] = 'ASUS'
    df.loc[df['DeviceInfo'].str.contains('BLADE', case=False, na=False, regex=False), 'DeviceCorp'] = 'ZTE'
    
    df['DeviceCorp'] = df['DeviceInfo'].astype('str').str.split(':', expand=True)[0].                                str.split('-', expand=True)[0].str.split(expand=True)[0]
    
    df.loc[df['DeviceInfo'].isin(['rv', 'SM', 'GT', 'SGH']), 'DeviceCorp'] = 'SAMSUNG'
    df.loc[df['DeviceInfo'].str.startswith('Z', na=False), 'DeviceCorp'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.startswith('KF', na=False), 'DeviceCorp'] = 'Amazon'
    
    for i in ['D', 'E', 'F', 'G']:
        df.loc[df['DeviceInfo'].str.startswith(i, na=False), 'DeviceCorp'] = 'SONY'

    minority = df['DeviceCorp'].value_counts()[df['DeviceCorp'].value_counts() < 100].index
    df.loc[df['DeviceCorp'].isin(minority), 'DeviceCorp'] = 'Other'
    df['DeviceCorp'] = df['DeviceCorp'].str.upper()
    
    return df


# In[34]:


train = transform_DeviceInfo(train)
test  = transform_DeviceInfo(test)


# ### 类别型变量labelEncoder

# In[35]:


target = "isFraud"
# Label Encoding
for f in tqdm_notebook([feature for feature in train.columns if feature != target]):
    if train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = LabelEncoder()
        temp = pd.DataFrame(train[f].astype(str).append(test[f].astype(str)))
        lbl.fit(temp[f])
        train[f] = lbl.transform(list(train[f].astype(str)))
        test[f] = lbl.transform(list(test[f].astype(str))) 


# ### 构造特征

# In[36]:


def transform_number(df):
    df['id_02_log'] = np.log10(df['id_02'])
    
    df['C5_d'] = df['C5']
    df['C5_d'].where(df['C5'] == 0, 1, inplace=True)
    
    df['D8_mul_D9'] = df['D8'] * df['D9']
    
    df['TransAmt_mul_dist1'] = df['TransactionAmt'] * df['dist1']
    df['TransAmt_per_TransDT'] = df['TransactionAmt'] * 24 * 60 * 60 / df['TransactionDT']
    
    return df


# In[37]:


train = transform_number(train)
test  = transform_number(test)


# ### 交叉特征
for feature in tqdm_notebook(['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 
                              'P_emaildomain__C2', 
                              'P_emaildomain__card1', 'P_emaildomain__card2',
                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 
                              'addr1__card1', 'card2__card4', 'card4__card6'
                             ]):

    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))



# ### 高阶交叉特征

for feature in tqdm_notebook([
                              'P_emaildomain__card1__card2', 'addr1__card1__card2'
                             ]):

    f1, f2, f3 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str) + '_' + train[f3].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str) + '_' + test[f3].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))




X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
test_X = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)


# X.to_csv("../temp/feature_X.csv", index = False)
# y.to_csv("../temp/feature_y.csv", index = False)
# test_X.to_csv("../temp/feature_test_X.csv", index = False)


# ### lightgbm参数

print("lgb model")
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }


n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
lgb_sub = sub.copy()
lgb_sub['isFraud'] = 0
aucs = []
training_start_time = time()
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

    start_time = time()
    print('Training on fold {}'.format(fold_n + 1))

    trn_data = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    val_data = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])
    clf = lgb.train(params, trn_data, num_boost_round=10000, valid_sets=[val_data], verbose_eval=100,
                    early_stopping_rounds=500)

    pred = clf.predict(test_X)
    val = clf.predict(X.iloc[valid_index])
    print('ROC accuracy: {}'.format(roc_auc_score(y.iloc[valid_index], val)))
    aucs.append(roc_auc_score(y.iloc[valid_index], val))
    lgb_sub['isFraud'] = lgb_sub['isFraud'] + pred / n_fold

    print('Fold {} finished in {}'.format(fold_n + 1, str(datetime.timedelta(seconds=time() - start_time))))


subname = '../label/ieee_lgb_kflod.csv'
lgb_sub.to_csv(subname, index=False)
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


# 真实效果
test1 = pd.read_csv('../temp/test1_label.csv', usecols=["TransactionID", "isFraud"])
test2 = pd.read_csv('../temp/test2_label.csv', usecols=["TransactionID", "isFraud"])
pre = pd.read_csv(subname)

df1 = test1.merge(pre, on="TransactionID", how="left")
print("test1 auc: ", roc_auc_score(df1["isFraud_x"], df1["isFraud_y"]))

df = test2.merge(pre, on="TransactionID", how="left")
print("test2 auc:", roc_auc_score(df["isFraud_x"], df["isFraud_y"]))

# # 结果记录

# - file/                 线下mean/线下fold5/线上test1/线上test2
# - ieee_lgb_label.csv/   0.9276/0.9378/0.9032/0.9057   -不带label数据
# - ieee_lgb_label_50.csv/0.9281/0.9394/0.9027/0.9056   -迭代次数1.0倍,增加50条样本
# - ieee_lgb_label_50.csv/0.9281/0.9394/0.9034/0.9058   -迭代次数1.2倍,增加50条样本(有效!)
# - ieee_lgb_label_50.csv/0.9278/0.9403/0.8981/0.9049   -增加12000条样本
# - ieee_lgb_label_50.csv/0.9278/0.9403/0.8987/0.9050   -增加12000条样本,1.0倍
#

# split=2 线下验证集
# (第一折是有效的?,在test1上有效,test2上失效...)
# 第一折/第二折/mean/线上test1/线上test2
# valid_0's auc: 0.9061, valid_0's auc: 0.9260, Mean:0.9160, test1:0.9013, test2:0.9054  -不带label数据, 1.0倍
# valid_0's auc: 0.9061, valid_0's auc: 0.9260, Mean:0.9160, test1:0.9022, test2:0.9057  -不带label数据, 1.1倍
# valid_0's auc: 0.9061, valid_0's auc: 0.9260, Mean:0.9160, test1:0.9028, test2:0.9059  -不带label数据, 1.2倍
# valid_0's auc: 0.9085, valid_0's auc: 0.9262, Mean:0.9174, test1:0.9028, test2:0.9052  -不带label数据, 1.2倍参数优化
# valid_0's auc: 0.9061, valid_0's auc: 0.9260, Mean:0.9160, test1:0.9031, test2:0.9058  -不带label数据, 1.3倍
# valid_0's auc: 0.9061, valid_0's auc: 0.9260, Mean:0.9160, test1:0.9032, test2:0.9057  -不带label数据, 1.4倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9015, test2:0.9046  -不带label数据, shift特征, 1.0倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9020, test2:0.9048  -不带label数据, shift特征, 1.1倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9026, test2:0.9049  -不带label数据, shift特征, 1.2倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9030, test2:0.9050  -不带label数据, shift特征, 1.3倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9033, test2:0.9051  -不带label数据, shift特征, 1.4倍
# valid_0's auc: 0.9071, valid_0's auc: 0.9264, Mean:0.9168, test1:0.9035, test2:0.9051  -不带label数据, shift特征, 1.5倍
# valid_0's auc: 0.9042, valid_0's auc: 0.9303, Mean:0.9172, test1:0.8986, test2:0.9048  -带label数据

# kfold
# Mean AUC/线上test1/线上test2
#   0.9714, 0.9043, 0.9063   - 原始
#   0.9802, 0.8687, 0.8620   - target encoding


# 比例
# 第一折:test1:test2
# 19:27:6

# nohup python -u lgb2_kfold.py 1.1 > kfold.log 2>&1 &
# nohup python -u label_lgb2.py > split_2_label.log 2>&1 &
# nohup python -u label_lgb2.py > split_2_shift.log 2>&1 &
# nohup python -u label_lgb2.py > split_2_shift_1.1.log 2>&1 &
# nohup python -u label_lgb2.py 1.5 > split_2_shift_1.5.log 2>&1 &
# nohup python -u label_lgb2_custom_kfold.py 1.2 > unfeature_para.log 2>&1 &
# nohup python -u label_lgb2_custom_kfold.py 1.4 > split_2_shift_1.4_unfeature.log 2>&1 &

