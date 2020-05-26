import pandas as pd
pd.set_option("display.max_columns", 500)
import plotly.offline as py
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

templist = [3328834,
 3329764,
 3329770,
 3329777,
 3329812,
 3329992,
 3330058,
 3330066,
 3330098,
 3330106,
 3330108,
 3330110,
 3330112,
 3330113,
 3330117,
 3330293,
 3330361,
 3330372,
 3330377,
 3330388,
 3330666,
 3330924,
 3330937,
 3330981,
 3331001,
 3331006,
 3331606,
 3331607,
 3332000,
 3332003,
 3332007,
 3332009,
 3332010,
 3332013,
 3332014,
 3335059,
 3335160,
 3336013,
 3336360,
 3337046,
 3337056,
 3339589,
 3339822,
 3339870,
 3339884,
 3339904,
 3341135,
 3341165,
 3341172,
 3341187]

train = pd.read_csv('../temp/train_label.csv')
print("train.shape", train.shape)

# test = pd.read_csv('../temp/test_label.csv')
# print("test.shape", test.shape)

test1 = pd.read_csv('../temp/test1_label.csv')

append_test = test1.loc[test1.TransactionID >= templist[0]]
append_test["isFraud"] = 0.0
append_test.loc[append_test.TransactionID.isin(templist), "isFraud"] = 1.0
print("append_test.shape", append_test.shape)

train = train.append(append_test)
print("train.shape after append", train.shape)

train.to_csv('../temp/train_label_50.csv', index=False)