# coding: utf-8
import pandas as pd
pd.set_option("display.max_columns", 500)
import plotly.offline as py
py.init_notebook_mode(connected=True)
from tqdm import tqdm_notebook
import gc
import warnings
warnings.filterwarnings('ignore')

NROWS = None
# NROWS = 50000

train_identity = pd.read_csv('../input/train_identity.csv', nrows=NROWS)
train_transaction = pd.read_csv('../input/train_transaction.csv', nrows=NROWS)
train = train_transaction.merge(train_identity, how='left', on='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', nrows=NROWS)
test_transaction = pd.read_csv('../input/test_transaction.csv', nrows=NROWS)
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

# sub = pd.read_csv('../input/sample_submission.csv', nrows=NROWS)

gc.enable()
del train_identity, train_transaction
del test_identity, test_transaction
gc.collect()

print("train.shape:", train.shape)
print("test.shape:", test.shape)


target = "isFraud"

test[target] = -1

df = train.append(test)
df.reset_index()

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) + "_" + df["card3"].apply(lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) + "_" + df["card5"].apply(lambda x: str(x)) + "_" + df["card6"].apply(lambda x: str(x)) + "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(lambda x: str(x))
H_move = 12
df["day"] = (df["TransactionDT"] + 3600 * H_move) // (24 * 60 * 60)

feature_list = ["uid", target, "D15", "day", "TransactionDT", "TransactionID"]

# day有可能是 7-2,也有可能是7-2-1
# D15 = int(delta秒/3600/24) 不对 3,16 bad case
# D15 = round(delta秒/3600/24)

# 如果是D15==0,一天内只有一笔交易的话,不能用
uid_D15 = []
for DAY in tqdm_notebook(range(32, 182 + 1)):  # 2, 182+1
    print("DAY: ", DAY)
    for D15 in range(31, min(121, DAY)):  # 1, DAY
        uid_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "uid"].values)
        TransactionID_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "TransactionID"].values)

        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            mean_ = 0
            sum_ = 0
            cnt_ = 0
            temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]

            if temp.shape[0] != 0:
                mean_ = temp["isFraud"].mean()
                sum_ = temp["isFraud"].sum()
                cnt_ = temp["isFraud"].shape[0]

            uid_D15.append([TransactionID_, mean_, sum_, cnt_])

uid_D15 = pd.DataFrame(uid_D15)
uid_D15.columns = ["TransactionID", "mean_", "sum_", "cnt_"]
uid_D15.to_csv("./uid_D15_train_modify.csv",index=False)


# ### 测试集特征构造
uid_D15_test = []

for DAY in tqdm_notebook(range(213, 395 + 1)):
    print("DAY: ", DAY)
    for D15 in range(DAY - 182, min(121, DAY - 1 + 1)):  # 212
        uid_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "uid"].values)
        TransactionID_list = list(df.loc[(df["D15"] == D15) & (df["day"] == DAY), "TransactionID"].values)

        for i in range(len(uid_list)):

            TransactionID_ = TransactionID_list[i]
            mean_ = 0
            sum_ = 0
            cnt_ = 0

            temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D15), feature_list]

            if temp.shape[0] != 0:
                mean_ = temp["isFraud"].mean()
                sum_ = temp["isFraud"].sum()
                cnt_ = temp["isFraud"].shape[0]

            uid_D15_test.append([TransactionID_, mean_, sum_, cnt_])


uid_D15_test = pd.DataFrame(uid_D15_test)
uid_D15_test.columns = ["TransactionID", "mean_", "sum_", "cnt_"]
uid_D15_test.to_csv("./uid_D15_test_modify.csv",index=False)


