import pandas as pd
pd.set_option("display.max_columns", 500)
import plotly.offline as py
py.init_notebook_mode(connected=True)
from tqdm import tqdm_notebook
import gc
import warnings
warnings.filterwarnings('ignore')

NROWS = None

train_identity = pd.read_csv('../input/train_identity.csv', nrows=NROWS)
train_transaction = pd.read_csv('../input/train_transaction.csv', nrows=NROWS)
train = train_transaction.merge(train_identity, how='left', on='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', nrows=NROWS)
test_transaction = pd.read_csv('../input/test_transaction.csv', nrows=NROWS)
test = test_transaction.merge(test_identity, how='left', on='TransactionID')

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

df['uid'] = df["card1"].apply(lambda x: str(x)) + "_" + df["card2"].apply(lambda x: str(x)) +\
                "_" + df["card3"].apply(lambda x: str(x)) + "_" + df["card4"].apply(lambda x: str(x)) +\
                "_" + df["card5"].apply(lambda x: str(x)) + "_" + df["card6"].apply(lambda x: str(x)) +\
                "_" + df["addr1"].apply(lambda x: str(x)) + "_" + df["addr2"].apply(lambda x: str(x))

df["day"] = (df["TransactionDT"] + 3600 * 12) // (24 * 60 * 60)
feature_list = ["uid", target, "D10", "day", "TransactionDT", "TransactionID"]


fraud_TransactionIDs = []

for DAY in tqdm_notebook(range(213, 395 + 1)):
    print("DAY: ", DAY)
    for D10 in range(DAY - 182, DAY - 1):  # 212
        uid_list = list(df.loc[(df["D10"] == D10) & (df["day"] == DAY) & (df["isFraud"] == -1), "uid"].values)
        TransactionID_list = list(
            df.loc[(df["D10"] == D10) & (df["day"] == DAY) & (df["isFraud"] == -1), "TransactionID"].values)
        # print(TransactionID_list)
        for i in range(len(uid_list)):
            TransactionID_ = TransactionID_list[i]
            mean_ = 0
            sum_ = 0
            cnt_ = 0
            temp = df.loc[(df["uid"] == uid_list[i]) & (df["day"] == DAY - D10), feature_list]
            if temp.shape[0] != 0:
                mean_ = temp["isFraud"].mean()
                sum_ = temp["isFraud"].sum()
                cnt_ = temp["isFraud"].shape[0]
            fraud_TransactionIDs.append([TransactionID_, mean_, sum_, cnt_])


fraud_TransactionIDs = pd.DataFrame(fraud_TransactionIDs)
fraud_TransactionIDs.columns = ["TransactionID", "mean", "sum", "cnt"]
fraud_TransactionIDs.to_csv("./fraud_TransactionIDs.csv", index=False)


