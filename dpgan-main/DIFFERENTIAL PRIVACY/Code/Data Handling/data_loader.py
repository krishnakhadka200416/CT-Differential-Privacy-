from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import  transforms
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
import numpy as np

train_df=pd.read_csv("/kaggle/input/fraud-detection/fraudTrain.csv")
train_df.head()

test_df=pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv")




test_df = test_df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'category',
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
        'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
        ])




train_df = train_df.drop(columns=['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'category',
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
        'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
    ])



X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values


X_train=X_train[:1000:]
y_train=y_train[:1000]
X_test=X_test[:200:]
y_test=y_test[:200]

X_train.shape,X_test.shape

train_df.columns

X_test






y_train=np.array(y_train)
y_test=np.array(y_test)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train=np.array(X_train)
X_test=np.array(X_test)

input_size,num_features=X_train.shape[0],X_train[1]


train_data=torch.tensor(X_train,dtype=torch.float32)
test_data=torch.tensor(X_test,dtype=torch.float32)
train_labels=torch.tensor(y_train,dtype=torch.float32)
test_labels=torch.tensor(y_test,dtype=torch.float32)



class FraudDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data,self.labels

train_dataset = FraudDataset(train_data,train_labels)
test_dataset = FraudDataset(test_data,test_labels)

batch_size = 64  # Specify the batch size for training

def load_data():

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader,test_dataloader
