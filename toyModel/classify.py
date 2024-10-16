#!/usr/bin/env python

# %%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
file = 'fake_dataset_merged.csv'
df = pd.read_csv(file)
# Divide db in train and test.
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
# Partition pandas dataframe into train and test.
train_df = df.sample(train_size)
test_df = df.drop(train_df.index)

# %%
class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, db):
        X, y = db[['x', 'y']].values, db['label'].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyClassifier(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 5),
      nn.ReLU(),
      nn.Linear(5, 3),
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)

train_df = MyDataset(train_df)
test_df = MyDataset(test_df)
trainloader = torch.utils.data.DataLoader(
    train_df, batch_size=10, shuffle=True
)

model = MyClassifier()
loss_f = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

# %%
# ==== Train on the merged dataset ====
n_epochs = 15
model.train()
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    print(f'Starting epoch {epoch+1}')
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        X, y = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        y_pred = model(X)
        loss = loss_f(y_pred, y)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
# ==== Test model ====
model.eval()
with torch.no_grad():
    X, y = test_df.X, test_df.y
    y_pred = model(X)
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).sum().item()

print(f'Accuracy: {100 * correct / y.size(0):.2f} %')