#!/usr/bin/env python

# %%
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
root = '/home/felipe/CFAA/Power'
machine = 'GMTK'
# machine = 'IBARMIA_ml'
db_file = f'federated_working_{machine}.csv'
db_file = os.path.join(root, db_file)

db = pd.read_csv(db_file)
# Remove columns that are not useful.
db = db.drop(columns=['precioPorKW', 'override_SPINDLE'])

x_var = [
    'load_X',
    'load_Z',
    'power_X',
    'power_Z',
    'load_SPINDLE',
    'speed_SPINDLE',
    'powerDrive_SPINDLE'
]
# GMTK: 'load_Z' and 'load_SPINDLE' are correlated.
# GMTK: 'power_X' and 'power_Z': in average, if one is zero, the other no.
# GMTK: 'override_SPINDLE' is not very informative.
# GMTK: 'power_Z' and 'speed_SPINDLE' are correlated.

Y_var = 'potenciaKW'

# Q1 = db.quantile(0.25)
# Q3 = db.quantile(0.75)
# IQR = Q3 - Q1
# print(f'N. samples before outlier remotion: {len(db)}')
# # Remove outliers.
# db = db[~((db < (Q1 - 3 * IQR)) | (db > (Q3 + 3 * IQR))).any(axis=1)]
# print(f'N. samples after outlier remotion: {len(db)}')

# remove rows where 'power_X' and 'power_Z' are both zero.
# db = db.loc[db['power_X'].abs() < 50]
# db_Z = db.loc[db['power_Z'].abs() > 50]

db_pred = []
if machine == 'GMTK':
    # db = db[(db['load_X'] > 10) & (db['load_X'] < 30)]
    db = db[(db['load_Z'] > 12.5) & (db['load_Z'] < 14)]
    db = db[(db['power_X'] > -10) & (db['power_X'] < 10)]
    # db = db[(db['power_Z'] > -5000) & (db['power_Z'] < 8000)]
    # db = db[(db['load_SPINDLE'] > -0.1) & (db['load_SPINDLE'] < 25)]
    # db = db[(db['speed_SPINDLE'] > -10) & (db['speed_SPINDLE'] < 10)]
    db = db[(db['powerDrive_SPINDLE'] > -0.1) & (db['powerDrive_SPINDLE'] < 0.2)]
    # db = db.loc[(db['load_Z'] > 12) & (db['load_Z'] < 15)]
    ranges = {0: (0.8, 1), 1: (3.5, 3.7)}
    for i, (low, high) in ranges.items():
        db_pred.append(db.loc[(db[Y_var] > low) & (db[Y_var] <= high)])

# %%
fig, axs = plt.subplots(nrows=3, ncols=3)
vars = ['load_X', 'load_Z', 'power_X']
colors = ['r', 'g', 'b']
for i, j in np.ndindex(3, 3):
    if i > j:
        for k, db_p in enumerate(db_pred):
            db_p.plot(
                x=vars[j], y=vars[i],
                kind='scatter',
                color=colors[k],
                ax=axs[i, j],
                label=f'{k}'
            )
    elif i == j:
        db.hist(column=vars[i], bins=500, ax=axs[i, j])
plt.show()

# %%
# Plot histograms in a single fig.
db.hist(column=x_var, bins=500)
plt.suptitle(machine)
plt.show()

# %%
# Select rows randomly from database.
n = 100
cols = x_var + [Y_var]
colors = ['r', 'g', 'b']
# Concatenate the columns in a single row.
for i, sub in enumerate(db_pred):
    for _, s in sub.sample(n, random_state=123456).iterrows():
        row = s[cols]
        plt.plot(
            cols,
            row,
            color=colors[i],
            alpha=0.05
        )
plt.show()

# %%
# NN
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


class PowerDataset(torch.utils.data.Dataset):
   
    def __init__(self, db):
        X, y = db[x_var].values, db[Y_var].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(3, 10),
      nn.ReLU(),
      nn.Linear(10, 1),
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)
  
# %%
# Shuffle db.
db = db.sample(frac=1).reset_index(drop=True)
# Divide db in train and test.
train_size = int(0.8 * len(db))
test_size = len(db) - train_size
# Partition pandas dataframe into train and test.
train_db = db.sample(train_size)
test_db = db.drop(train_db.index)
train_db = PowerDataset(train_db)
test_db = PowerDataset(test_db)
trainloader = torch.utils.data.DataLoader(
    train_db, batch_size=10, shuffle=True
)

mlp = MLP()
loss_f = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

for epoch in range(0, 4): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    mlp.train()
    # Set current loss value
    # current_loss = 0.0
    total_loss = 0.0
    test_loss = 0.0
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader):
        X, y = data
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
        y_pred = mlp(X)
        # Compute loss
        loss = loss_f(y_pred, y)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Print statistics
        # current_loss += loss.item()
        total_loss += loss.item()
        # if (i > 0) and (i % 100 == 0):
        #     print('Loss after mini-batch %5d: %.4f' %
        #           (i + 1, current_loss))
        #     current_loss = 0.0
    print(f'Total loss: {total_loss / len(train_db):.4f}')

    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(test_db.X)
        loss = loss_f(y_pred, test_db.y).item()
        print(f'Test loss: {loss / len(test_db):.4f}')

# Process is complete.
print('Training process has finished.')