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
db = db.drop(columns=['precioPorKW', 'override_SPINDLE', 'power_X'])

x_var = [
    'load_X',
    'load_Z',
    # 'power_X',
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

if machine == 'GMTK':
    # db = db[(db['load_X'] > 10) & (db['load_X'] < 30)]
    ranges = {0: (-np.inf, 1), 1: (1, 4.118), 2: (4.118, np.inf)}
    for i, (low, high) in ranges.items():
        db.loc[(db[Y_var] > low) & (db[Y_var] <= high), 'potenciaKW'] = i

# %%
fig, axs = plt.subplots(nrows=3, ncols=3)
vars = ['load_X', 'load_Z', 'power_Z']
colors = ['r', 'g', 'b']
for i, j in np.ndindex(3, 3):
    if i > j:
        for k in range(3):
            db[(db['potenciaKW'] == k)].plot(
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
# ==== Linear classification ====
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Only select two classes.
tmp = db[(db['potenciaKW'] == 0) | (db['potenciaKW'] == 2)]
# Split data set into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    tmp[x_var], tmp[Y_var], test_size=0.2, random_state=12345
)
# Rescale data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train classifier.
clf = LinearSVC()
clf.fit(X_train, y_train)
# Test classifier.
X_test = scaler.transform(X_test)
score = clf.score(X_test, y_test)
print(f'Accuracy: {score:.4f}')

del tmp

# %%
# === Change coordinate system and draw separating line ===
dim = len(x_var)
w = clf.coef_
b = clf.intercept_
basis = np.eye(dim)
basis[:, 0] = w / np.linalg.norm(w)
sign = w[0, 0] / np.linalg.norm(w)
basis = np.linalg.qr(basis)[0]
basis = np.linalg.inv(basis)
sign = int(sign / basis[0, 0])
basis[:, 0] = sign * basis[:, 0]
X_star = basis.dot(X_train.T)
X_star = X_star.T

fig, axs = plt.subplots(nrows=3, ncols=2)
cdict = {0: 'r', 2: 'g'}
x_sep = -b / np.linalg.norm(w)
for i, ax in enumerate(axs.flat, 1):
    for g in np.unique(y_train):
        ix = np.where(y_train == g)
        if i < 6:
            ax.scatter(X_star[ix, 0], X_star[ix, i], c = cdict[g], label=g)
            ax.set_xlabel('x_new_0')
            ax.set_ylabel(f'x_new_{i}')
            # Draw vertical line.
            ax.axvline(x=0, color='k')
    ax.legend()
plt.show()

# %%
# Plot histograms in a single fig.
db.hist(column=x_var, bins=500)
plt.suptitle(machine)
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