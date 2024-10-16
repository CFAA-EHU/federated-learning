#!/usr/bin/env python

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# ====== Create fake dataset =====
# Create random number generator.
rng = np.random.default_rng(seed=123456)
n_classes = 3
v = np.array(
    [[1, 0],
     [-1, 1.2],
     [-0.6, -0.8]]
)

radii = [0.8, 1.2, 0.3]
numbers = [4243, 583, 1034]
points = []
for i, e in enumerate(v):
    mu = rng.normal(loc=e, scale=radii[i], size=(10, 2))
    weights = rng.integers(1, 21, size=10)
    weights = numbers[i] * weights / np.sum(weights)
    # Approximate each weight by the nearest integer.
    weights = np.round(weights).astype(int)
    p = [
        rng.normal(loc=c, scale=0.3, size=(w, 2))
        for c, w in zip(mu, weights)
    ]
    points.append(np.concatenate(p))
    numbers[i] = len(points[-1])

colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(*points[i].T, color=colors[i])
plt.show()

# Concatenate points along axis = 0.
points_ = np.concatenate(points)
labels = []
for i in range(n_classes):
    labels.extend([i] * numbers[i])
labels = np.array(labels)
# Shuffle points and labels.
indices = np.arange(len(points_))
rng.shuffle(indices)

# Save dataset as points and labels in the same CSV.
df = pd.DataFrame(points_[indices], columns=['x', 'y'])
df['label'] = labels[indices]
df.to_csv('fake_dataset_merged.csv', index=False)
del points_, labels

# %%
# ==== Separate dataset into local datasets ====
rng = np.random.default_rng(seed=123456)
n_locals = 4
# Create n_locals empty dataframes with columns x, y, and label.
dfs = [pd.DataFrame(columns=['x', 'y', 'label']) for _ in range(n_locals)]
# Create weights for each local dataset.
weights = rng.integers(1, 51, size=(n_classes, n_locals))
totals = np.sum(weights, axis=1)
for i, r in enumerate(weights):
    r = r / totals[i]
    r = numbers[i] * r
    r = np.round(r).astype(int)
    r[-1] = numbers[i] - np.sum(r[:-1])
    idxs = np.arange(numbers[i])
    rng.shuffle(idxs)
    idxs = np.split(idxs, np.cumsum(r)[:-1])
    for j in range(n_locals):
        df = pd.DataFrame(points[i][idxs[j]], columns=['x', 'y'])
        df['label'] = r[j] * [i]
        dfs[j] = pd.concat([dfs[j], df], ignore_index=True)

# Shuffle each local dataset.
for i in range(n_locals):
    dfs[i] = dfs[i].sample(frac=1)
    dfs[i].to_csv(f'fake_dataset_local_{i}.csv', index=False)

fig, axs = plt.subplots(2, 2)
colors = ['red', 'green', 'blue']
for i, ax in enumerate(axs.flat):
    c = dfs[i]['label'].apply(lambda x: colors[x])
    ax.scatter(dfs[i]['x'], dfs[i]['y'], color=c)
plt.show()