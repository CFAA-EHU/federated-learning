import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ======================
# Reproducibility
# ======================
np.random.seed(42)

# ======================
# Load training data (6000 aggregated)
# ======================
df_train = pd.read_csv("/home/ubuntu/aprendizaje_federado/mlp6000.csv")

X = df_train[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE',
              'override_SPINDLE', 'powerDrive_SPINDLE']].values
y = df_train['consumo_potencia'].values

# Split (igual que antes)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# (Opcional pero lo mantienes igual que MLP → coherencia)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ======================
# Random Forest model
# ======================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train
rf.fit(X_train_scaled, y_train)

# ======================
# Load TEST SET (global)
# ======================
df_test = pd.read_csv("/home/ubuntu/aprendizaje_federado/test.csv")

X_test = df_test[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE',
                  'override_SPINDLE', 'powerDrive_SPINDLE']].values
y_test = df_test['consumo_potencia'].values

X_test_scaled = scaler.transform(X_test)

# ======================
# Evaluation
# ======================
y_pred = rf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
