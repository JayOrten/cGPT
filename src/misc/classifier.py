from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

# Load data
path = Path("/home/jo288/nobackup/autodelete/cGPT/runs/109/logit_pairs.csv")
df = pd.read_csv(path)

# Name column 1 logits and column 2 labels
df.columns = ['logits', 'labels']

# Get X, y
# X should be an array of integers, not a string
X = df['logits'].values

X = np.array([np.fromstring(x[1:-1], sep=', ') for x in X])

y = df['labels'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(solver='sag', max_iter=10000)
model.fit(X_train, y_train)
