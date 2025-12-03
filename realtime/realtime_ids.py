import time
import numpy as np
from joblib import load
import os

# Load trained model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(base_dir, '../classical/logistic_model.joblib'))
scaler = load(os.path.join(base_dir, '../classical/normalizer.joblib'))

# Simulate streaming: use 5% of test data
import pandas as pd
testdata = pd.read_csv(os.path.join(base_dir, '../kddtest.csv'), header=None)
test_frac = int(0.05 * len(testdata))
testdata = testdata.iloc[:test_frac]
T = testdata.iloc[:,1:42]
C = testdata.iloc[:,0]

T_scaled = scaler.transform(T)

print('--- Real-Time IDS Simulation (5% of test data) ---')
for i, (sample, label) in enumerate(zip(T_scaled, C)):
    pred = model.predict(sample.reshape(1, -1))[0]
    out = 'Normal' if pred == 0 else 'Attack'
    print(f'[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sample {i+1}: {out}')
    time.sleep(0.5)  # simulate stream, adjust as needed
