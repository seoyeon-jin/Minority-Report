import numpy as np
import pandas as pd

def generate_toy_data(n_samples=1000, seq_len=100):
    data = []
    texts = []
    
    for _ in range(n_samples):
        # 1. Random Factors (Shared Info)
        freq = np.random.choice(['low', 'high'])
        amp = np.random.choice(['small', 'large'])
        
        # 2. Numerical Values
        f_val = 1.0 if freq == 'low' else 5.0
        a_val = 0.5 if amp == 'small' else 2.0
        
        # 3. Private Info (Time-specific noise/phase)
        phase = np.random.uniform(0, 2*np.pi) 
        
        # 4. Generate Time Series
        t = np.linspace(0, 10, seq_len)
        ts = a_val * np.sin(f_val * t + phase) + np.random.normal(0, 0.1, seq_len)
        
        # 5. Generate Text
        text = f"A {freq} frequency wave with {amp} amplitude."
        
        data.append(ts)
        texts.append(text)
        
    return np.array(data), texts

# 실행 및 확인
X, Y = generate_toy_data()
print(f"Time Shape: {X.shape}") # (1000, 100)
print(f"Sample Text: {Y[0]}")   # "A high frequency wave..."