import pandas as pd
import ast


df = pd.read_csv('./test_2.csv')
total = len(df)
succes = 0
for idx in range(len(df)):
    if type(df['sn2_energies'].values[idx]) == str:
        if len(df['sn2_energies'].values[idx]) > 3:
            succes += 1
print(total, succes, succes / total * 100)