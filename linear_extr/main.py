import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


full_test = pd.read_csv("test.csv", index_col='id')
full_train = pd.read_csv("train.csv", index_col='id')


for i, col in enumerate(['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']):
    full_train[col + '_num'] = 0
    arr = full_train[col + '_num'].values
    start = 0
    for sat_id in range(600):
        df = full_train[full_train.sat_id == sat_id]
        step = df.shape[0]
        idx_min = df[col].idxmin() % 24
        arr[start: start + step] = np.fromfunction(lambda i: (i + idx_min) % 24, (step, ))
        start += step
    full_train[col + '_num'] = arr

for col_name in ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']:
    full_test[col_name] = 0.0

for i, col in enumerate(['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']):
    full_test[col + '_num'] = 0
    arr = full_test[col + '_num'].values
    start = 0
    for sat_id in full_test.sat_id.unique():
        df = full_test[full_test.sat_id == sat_id]
        step = df.shape[0]
        idx_last = full_train[full_train.sat_id == sat_id][col + '_num'].iloc[-1] + 1
        arr[start: start + step] = np.fromfunction(lambda i: (i + idx_last) % 24, (step, ))
        start += step
    full_test[col + '_num'] = arr


width = 4
for sat_id in full_test.sat_id.unique():
    df1_train = full_train[full_train.sat_id == sat_id]
    df1_test =  full_test[full_test.sat_id == sat_id]
    for col in ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']:
        for i in range(24):
            df2_train = df1_train[df1_train[col + '_sim_num'] == i]
            df2_test = df1_test[df1_test[col + '_sim_num'] == i]
            lin_model = LinearRegression()
            X_train = df2_train.reset_index()['id'].values[-width:].reshape(-1, 1)
            y_train = df2_train[col].values[-width:]
            lin_model.fit(X_train, y_train)
            X_test = df2_test.reset_index()['id'].values.reshape(-1, 1)
            y_test = lin_model.predict(X_test)
            full_test.loc[X_test.ravel(), col] = y_test


# just sending simulated values as the answer
submission = full_test.reset_index()[["id", "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]]
submission.columns = ["id", "x", "y", "z", "Vx", "Vy", "Vz"]
submission.to_csv("submission.csv", index=False)
