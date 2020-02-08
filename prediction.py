from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class Model2D:
    

    def __init__(self, WIDTH=4, period=24, coord_cols=None, vel_cols=None,
                 pred_coord_cols=None, pred_vel_cols=None):
        #
        #   WIDTH - ширина окна предсказаний, то есть то число значений,
        #   которые мы берём вунтри кластера для агрегации по period классам
        #

        if coord_cols is None:
            self.coord_cols = ['x_transformed', 'y_transformed']
        else:
            self.coord_cols = coord_cols

        if vel_cols is None:
            self.vel_cols = ['Vx_transformed', 'Vy_transformed']

        if pred_coord_cols is None:
            self.pred_coord_cols = ['x_pred', 'y_pred']
        else:
            self.pred_coord_cols = pred_coord_cols

        if pred_vel_cols is None:
            self.pred_vel_cols = ['Vx_pred', 'Vy_pred']
        else:
            self.pred_vel_cols = pred_vel_cols

        self.WIDTH = WIDTH
        self.width = WIDTH * period
        self.all_pred_cols = self.pred_coord_cols + self.pred_vel_cols


    
    def predict(self, full_train, full_test, clusters_map):
        #
        #   full_train, full_test - уже трансформированные данные
        #
        full_train['cluster'] = full_train.sat_id.map(clusters_map)
        full_test['cluster'] = full_test.sat_id.map(clusters_map)

        UNIQ = full_test.sat_id.unique()
        num_cols = ['coord_num', 'vel_num']
        for num_col in num_cols:
            full_train[num_col] = 0
            full_test[num_col] = 0

        # enumerating coords
        for sat_id in full_train.sat_id.unique():
            df_train = full_train[full_train.sat_id == sat_id]
            
            df = df_train.iloc[-self.width:]
            if sat_id in UNIQ:
                df_test = full_test[full_test.sat_id == sat_id]
            for trans_col, num_col in zip([self.coord_cols[0], self.vel_cols[0]], num_cols):
                idx_min  = df[trans_col].idxmin()
                idx_last = (len(df_train) + idx_min) % 24
                full_train.loc[df_train.index, num_col] = np.fromfunction(lambda i: (i + idx_min) % 24,
                                                                          (len(df_train), ), dtype=np.int16)
                if sat_id in UNIQ:
                    full_test.loc[df_test.index, num_col] = np.fromfunction(lambda i: (i + idx_last) % 24,
                                                                            (len(df_test), ), dtype=np.int16)

        print('COORDS HAS ENUMERATED')
        for pred_col in self.all_pred_cols:
            full_test[pred_col] = 0.0

        for cluster in set(clusters_map.values()):
            for trans_cols, pred_cols, num_col in zip([self.coord_cols,
                                                       self.vel_cols],
                                                      [self.pred_coord_cols,
                                                       self.pred_vel_cols],
                                                      num_cols):
                df_train = full_train[full_train.cluster == cluster]
                df_test = full_test[full_test.cluster == cluster]
                df_train = df_train[df_train.sat_id.isin(df_test.sat_id.unique())]

                for i in range(24):
                    X_train = np.arange(self.WIDTH, dtype=np.float64)
                    per_train = df_train[df_train[num_col] == i]
                    per_test = df_test[df_test[num_col] == i]
                    y_train = np.zeros(self.WIDTH, dtype=np.float64)
                    y_train_cnt = np.zeros(self.WIDTH, dtype=np.float64)

                    # select the last width of observations for every sat_id
                    for col, pred_col in zip(trans_cols, pred_cols):
                        linear_reg = LinearRegression()
                        for j in range(-1, -self.WIDTH, -1):
                            for sat_id in per_test.sat_id.unique():
                                sat_col_val = per_train.loc[per_train[per_train.sat_id == sat_id].index, col].values
                                if abs(j) <= len(sat_col_val):
                                    y_train[j] += sat_col_val[j]
                                    y_train_cnt[j] += 1

                        # mean aggregation
                        y_train_cnt[y_train_cnt == 0] = 1
                        y_train /= y_train_cnt

                        linear_reg = LinearRegression()
                        linear_reg.fit(X_train.reshape(-1, 1), y_train)

                        for sat_id in per_test.sat_id.unique():
                            X_test = np.arange(len(per_test[per_test.sat_id == sat_id])) + self.WIDTH
                            y_pred = linear_reg.predict(X_test.reshape(-1, 1))
                            full_test.loc[per_test[per_test.sat_id == sat_id].index, pred_col] = y_pred
        return full_test