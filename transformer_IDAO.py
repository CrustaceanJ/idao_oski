import numpy as np
import pandas as pd
from random import shuffle
np.random.seed(31415)


class TransformerIDAO:
    def __init__(self):
        pass

        
    def __get_plane_coefs(self, df, train_coefs, col1, col2, col3):
        for sat_id in train_coefs.sat_id:
            df_part = df[df.sat_id == sat_id]
            n = 0; i = 0

            a = np.zeros(df_part.shape[0] // 3); b = np.zeros(df_part.shape[0] // 3)
            c = np.zeros(df_part.shape[0] // 3); d = np.zeros(df_part.shape[0] // 3)

            p1 = df_part[[col1, col2, col3]].sample(frac=1).values
            p2 = df_part[[col1, col2, col3]].sample(frac=1).values
            p3 = df_part[[col1, col2, col3]].sample(frac=1).values

            v1 = p3 - p1
            v2 = p2 - p1

            # the cross product is a vector normal to the plane
            cp = np.cross(v1, v2)
            sgn_ = 2 * (cp[:, 0] >= 0) - 1
            cp *= sgn_.reshape(-1, 1)
            a, b, c = cp[:, 0], cp[:, 1], cp[:, 2]

            norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)

            a /= norm
            b /= norm
            c /= norm

            d = -(a * p1[:, 0] + b * p1[:, 1] + c * p1[:, 2])

            train_coefs.loc[self.train_coefs.sat_id == sat_id, 'A'] = np.nanmedian(a)
            train_coefs.loc[self.train_coefs.sat_id == sat_id, 'B'] = np.nanmedian(b)
            train_coefs.loc[self.train_coefs.sat_id == sat_id, 'C'] = np.nanmedian(c)
            train_coefs.loc[self.train_coefs.sat_id == sat_id, 'D'] = np.nanmedian(d)
        
        return train_coefs
        
        
    def fit_transform(self, df, train=True, train_coefs_V=None):
        if train:
            test_sz = df.sat_id.nunique()
            self.train_coefs = pd.DataFrame({'sat_id': df.sat_id.unique(), 'A': np.zeros(test_sz),
                               'B': np.zeros(test_sz), 'C': np.zeros(test_sz), 'D': np.zeros(test_sz)})
            self.train_coefs = self.__get_plane_coefs(df, self.train_coefs, 'x', 'y', 'z')
        self.train_coefs_V = train_coefs_V
        if self.train_coefs_V is not None:
            test_sz = df.sat_id.nunique()
            self.train_coefs_V = pd.DataFrame({'sat_id': df.sat_id.unique(), 'A': np.zeros(test_sz),
                               'B': np.zeros(test_sz), 'C': np.zeros(test_sz), 'D': np.zeros(test_sz)})
            self.train_coefs_V = self.__get_plane_coefs(df, self.train_coefs_V, 'Vx', 'Vy', 'Vz')
 
        
        
        df['x_transformed'] = df['x']; df['y_transformed'] = df['y']; df['z_transformed'] = df['z']
        df['Vx_transformed'] = df['Vx']; df['Vy_transformed'] = df['Vy']; df['Vz_transformed'] = df['Vz']
        # TRANSFORM in 3d (new_coord)
        for sat_id in df.sat_id.unique():
            A, B, C, D = self.train_coefs[self.train_coefs.sat_id == sat_id][['A', 'B', 'C', 'D']].values[0]
            d = np.sqrt(B ** 2 + C ** 2)
    #         cos_alpha = C / d; sin_alpha = B / d
    #         cos_beta = d / np.sqrt(A ** 2 + d ** 2); sin_beta = A / np.sqrt(A ** 2 + d ** 2)

            # My(-beta) * Mx(alpha)
            M = [[d/np.sqrt(A ** 2 + d ** 2), -B*A/d/ np.sqrt(A ** 2 + d ** 2), -C*A/d/ np.sqrt(A ** 2 + d ** 2)],
                 [0, C/d, -B/d],
                 [A / np.sqrt(A ** 2 + d ** 2), B / np.sqrt(A ** 2 + d ** 2), C / np.sqrt(A ** 2 + d ** 2)]]

            sz = df[df.sat_id == sat_id].shape[0]

            coord = np.dot(M, (df[df.sat_id == sat_id][['x_transformed', 'y_transformed', 'z_transformed']].values.T \
                        + np.array([np.zeros(sz), np.zeros(sz), -D/C * np.ones(sz)]))).T
            df.loc[df.sat_id == sat_id, 'x_transformed'] = coord[:, 0]
            df.loc[df.sat_id == sat_id, 'y_transformed'] = coord[:, 1]
            df.loc[df.sat_id == sat_id, 'z_transformed'] = coord[:, 2]

            if self.train_coefs_V is not None:
                A, B, C, D = self.train_coefs_V[self.train_coefs_V.sat_id == sat_id][['A', 'B', 'C', 'D']].values[0]
                d = np.sqrt(B ** 2 + C ** 2)

                M = [[d/np.sqrt(A ** 2 + d ** 2), -B*A/d/ np.sqrt(A ** 2 + d ** 2), -C*A/d/ np.sqrt(A ** 2 + d ** 2)],
                     [0, C/d, -B/d],
                     [A / np.sqrt(A ** 2 + d ** 2), B / np.sqrt(A ** 2 + d ** 2), C / np.sqrt(A ** 2 + d ** 2)]]

            velocity = np.dot(M, (df[df.sat_id == sat_id][['Vx_transformed', 'Vy_transformed', 'Vz_transformed']].values.T \
                        + np.array([np.zeros(sz), np.zeros(sz), -D/C * np.ones(sz)]))).T
            df.loc[df.sat_id == sat_id, 'Vx_transformed'] = velocity[:, 0]
            df.loc[df.sat_id == sat_id, 'Vy_transformed'] = velocity[:, 1]
            df.loc[df.sat_id == sat_id, 'Vz_transformed'] = velocity[:, 2]        

        # TRANSFORM in 2d (shift_to_center_and_rotate)
        if train:
            self.train_coefs['xc'] = 0.0; self.train_coefs['yc'] = 0.0

        if (self.train_coefs_V is not None and train):
            self.train_coefs_V['xc'] = 0.0; self.train_coefs_V['yc'] = 0.0
        #shift
        for sat_id in df.sat_id.unique():
            if train:
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'xc'] = (df[df.sat_id == sat_id].x_transformed.max() + df[df.sat_id == sat_id].x_transformed.min()) / 2
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'yc'] = (df[df.sat_id == sat_id].y_transformed.max() + df[df.sat_id == sat_id].y_transformed.min()) / 2

            df.loc[df.sat_id == sat_id, 'x_transformed'] = df[df.sat_id == sat_id].x_transformed.values - self.train_coefs[self.train_coefs.sat_id == sat_id].xc.values
            df.loc[df.sat_id == sat_id, 'y_transformed'] = df[df.sat_id == sat_id].y_transformed.values - self.train_coefs[self.train_coefs.sat_id == sat_id].yc.values

            if self.train_coefs_V is None:
                df.loc[df.sat_id == sat_id, 'Vx_transformed'] = df[df.sat_id == sat_id].Vx_transformed.values - self.train_coefs[self.train_coefs.sat_id == sat_id].xc.values
                df.loc[df.sat_id == sat_id, 'Vy_transformed'] = df[df.sat_id == sat_id].Vy_transformed.values - self.train_coefs[self.train_coefs.sat_id == sat_id].yc.values
            else:
                if train:
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'xc'] = (df[df.sat_id == sat_id].Vx_transformed.max() + df[df.sat_id == sat_id].Vx_transformed.min()) / 2
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'yc'] = (df[df.sat_id == sat_id].Vy_transformed.max() + df[df.sat_id == sat_id].Vy_transformed.min()) / 2  

                df.loc[df.sat_id == sat_id, 'Vx_transformed'] = df[df.sat_id == sat_id].Vx_transformed.values - self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].xc.values
                df.loc[df.sat_id == sat_id, 'Vy_transformed'] = df[df.sat_id == sat_id].Vy_transformed.values - self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].yc.values            

        #rotate
        if train:
            self.train_coefs['small_polyos'] = 0.0; self.train_coefs['big_polyos'] = 0.0 

        if (self.train_coefs_V is not None and train):
            self.train_coefs_V['small_polyos'] = 0.0; self.train_coefs_V['big_polyos'] = 0.0 

        for sat_id in df.sat_id.unique():
            if train:
                distances = np.sqrt((df[df.sat_id == sat_id].x_transformed.values) ** 2 + (df[df.sat_id == sat_id].y_transformed.values) ** 2)
                id_min = np.argmin(distances); id_max = np.argmax(distances)
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'small_polyos'] = distances[id_min]
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'big_polyos'] = distances[id_max]

                a = df.loc[df[df.sat_id == sat_id].index[0] + id_max][['x_transformed', 'y_transformed']].values #vector where big polyos
                cos_teta =  a[0] / np.sqrt(a[0] ** 2 + a[1] ** 2)
                sin_teta = a[1] / np.sqrt(a[0] ** 2 + a[1] ** 2)
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'cos_teta'] = cos_teta
                self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'sin_teta'] = sin_teta
            else:
                cos_teta = self.train_coefs[self.train_coefs.sat_id == sat_id].cos_teta.values[0]
                sin_teta = self.train_coefs[self.train_coefs.sat_id == sat_id].sin_teta.values[0]

            M = [[cos_teta, sin_teta],
                [-sin_teta, cos_teta]]
            coord = (np.dot(M, df[df.sat_id == sat_id][['x_transformed', 'y_transformed']].values.T)).T
            df.loc[df.sat_id == sat_id, 'x_transformed'] = coord[:, 0]
            df.loc[df.sat_id == sat_id, 'y_transformed'] = coord[:, 1]

            if self.train_coefs_V is not None:
                if train:
                    distances = np.sqrt((df[df.sat_id == sat_id].Vx_transformed.values) ** 2 + (df[df.sat_id == sat_id].Vy_transformed.values) ** 2)
                    id_min = np.argmin(distances); id_max = np.argmax(distances)
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'small_polyos'] = distances[id_min]
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'big_polyos'] = distances[id_max]

                    a = df.loc[df[df.sat_id == sat_id].index[0] + id_max][['Vx_transformed', 'Vy_transformed']].values #vector where big polyos
                    cos_teta =  a[0] / np.sqrt(a[0] ** 2 + a[1] ** 2)
                    sin_teta = a[1] / np.sqrt(a[0] ** 2 + a[1] ** 2)
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'cos_teta'] = cos_teta
                    self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'sin_teta'] = sin_teta
                else:
                    cos_teta = self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].cos_teta.values[0]
                    sin_teta = self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].sin_teta.values[0]

                M = [[cos_teta, sin_teta],
                [-sin_teta, cos_teta]]

            velocity = (np.dot(M, df[df.sat_id == sat_id][['Vx_transformed', 'Vy_transformed']].values.T)).T
            df.loc[df.sat_id == sat_id, 'Vx_transformed'] = velocity[:, 0]
            df.loc[df.sat_id == sat_id, 'Vy_transformed'] = velocity[:, 1]

        return df, self.train_coefs
    
    
    
    def inv_transform(self, df):
        
        # INVERSE TRANSFORM in 2d (inv_shift_to_center_and_rotate)
        #rotate
        df['x_pred_transformed'] = df['x_pred']; df['y_pred_transformed'] = df['y_pred']; df['z_pred_transformed'] = df['z_pred']
        df['Vx_pred_transformed'] = df['Vx_pred']; df['Vy_pred_transformed'] = df['Vy_pred']; df['Vz_pred_transformed'] = df['Vz_pred']
        for sat_id in df.sat_id.unique():     
            cos_teta =  self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'cos_teta'].values[0]
            sin_teta = self.train_coefs.loc[self.train_coefs.sat_id == sat_id, 'sin_teta'].values[0]

            M = [[cos_teta, -sin_teta],
                [sin_teta, cos_teta]]
            coord = (np.dot(M, df[df.sat_id == sat_id][['x_pred_transformed', 'y_pred_transformed']].values.T)).T
            df.loc[df.sat_id == sat_id, 'x_pred_transformed'] = coord[:, 0]
            df.loc[df.sat_id == sat_id, 'y_pred_transformed'] = coord[:, 1]

            if self.train_coefs_V is not None:
                cos_teta =  self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'cos_teta'].values[0]
                sin_teta = self.train_coefs_V.loc[self.train_coefs_V.sat_id == sat_id, 'sin_teta'].values[0]

                M = [[cos_teta, -sin_teta],
                    [sin_teta, cos_teta]]

            velocity = (np.dot(M, df[df.sat_id == sat_id][['Vx_pred_transformed', 'Vy_pred_transformed']].values.T)).T
            df.loc[df.sat_id == sat_id, 'Vx_pred_transformed'] = velocity[:, 0]
            df.loc[df.sat_id == sat_id, 'Vy_pred_transformed'] = velocity[:, 1]            

        #shift
        for sat_id in df.sat_id.unique():
            df.loc[df.sat_id == sat_id, 'x_pred_transformed'] = df[df.sat_id == sat_id].x_pred_transformed.values + self.train_coefs[self.train_coefs.sat_id == sat_id].xc.values
            df.loc[df.sat_id == sat_id, 'y_pred_transformed'] = df[df.sat_id == sat_id].y_pred_transformed.values + self.train_coefs[self.train_coefs.sat_id == sat_id].yc.values

            if self.train_coefs_V is None:
                df.loc[df.sat_id == sat_id, 'Vx_pred_transformed'] = df[df.sat_id == sat_id].Vx_pred_transformed.values + self.train_coefs[self.train_coefs.sat_id == sat_id].xc.values
                df.loc[df.sat_id == sat_id, 'Vy_pred_transformed'] = df[df.sat_id == sat_id].Vy_pred_transformed.values + self.train_coefs[self.train_coefs.sat_id == sat_id].yc.values
            else:
                df.loc[df.sat_id == sat_id, 'Vx_pred_transformed'] = df[df.sat_id == sat_id].Vx_pred_transformed.values + self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].xc.values
                df.loc[df.sat_id == sat_id, 'Vy_pred_transformed'] = df[df.sat_id == sat_id].Vy_pred_transformed.values + self.train_coefs_V[self.train_coefs_V.sat_id == sat_id].yc.values

        # INVERSE TRANSFORM in 3d (inv_new_coord)
        for sat_id in df.sat_id.unique():
            A, B, C, D = self.train_coefs[self.train_coefs.sat_id == sat_id][['A', 'B', 'C', 'D']].values[0]
            d = np.sqrt(B ** 2 + C ** 2)

            # Mx(-alpha) * My(beta)
            M = [[d/np.sqrt(A ** 2 + d ** 2), 0, A / np.sqrt(A ** 2 + d ** 2)],
                 [-B*A/d/ np.sqrt(A ** 2 + d ** 2), C/d, B / np.sqrt(A ** 2 + d ** 2)],
                 [-C*A/d/ np.sqrt(A ** 2 + d ** 2), -B/d, C / np.sqrt(A ** 2 + d ** 2)]]        

            sz = df[df.sat_id == sat_id].shape[0]

            coord = (np.dot(M, df[df.sat_id == sat_id][['x_pred_transformed', 'y_pred_transformed', 'z_pred_transformed']].values.T) \
                    + np.array([np.zeros(sz), np.zeros(sz), D/C * np.ones(sz)])).T
            df.loc[df.sat_id == sat_id, 'x_pred_transformed'] = coord[:, 0]
            df.loc[df.sat_id == sat_id, 'y_pred_transformed'] = coord[:, 1]
            df.loc[df.sat_id == sat_id, 'z_pred_transformed'] = coord[:, 2]

            if self.train_coefs_V is not None:
                A, B, C, D = self.train_coefs_V[self.train_coefs_V.sat_id == sat_id][['A', 'B', 'C', 'D']].values[0]
                d = np.sqrt(B ** 2 + C ** 2)

                # Mx(-alpha) * My(beta)
                M = [[d/np.sqrt(A ** 2 + d ** 2), 0, A / np.sqrt(A ** 2 + d ** 2)],
                     [-B*A/d/ np.sqrt(A ** 2 + d ** 2), C/d, B / np.sqrt(A ** 2 + d ** 2)],
                     [-C*A/d/ np.sqrt(A ** 2 + d ** 2), -B/d, C / np.sqrt(A ** 2 + d ** 2)]]   

            velocity = (np.dot(M, df[df.sat_id == sat_id][['Vx_pred_transformed', 'Vy_pred_transformed', 'Vz_pred_transformed']].values.T) \
                    + np.array([np.zeros(sz), np.zeros(sz), D/C * np.ones(sz)])).T
            df.loc[df.sat_id == sat_id, 'Vx_pred_transformed'] = velocity[:, 0]
            df.loc[df.sat_id == sat_id, 'Vy_pred_transformed'] = velocity[:, 1]
            df.loc[df.sat_id == sat_id, 'Vz_pred_transformed'] = velocity[:, 2]            

        return df
