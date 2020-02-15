from sklearn.linear_model import LinearRegression
import numpy as np


class AutoRegression:
    def __init__(self, width=4, degree=1, **kwargs):
        self.width = width
        self.model = LinearRegression(**kwargs)
        self.degree = degree
        
    def fit(self, X):
        X_train = np.empty((len(X) - self.width, self.width), dtype=np.float64)
        for i in range(len(X) - self.width):
            X_train[i, :] = X[i: i + self.width]
        y_train = X[self.width:]
        self.train_last = X[-self.width:].reshape(1, -1)
        self.model.fit(X_train, y_train)
        
    def predict(self, steps=1):
        X_test = self.train_last

        y_pred = np.empty((steps, ), dtype=np.float64)
        for step in range(steps):
            y_pred[step] = self.model.predict(X_test)
            X_test[0, :-1] = X_test[0, 1:]
            X_test[0, -1] = y_pred[step]
        return y_pred
