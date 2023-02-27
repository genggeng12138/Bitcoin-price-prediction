import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from scipy.sparse.linalg import lsmr
import pandas as pd
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

class LSSVR(BaseEstimator, RegressorMixin):

    def __init__(self, C=2.0, kernel='linear', gamma=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y, support=None):

        X, y = check_X_y(X, y, multi_output=True, dtype='float')

        if not support:
            self.support_ = np.ones(X.shape[0], dtype=bool)
        else:
            self.support_ = check_array(support, ensure_2d=False, dtype='bool')

        self.support_vectors_ = X[self.support_, :]
        support_labels = y[self.support_]

        self.K_ = self.kernel_func(X, self.support_vectors_)
        omega = self.K_.copy()
        np.fill_diagonal(omega, omega.diagonal()+self.support_/self.C)

        D = np.empty(np.array(omega.shape) + 1)

        D[1:, 1:] = omega
        D[0, 0] = 0
        D[0, 1:] = 1
        D[1:, 0] = 1

        shape = np.array(support_labels.shape)
        shape[0] += 1
        t = np.empty(shape)

        t[0] = 0
        t[1:] = support_labels

        # TODO: maybe give access to  lsmr atol and btol ?
        try:
            z = lsmr(D.T, t)[0]
        except:
            z = np.linalg.pinv(D).T @ t

        self.bias_ = z[0]
        self.alpha_ = z[1:]
        self.alpha_ = self.alpha_[self.support_]

        return self

    def predict(self, X):
   

        if not hasattr(self, 'support_vectors_'):
            raise NotFittedError

        X = check_array(X, ensure_2d=False)
        K = self.kernel_func(X, self.support_vectors_)
        return (K @ self.alpha_) + self.bias_

    def kernel_func(self, u, v):
        if self.kernel == 'linear':
            return np.dot(u, v.T)

        elif self.kernel == 'rbf':
            return rbf_kernel(u, v, gamma=self.gamma)

        elif callable(self.kernel):
            if hasattr(self.kernel, 'gamma'):
                return self.kernel(u, v, gamma=self.gamma)
            else:
                return self.kernel(u, v)
        else:
            # default to linear
            return np.dot(u, v.T)

    def score(self, X, y):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2

    def norm_weights(self):
        A = self.alpha_.reshape(-1, 1) @ self.alpha_.reshape(-1, 1).T

        W = A @ self.K_[self.support_, :]
        return np.sqrt(np.sum(np.diag(W)))
# 数据
def create_data():
    data = pd.read_csv(r'C:\Users\mjgeng\Desktop\比特币及其相关数据\归一化后数据.csv')
    data.dropna(inplace=True)
    # print(data)
    return np.array(data.iloc[0:299, 1:11]),  np.array(data.iloc[300:499, 1:11]),\
           np.array(data.iloc[0:299, 0]), np.array(data.iloc[300:499, 0])
    # 前3-最后列，取第2列,
X_train, X_test, y_train, y_test = create_data()

if __name__ == '__main__':
    # 选择默认参数
    ls_svm = LSSVR(kernel='rbf')
    ls_svm.fit(X_train, y_train.astype('str'))
    y_hat = ls_svm.predict(X_test)
    print(y_hat)
    print(y_test)
    print(y_hat-y_test)
    evs = explained_variance_score(y_hat, y_test)
    print('evs为:', evs)
    mse = mean_squared_error(y_hat, y_test)
    print('mse为:', mse)
    r2 = r2_score(y_hat, y_test)
    print('r2为:', r2)
    mae = mean_absolute_error(y_hat, y_test)
    print('mae为:', mae)
    mape = mean_absolute_percentage_error(y_hat, y_test)
    print('mape为:', mape)
    rmse = mse**0.5
    print('rmse为:', rmse)
    # y_h = pd.DataFrame(y_test, y_hat)
    # y_h.to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\lssvr-evs.xlsx")
    t = timeit.timeit(stmt='test()', setup="from lssvr import test, LSSVR, create_data",
                      number=100)
    print(t)

def test():
    X_train, X_test, y_train, y_test = create_data()
    ls_svm = LSSVR(kernel='rbf')
    ls_svm.fit(X_train, y_train.astype('str'))
    y_hat = ls_svm.predict(X_test)
    return y_hat



