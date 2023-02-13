import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, RegressorMixin
import KernelFunction as kf
import TwinPlane1
import TwinPlane2
import pandas as pd
import random
import math
from matplotlib import pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# __copyright__ = ""
# __license__ = "GPL"
# __version__ = "1.1"
# __maintainer__ = "Arnav Kansal"
# __email__ = "ee1130440@ee.iitd.ac.in"
# __status__ = "Production"

class TwinSVMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, Epsilon1=0.1, Epsilon2=0.1, C1=1, C2=1, kernel_type=0, kernel_param=1, regulz1=0.0001,
                 regulz2=0.0001, _estimator_type="regressor"):
        self.Epsilon1 = float(Epsilon1)
        self.Epsilon2 = float(Epsilon2)
        self.C1 = float(C1)
        self.C2 = float(C2)
        self.regulz1 = float(regulz1)
        self.regulz2 = float(regulz2)
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param

    def fit(self, X, Y):
        Y = pd.Series(Y)
        Y = Y.values.reshape(len(Y), 1)
        assert (type(self.Epsilon1) in [float, int, str])
        assert (type(self.Epsilon2) in [float, int, str])
        assert (type(self.C1) in [float, int, str])
        assert (type(self.C2) in [float, int, str])
        assert (type(self.regulz1) in [float, int, str])
        assert (type(self.regulz2) in [float, int, str])
        assert (self.kernel_type in [0, 1, 2, 3])
        r_x, c = X.shape
        r_y = Y.shape[0]
        assert (r_x == r_y)
        r = r_x

        e = np.ones((r, 1))

        if (self.kernel_type == 0):  # no need to cal kernel
            H = np.hstack((X, e))
        else:
            H = np.zeros((r, r))

            for i in range(r):
                for j in range(r):
                    H[i][j] = kf.kernelfunction(self.kernel_type, X[i], X[j], self.kernel_param)
            H = np.hstack((H, e))

        #####################Calculation of Function Parameters(Equation of planes)
        # print(H)
        [w1, b1] = TwinPlane1.Twin_plane_1(H, Y, self.C1, self.Epsilon1, self.regulz1)
        [w2, b2] = TwinPlane2.Twin_plane_2(H, Y, self.C2, self.Epsilon2, self.regulz2)
        self.plane1_coeff_ = w1
        self.plane1_offset_ = b1
        self.plane2_coeff_ = w2
        self.plane2_offset_ = b2
        self.data_ = X

        np.savetxt("w1.txt", w1)
        np.savetxt("b1.txt", b1)
        np.savetxt("w2.txt", w2)
        np.savetxt("b2.txt", b2)

        return self

    def get_params(self, deep=True):
        return {"Epsilon1": self.Epsilon1, "Epsilon2": self.Epsilon2, "C1": self.C1, "C2": self.C2,
                "regulz1": self.regulz1,
                "regulz2": self.regulz2, "kernel_type": self.kernel_type, "kernel_param": self.kernel_param}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # self.setattr(parameter, value)
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        # X_test = preprocessing.scale(X)
        if (self.kernel_type == 0):  # no need to cal kernel
            S = X
        else:
            S = np.zeros((X.shape[0], self.data_.shape[0]))
            for i in range(X.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = kf.kernelfunction(self.kernel_type, X[i], self.data_[j].T, self.kernel_param)

        y1 = np.dot(S, self.plane1_coeff_) + ((self.plane1_offset_) * (np.ones((X.shape[0], 1))))

        y2 = np.dot(S, self.plane2_coeff_) + ((self.plane2_offset_) * (np.ones((X.shape[0], 1))))

        ###############Compute test data predictions

        return (y1 + y2) / 2

# 数据
def create_data():
    data = pd.read_csv(r'C:\Users\mjgeng\Desktop\doc_2018_CDOM_rastervalue.csv')

    # print(data)
    return np.array(data.iloc[0:19, 4:5]),  np.array(data.iloc[0:19, 4:5]), np.array(data.iloc[0:19, 6]), np.array(data.iloc[0:19, 6])
    # 前3-最后列，取第2列,
X_train, X_test, y_train, y_test = create_data()
x_train, x_test, y_train, y_test = create_data()

if __name__ == '__main__':
    tw_svr = TwinSVMRegressor(C1=6.45522226e-03, C2=6.45522226e-03, kernel_type=3, kernel_param=2.70600616e+01)
    tw_svr.fit(X_train, y_train)
    y_hat = tw_svr.predict(X_test)[:, 0]
    y_h = pd.DataFrame(y_test,y_hat)
    print(y_hat - y_test)
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
    rmse = mse ** 0.5
    print('rmse为:', rmse)
    pd.DataFrame(y_test).to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\y_test.xlsx")
    y_h.to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\TWSVR.xlsx")
    t = timeit.timeit(stmt='test()', setup="from TWSVR import test, TwinSVMRegressor, create_data",
                          number=100)
    print(t)

def test():
    X_train, X_test, y_train, y_test = create_data()
    # tw_svr = TwinSVMRegressor(C1=56, C2=0.011, kernel_type=3)
    tw_svr = TwinSVMRegressor()
    tw_svr.fit(X_train, y_train)
    y_hat = tw_svr.predict(X_test)[:, 0]
    return y_hat
# if __name__ == '__main__':
#     t = timeit.timeit(stmt='test()', setup="from TWSVR import test, TwinSVMRegressor, create_data",
#                       number=100)
#     print(t)
#20s