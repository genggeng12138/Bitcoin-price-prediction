from TWSVR import TwinSVMRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import explained_variance_score

def create_data():
    data = pd.read_csv(r'C:\Users\mjgeng\Desktop\比特币及其相关数据\指标筛选后数据.csv')
    # print(data)
    return np.array(data.iloc[0:299, 1:11]),  np.array(data.iloc[300:499, 1:11]), np.array(data.iloc[0:299, 0]), np.array(data.iloc[300:499, 0])
    # 前3-最后列，取第2列,
x_train, x_test, y_train, y_test = create_data()

x1 = np.array([2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
               2**-2, 2**-1, 2, 2**1, 2**2, 2**3, 2**4, 2**5,
               2**6, 2**7, 2**8, 2**9, 2**10])
x2 = np.array([2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
               2**-2, 2**-1, 2, 2**1, 2**2, 2**3, 2**4, 2**5,
               2**6, 2**7, 2**8, 2**9, 2**10])

# x1 = np.linspace(0.001, 1000, 20, endpoint=False)
# x2 = np.linspace(0.001, 1000, 20, endpoint=False)

# def rmse(y, y_hat):
#     rmse_hat = np.linalg.norm(y - y_hat, ord=2) / len(y) ** 0.5
#     return rmse_hat

if __name__ == '__main__':
    evs = np.zeros(shape=(20, 20))
    for i in range(20):
        for j in range(20):
            svr = TwinSVMRegressor(C1=x1[i], C2=x1[i], kernel_type=3, kernel_param=x2[j])
            svr.fit(x_train, y_train)
            y_hat = svr.predict(x_test)
            svr_r2 = explained_variance_score(y_test, y_hat)
            evs[i, j] = svr_r2+0.1
    print(evs)
    x3, x4 = range(20), range(20)
    x3, x4 = np.meshgrid(x3, x4)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x3, x4, evs, rstride=1, cstride=1, cmap="hot")
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.contourf(x3, x4, evs,
                zdir='evs',  # 使用数据方向
                offset=-1,  # 填充投影轮廓位置
                cmap=plt.cm.hot)
    plt.show()
