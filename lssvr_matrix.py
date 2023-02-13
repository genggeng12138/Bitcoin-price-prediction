from lssvr import LSSVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import explained_variance_score
import math

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


def rmse(y, y_hat):
    rmse_hat = np.linalg.norm(y - y_hat, ord=2) / len(y) ** 0.5
    return rmse_hat

if __name__ == '__main__':
    evs = np.zeros(shape=(20, 20))
    for i in range(20):
        for j in range(20):
            svr = LSSVR(C=x1[i], kernel='rbf', gamma=x2[j])
            svr.fit(x_train, y_train)
            y_hat = svr.predict(x_test)
            svr_r2 = explained_variance_score(y_test, y_hat)
            evs[i, j] = svr_r2
    print(evs)
    x3, x4 = np.linspace(-9, 11, 20, endpoint=False), np.linspace(-9, 11, 20, endpoint=False)
    x3, x4 = np.meshgrid(x3, x4)
    fig = plt.figure(figsize=(9, 7))
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    ax.plot_surface(x3, x4, evs, rstride=1, cstride=1, cmap="hot", vmin=0.0001)
    ax.set_zlim3d(-0.5, 1)
    ax.set_xlabel('Regularization factor(C)')
    ax.set_ylabel(r'Kernel bandwidth for RBF kernel($\gamma$)')
    ax.set_zlabel('Explained variance score')
    ax.set_xticks([-9, -5, -1, 3, 7, 10])
    ax.set_xticklabels(['2^-9', '2^-5', '2^-1', '2^3', '2^7', '2^10'])
    ax.set_yticks([-9, -5, -1, 3, 7, 10])
    ax.set_yticklabels(['2^-9', '2^-5', '2^-1', '2^3', '2^7', '2^10'])
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.contourf(x3, x4, evs,
                zdir='evs',  # 使用数据方向
                offset=-0.5,  # 填充投影轮廓位置
                cmap=plt.cm.hot)
    # ax.view_init(32, -32)
    # ax.invert_xaxis()  # x轴反向
    # ax.invert_yaxis()  # y轴反向
    plt.show()
    evs = pd.DataFrame(evs)
    evs.to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\lssvr+matrix.xlsx")