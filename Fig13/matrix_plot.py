import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.rcParams.update({'font.size': 13})
evs = pd.read_excel(r'D:\100.xlsx')
# evs = pd.read_excel(r'D:\svr+matrix.xlsx')
# evs = pd.read_excel(r'D:\twsvr-matrix.xlsx')
evs = np.array(evs)
x3, x4 = np.linspace(-9, 11, 20, endpoint=False), np.linspace(-9, 11, 20, endpoint=False)
x3, x4 = np.meshgrid(x3, x4)
fon = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
        }
fig = plt.figure(figsize=(9, 7))
ax = Axes3D(fig)
# ax = fig.gca(projection='3d')
ax.plot_surface(x3, x4, evs, rstride=1, cstride=1, cmap="hot", vmin=0.0001)
ax.set_zlim3d(-0.5, 1)
ax.set_xlabel('Regularization factor(C)', fon, labelpad=10)
ax.set_ylabel(r'Kernel bandwidth($\gamma$)', fon, labelpad=10)
ax.set_zlabel('Explained variance score', fon, labelpad=10)
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
