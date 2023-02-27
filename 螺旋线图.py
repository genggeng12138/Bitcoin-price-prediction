# #第1步：导入模块
import matplotlib.pyplot as plt
import numpy as np
# #第2步：定义极坐标图
plt.subplot(111, polar=True)
# #第3步：参数定义
# 4个半圈=2个圆=4个180#x角度,100代表平滑度，越大越平滑
N = 4
xtheta = np.arange(0, N * np.pi, np.pi / 100)
# 画图
plt.plot(xtheta, xtheta, color="red")
# 第4步：标题和图片展示

plt.show()
