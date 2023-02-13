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
from sklearn.metrics import explained_variance_score



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
    data = pd.read_csv(r'C:\Users\mjgeng\Desktop\比特币及其相关数据\归一化后数据.csv')
    # print(data)
    return np.array(data.iloc[0:299, 1:28]),  np.array(data.iloc[300:499, 1:28]), np.array(data.iloc[0:299, 0]), np.array(data.iloc[300:499, 0])
    # 前3-最后列，取第2列,
X_train, X_test, y_train, y_test = create_data()
x_train, x_test, y_train, y_test = create_data()


# 函数
def fun(x1, x2):
    svc = TwinSVMRegressor(C1=x1, C2=x1, kernel_type=3, kernel_param=x2)  # 参数gamma和惩罚参数c
    # cv_scores = cross_val_score(svc, x_train, y_train, cv=3, scoring='accuracy')
    # print(cv_scores.mean())
    svc.fit(x_train, y_train)
    predict = svc.predict(x_test)
    print("得分：", explained_variance_score(y_test, predict))
    return - explained_variance_score(y_test, predict)


# 种群初始化函数

def initial(pop, dim, ub, lb):
    X = np.zeros((pop, dim))
    bound = [lb, ub]
    print(bound)
    for i in range(pop):
        for j in range(dim):
            X[i][j] = np.random.uniform(bound[0][j], bound[1][j])
        X[i] = (X[i, 0], X[i, 1])
    print(X)
    return X, lb, ub


# 边界检查函数


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


# 计算适应度函数


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros((pop, 1))
    for i in range(pop):
        fitness[i] = fun(X[i, 0], X[i, 1])
    return fitness


# 适应度排序


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


# 根据适应度对位置进行排序


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def WOA(pop, dim, lb, ub, Max_iter, fun):
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros((1, dim))
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):

        Leader = X[0, :]  # 领头鲸鱼
        a = 2 - t * (2 / MaxIter)  # 线性下降权重2 - 0
        a2 = -1 + t * (-1 / MaxIter)  # 线性下降权重-1 - -2
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()

            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a2 - 1) * random.random() + 1

            for j in range(dim):

                p = random.random()
                if p < 0.5:
                    if np.abs(A) >= 1:
                        rand_leader_index = min(int(np.floor(pop * random.random() + 1)), pop - 1)
                        X_rand = X[rand_leader_index, :]
                        D_X_rand = np.abs(C * X_rand[j] - X[i, j])
                        X[i, j] = X_rand[j] - A * D_X_rand
                    elif np.abs(A) < 1:
                        D_Leader = np.abs(C * Leader[j] - X[i, j])
                        X[i, j] = Leader[j] - A * D_Leader
                elif p >= 0.5:
                    distance2Leader = np.abs(Leader[j] - X[i, j])
                    X[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j]

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0, :] = (X[0, :])
        Curve[t] = GbestScore
        print(['迭代次数为' + str(t) + ' 的迭代结果' + str(GbestScore)])
        print(['迭代次数为' + str(t) + ' 的最优参数' + str(GbestPositon)])

    return GbestScore, GbestPositon, Curve


# 设置参数
pop = 50  # 种群数量
MaxIter = 50  # 最大迭代次数
dim = 2  # 维度
lb = [0.001, 0.001]   # 下边界
ub = [1000, 1000]   # 上边界

GbestScore, GbestPositon, Curve = WOA(pop, dim, lb, ub, MaxIter, fun)
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.grid()
plt.title('WOA', fontsize='large')
plt.show()
Curve = pd.DataFrame(Curve)
Curve.to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\woatwsvr27.xlsx")