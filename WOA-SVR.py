from sklearn.svm import SVR
import numpy as np
import pandas as pd
import random
import math
from matplotlib import pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn import metrics

def create_data():
    data = pd.read_csv(r'C:\Users\mjgeng\Desktop\比特币及其相关数据\归一化后数据.csv')
    # print(data)
    return np.array(data.iloc[0:299, 1:12]),  np.array(data.iloc[300:499, 1:12]), np.array(data.iloc[0:299, 0]), np.array(data.iloc[300:499, 0])
    # 前3-最后列，取第2列,
x_train, x_test, y_train, y_test = create_data()

# 定义标准
# rmse
def rmse(y, y_hat):
    rmse_hat = np.linalg.norm(y - y_hat, ord=2) / len(y) ** 0.5
    return rmse_hat

# 函数
def fun(x1, x2):
    svr = SVR(C=x1, epsilon=x2, kernel='rbf')  # 参数gamma和惩罚参数c
    # cv_scores = cross_val_score(svc, x_train, y_train, cv=3, scoring='accuracy')
    # print(cv_scores.mean())
    svr.fit(x_train, y_train)
    predict = svr.predict(x_test)
    svr_rmse = explained_variance_score(y_test, predict)
    print("得分：", svr_rmse)
    return - svr_rmse


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
MaxIter = 100  # 最大迭代次数
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
Curve.to_excel(r"C:\Users\mjgeng\Desktop\比特币及其相关数据\woasvr11.xlsx")