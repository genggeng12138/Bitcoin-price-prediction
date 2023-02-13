from sklearn.svm import SVR
import timeit
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def create_data():
    data = pd.read_excel(r'C:\Users\mjgeng\Desktop\fujian2.xlsx')
    # print(data)
    return np.array(data.iloc[:, 4:]),  np.array(data.iloc[:, 0])
    # 前3-最后列，取第2列,
X, y = create_data()

x_train, x_test = train_test_split(X, test_size=0.2)
y_train, y_test = train_test_split(y, test_size=0.2)

def create_data1():
    data = pd.read_excel(r'C:\Users\mjgeng\Desktop\111.xlsx')
    # print(data)
    return np.array(data.iloc[:, 0:])
    # 前3-最后列，取第2列,
x = create_data1()


if __name__ == '__main__':
    svr = SVR(C=0.52906546, kernel='rbf', epsilon=0.001)
    svr.fit(x_train, y_train)
    y_hat = svr.predict(x)
    y_h = pd.DataFrame(y_hat)
    print(y_hat)


