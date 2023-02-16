import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def cost_function(data, theta, y):
    """
    损失函数
    :param data:
    :param theta:
    :param y:
    :return:
    """
    cost = np.sum((data.dot(theta) - y) ** 2)
    return cost / (2 * m)


def gradient(data, theta, y):
    """
    梯度计算函数
    :param data:
    :param theta:
    :param y:
    :return:
    """
    grad = np.empty(len(theta))
    grad[0] = np.sum(data.dot(theta) - y)
    for i in range(1, len(theta)):
        grad[i] = (data.dot(theta) - y).dot(data[:, i])
    return grad


def gradient_descent(data, theta, y, eta):
    """
    循环第4、5步
    :param data:
    :param theta:
    :param y:
    :param eta:
    :return:
    """
    while True:
        last_theta = theta
        grad = gradient(data, theta, y)
        theta = theta - eta * grad
        print(theta)
        # 精度到小数点后15位
        if abs(cost_function(data, last_theta, y) - cost_function(data, theta, y)) < 1e-15:
            break
    return theta


if __name__ == '__main__':
    data = pd.read_csv('data.txt', header=None)  # 两列数据，第一列是X，第二列是Y
    plt.scatter(data[:][0], data[:][1], marker='+')
    data = np.array(data)
    m = data.shape[0]
    data = np.hstack([np.ones([m, 1]), data])  # 对截距增加系数1，方便后面矩阵点乘计算
    y = data[:, 2]  # 取得第二列
    data = data[:, :2]
    # 上面完成对数据清洗、初步加工
    theta = np.array([0, 0])  # 初始线，即斜率和截距都为0的直线
    res = gradient_descent(data, theta, y, 0.0001)
    X = np.arange(3, 25)
    Y = res[0] + res[1] * X
    plt.plot(X, Y, color='r')
    plt.show()
