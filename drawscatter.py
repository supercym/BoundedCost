# Author: cym

import matplotlib.pyplot as plt
import numpy as np


def draw_scatter(name, cost, value):
    """
    :param n: 点的数量，整数
    :param s:点的大小，整数
    :return: None
    """
    # 加载数据
    name = "".join(list(name)[:-4])
    x = []
    y = []
    for k, v in cost.items():
        x.append(v)
        y.append(value[k])
    # 通过切片获取横坐标x1
    # x1 = data[:, 0]
    # 通过切片获取纵坐标R
    # y1 = data[:, 3]
    # 横坐标x2
    # x2 = np.random.uniform(0, 5, n)
    # 纵坐标y2
    # y2 = np.array([3] * n)
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax1 = fig.add_subplot(1, 1, 1)
    # 设置标题
    # ax1.set_title('Result Analysis')
    # 设置横坐标名称
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    ax1.set_xlabel('node cost', font1)
    # 设置纵坐标名称
    ax1.set_ylabel('node value', font1)
    # 画散点图
    color = '#00CED1'  # 点的颜色
    ax1.scatter(x, y, c=color, marker='o', alpha=0.4)
    # 画直线图
    # ax1.plot(x2, y2, c='b', ls='--')
    # 调整横坐标的上下界
    plt.xlim(xmax=1, xmin=0)
    # 显示

    plt.savefig('./out/' + 'CV_' + name + '.png', dpi=300)
    plt.show()

