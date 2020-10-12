# -*- coding: utf-8 -*-
# @Time : 2020/10/10 2:31 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : plot.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


figure_size = (9, 6)

def hist_distribute(x: pd.Series, title: str, nbin=10):
    '''
    :param x: pandas series
    :param title: plot name
    :return: matplot figure
    '''
    a = plt.figure(figsize=figure_size)
    a = x.hist(color=sns.desaturate("indianred", .8), bins=nbin).get_figure()
    plt.title(title)
    plt.close('all')
    return a


def hist_distribute_with_target(df_: pd.DataFrame, x: str, target: str):
    '''
    :param df_: pandas dataframe
    :param x: columns name
    :param target: target columns name
    :return: figure
    '''
    a = plt.figure(figsize=figure_size)
    bins = len(df_[x].unique()) if len(df_[x].unique()) < 10 else 10
    df_[str(x) + '_bins'] = pd.cut(df_[x], bins=bins, right=False)
    a = df_.groupby([str(x) + '_bins', target]).size().unstack().plot(kind='bar', stacked=False).get_figure()
    plt.title(x)
    plt.close('all')
    return a


def hist_distribute_with_target_norminal(df_: pd.DataFrame, x: str, target: str):
    '''
    :param df_: pandas dataframe
    :param x: columns name
    :param target: target columns name
    :return: figure
    '''
    a = plt.figure(figsize=figure_size)
    a = df_.groupby([x, target]).size().unstack().plot(kind='bar', stacked=False).get_figure()
    plt.title(x)
    plt.close('all')
    return a


def cap_plot(predictions, labels, cut_point=100):
    a = plt.figure(figsize=figure_size)
    sample_size = len(labels)
    bad_label_size = len([i for i in labels if i == 1])
    score_thres = np.linspace(1, 0, cut_point)
    x_list = []
    y_list = []
    for thres in score_thres:
        # 阈值以上的样本数 / 总样本数
        x = len([i for i in predictions if i >= thres])
        x_list.append(x / sample_size)
        # 阈值以上的样本真实为坏客户的样本数 / 总坏客户样本数
        y = len([(i, j) for i, j in zip(predictions, labels) if i >= thres and j == 1])
        y_list.append(y / bad_label_size)

    # 绘制实际曲线
    plt.plot(x_list, y_list, color="green", label="实际曲线")

    # 绘制最优曲线
    best_point = [bad_label_size / sample_size, 1]
    plt.plot([0, best_point[0], 1], [0, best_point[1], 1], color="red", label="最优曲线", zorder=10)
    # 增加最优情况的点的坐标
    plt.scatter(best_point[0], 1, color="white", edgecolors="red", s=30, zorder=30)
    plt.text(best_point[0] + 0.1, 0.95, "{}/{},{}".format(bad_label_size, sample_size, 1), ha="center", va="center")

    # 随机曲线
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="随机曲线")

    # 颜色填充
    plt.fill_between(x_list, y_list, x_list, color="yellow", alpha=0.3)
    plt.fill_between(x_list,
                     [1 if i * sample_size / bad_label_size >= 1 else i * sample_size / bad_label_size for i in x_list],
                     y_list, color="gray", alpha=0.3)

    # 计算AR值
    # 实际曲线下面积
    actual_area = np.trapz(y_list, x_list) - 1 * 1 / 2
    best_area = 1 * 1 / 2 - 1 * bad_label_size / sample_size / 2
    ar_value = actual_area / best_area
    plt.title("CAP曲线 AR={:.3f}".format(ar_value))

    plt.legend(loc=4)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
    plt.close('all')

    return  a


def hist_value(x, name, nbin=10, xname='档位', yname='数量'):
    '''
        画连续型数值频率直方图
        @param x:
        @param nbin:
        @param name:
        @return:
        '''
    a = plt.figure(figsize=(9, 6))
    x = np.array(x, dtype=float)
    plt.title(name)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.hist(x, nbin, color=sns.desaturate("indianred", .8), alpha=.4)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
    plt.close('all')
    return a


def hist_bin(x, name):
    '''
    画离散型的频率直方图
    @param x:
    @param name:
    @return:
    '''

    a = plt.figure(figsize=(9, 6))
    x_counts = x.value_counts(dropna=False)
    x2 = x_counts.index.to_list()
    y = x_counts.to_list()
    rect_width = 0.8 if x_counts.shape[0] < 10 else 0.4

    plt.xlabel('取值')
    plt.ylabel('数量')
    plt.title(name)
    plt.bar(x=range(len(x2)), height=y, width=rect_width, edgecolor='k', color=sns.desaturate("indianred", .8),
            linewidth=0.5, yerr=0.000001, align="center")
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
    plt.xticks(range(len(x2)), x2)
    plt.close('all')
    return a



