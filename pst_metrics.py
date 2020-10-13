# -*- coding: utf-8 -*-
# @Time : 2020/10/13 10:31 上午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : pst_metrics.py


from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
from pandas import DataFrame


def mutual_info_matrix(df_: DataFrame):
    re = DataFrame(index=df_.columns)

    for i in tqdm(df_.columns):
        re[i] = [normalized_mutual_info_score(df_[i], df_[j]) for j in df_.columns]

    return re